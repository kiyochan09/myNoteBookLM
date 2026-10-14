from pathlib import Path
import pymupdf as fitz
from PIL import Image
import io,json,hashlib,shutil,sys,collections
sys.stdout.reconfigure(encoding='utf-8')
APP_ROOT=Path(__file__).resolve().parents[2]
ROOT=APP_ROOT/'data'/'ocr_archives'
OUT=APP_ROOT/'成果物'/'ScanLight'/'pdf'/'編集情報保持_最小サイズ'
from datetime import date
QA=APP_ROOT/'開発'/'作業中'/(date.today().isoformat()+'_ScanLight検証')/'pdfs'
ATTACHMENT='MyNotebookLM-OCR-edit-v1.json'
def sha(b): return hashlib.sha256(b).hexdigest()
def text_hash(page):
    return sha(page.get_text('text').encode('utf-8'))
def verify_streams(d,meta):
    n=0
    for key,records in meta['streams'].items():
        contents=d[int(key)-1].get_contents()
        for rec in records:
            assert rec['xref'] in contents,(key,'xref')
            assert sha(d.xref_stream(rec['xref']))==rec['sha256'],(key,'hash')
            n+=1
    return n
def page_geometry(p): return (tuple(p.mediabox),tuple(p.cropbox),p.rotation)
OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(parents=True,exist_ok=True)
summary=json.loads((OUT/'検証結果.json').read_text(encoding='utf-8'))[:1] if '--bilevel-only' in sys.argv else []
for source in ROOT.rglob('*.pdf'):
    if '--bilevel-only' in sys.argv and source.parent.name=='大いなる聖戦 上':continue
    folder=OUT/source.parent.name;folder.mkdir(exist_ok=True)
    dest=folder/source.name
    print('開始: '+source.name,flush=True)
    original_hash=sha(source.read_bytes())
    with fitz.open(source) as d:
        attachments={name:d.embfile_get(name) for name in d.embfile_names()}
        meta=json.loads(attachments[ATTACHMENT]);before_streams=verify_streams(d,meta)
        before_text=[text_hash(p) for p in d]
        geometry=[page_geometry(p) for p in d]
        images={};changed=0;skipped=0
        for page in d:
            for im in page.get_images(full=True):
                if im[0]:images.setdefault(im[0],page.number)
        for count,(xref,number) in enumerate(images.items(),1):
            if count%50==0: print(f'  画像 {count}/{len(images)}',flush=True)
            # Masked images must keep their original alpha/decoding semantics.
            if any(d.xref_get_key(xref,k)[0]!='null' for k in ('SMask','Mask')):
                skipped+=1;continue
            pix=fitz.Pixmap(d,xref)
            if pix.alpha:skipped+=1;continue
            if pix.colorspace.n not in (1,3): pix=fitz.Pixmap(fitz.csRGB,pix)
            im=Image.frombytes('L' if pix.colorspace.n==1 else 'RGB',(pix.width,pix.height),pix.samples)
            original_im=im
            boxes=d[number].get_image_rects(xref)
            if not boxes:skipped+=1;continue
            # Never upscale. Shared images retain enough pixels for their largest placement.
            scale=max(max(b.width*96/72/im.width,b.height*96/72/im.height) for b in boxes)
            scale=min(1,scale)
            if scale<1: im=im.resize((max(1,round(im.width*scale)),max(1,round(im.height*scale))),Image.Resampling.LANCZOS)
            buf=io.BytesIO();im.save(buf,format='JPEG',quality=45,optimize=True)
            data=buf.getvalue();encoding='jpeg'
            # Already-bilevel scans need fax compression, not noisy JPEG edges.
            if d.xref_get_key(xref,'BitsPerComponent')==('int','1') and original_im.mode=='L':
                mono_scale=min(1,max(max(b.width*150/72/original_im.width,b.height*150/72/original_im.height) for b in boxes))
                mono=original_im.resize((max(1,round(original_im.width*mono_scale)),max(1,round(original_im.height*mono_scale))),Image.Resampling.LANCZOS).point(lambda v:255 if v>150 else 0,mode='1')
                tiff=io.BytesIO();mono.save(tiff,format='TIFF',compression='group4',strip_size=1024*1024*1024)
                encoded=tiff.getvalue();tags=Image.open(io.BytesIO(encoded)).tag_v2
                if len(tags[273])==1:
                    g4=encoded[tags[273][0]:tags[273][0]+tags[279][0]]
                    if len(g4)+200<len(data):data=g4;im=mono;encoding='group4'
            if len(data)+200>=len(d.xref_stream_raw(xref)):skipped+=1;continue
            # Modify only this image's object, preserving every page/text xref.
            colorspace='/DeviceGray' if im.mode in ('L','1') else '/DeviceRGB'
            d.update_object(xref,f'<< /Type /XObject /Subtype /Image /Width {im.width} /Height {im.height} /ColorSpace {colorspace} /BitsPerComponent {1 if encoding=="group4" else 8} >>')
            d.update_stream(xref,data,compress=False)
            d.xref_set_key(xref,'Filter','/CCITTFaxDecode' if encoding=='group4' else '/DCTDecode')
            if encoding=='group4':
                d.xref_set_key(xref,'DecodeParms',f'<< /K -1 /Columns {im.width} /Rows {im.height} /BlackIs1 true >>')
                assert fitz.Pixmap(d,xref).samples==im.convert('L').tobytes(),'bilevel encoding verification'
            changed+=1
        verify_streams(d,meta)
        d.save(dest,garbage=1,deflate=True)
    if dest.stat().st_size>=source.stat().st_size:shutil.copyfile(source,dest)
    with fitz.open(dest) as result:
        assert len(result)==len(before_text)
        assert {n:result.embfile_get(n) for n in result.embfile_names()}==attachments
        assert [page_geometry(p) for p in result]==geometry
        assert [text_hash(p) for p in result]==before_text
        assert verify_streams(result,meta)==before_streams
        # Pick a dense body page for later visual review.
        sample=max(range(len(result)),key=lambda i:len(result[i].get_text()))
        pix=result[sample].get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False)
        pix.save(QA/(source.parent.name+'_圧縮後.png'))
        with fitz.open(source) as original:
            original[sample].get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(QA/(source.parent.name+'_圧縮前.png'))
        pages=len(result)
    receipt=json.loads((source.parent/'receipt.json').read_text(encoding='utf-8'))
    # Copies are editing-only and cannot authorize deletion of any source OCR data.
    receipt.update(pdf_sha256=sha(dest.read_bytes()),editing_only=True,inventory={},compression={'jpeg_dpi':96,'bilevel_dpi':150,'jpeg_quality':45,'original_sha256':original_hash,'text_and_edit_data_preserved':True})
    (folder/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    assert sha(source.read_bytes())==original_hash
    item={'name':source.parent.name,'pages':pages,'before_bytes':source.stat().st_size,'after_bytes':dest.stat().st_size,'reduction_percent':round((1-dest.stat().st_size/source.stat().st_size)*100,1),'images_changed':changed,'images_kept':skipped,'text_streams_verified':before_streams,'edit_pages':len(meta['pages']),'visual_sample_page':sample+1,'output':str(dest),'source_unchanged':True}
    summary.append(item)
    (OUT/'検証結果.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(item,ensure_ascii=False),flush=True)
print('全件完了',flush=True)
