from pathlib import Path
import pymupdf as fitz,sys,json,hashlib,copy,importlib.util
sys.stdout.reconfigure(encoding='utf-8')
BASE=Path(__file__).parent
APP_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(APP_ROOT))
APP=APP_ROOT/'app'/'ocr_pdf_archive.py'
from datetime import date
QA=APP_ROOT/'開発'/'作業中'/(date.today().isoformat()+'_ScanLight検証')/'pdfs'
QA.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('archive_under_test',APP);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=APP_ROOT/'成果物'/'ScanLight'/'pdf'/'編集情報保持_最小サイズ'
service=module.OcrPdfArchive(QA/'empty_ocr',root,QA/'cache')
results=[]
for folder in root.iterdir():
 if not folder.is_dir():continue
 receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
 opened=service.open_editor(receipt['archive_id'])
 assert len(opened['pages'])==receipt['page_count']
 results.append({'name':folder.name,'app_open_editor':'passed','restored_pages':len(opened['pages'])})
 print('編集画面の復元確認: '+folder.name,flush=True)
# Run real edit/save/reopen on isolated representative pages, for JPEG and CCITT.
for name in ['大いなる聖戦 上','大いなる聖戦 下']:
 folder=root/name;receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
 qa=QA/('再編集テスト_'+name);qa.mkdir(exist_ok=True)
 target=qa/receipt['output_name']
 with fitz.open(folder/receipt['output_name']) as d:
  meta=module.read_edit_data(d)
  key=next(k for k,v in meta['pages'].items() if v.get('body_text') and any(r.get('type') in (None,'body','paragraph') for r in v.get('regions',[])))
  page=copy.deepcopy(meta['pages'][key]);records=meta['streams'][key]
  d.select([int(key)-1])
  if module.ATTACHMENT in d.embfile_names():d.embfile_del(module.ATTACHMENT)
  meta['page_count']=1;meta['pages']={'1':page};meta['streams']={'1':records}
  d.embfile_add(module.ATTACHMENT,module.encode(meta),filename=module.ATTACHMENT)
  d.save(target,garbage=1,deflate=True)
 receipt.update(pdf_sha256=module.digest(target.read_bytes()),page_count=1,total_pages=1,text_page_count=1,editing_only=True,inventory={})
 (qa/'receipt.json').write_bytes(module.encode(receipt))
 svc=module.OcrPdfArchive(QA/'empty_ocr',qa.parent,QA/'cache')
 # Give the test an independent root to avoid colliding IDs from other test copies.
 svc.archive_dir=qa.parent
 opened=svc.open_editor(receipt['archive_id'])
 edited=copy.deepcopy(opened['pages']['1']);marker='圧縮編集確認'
 edited['body_text']+=marker
 assert marker in ''.join(r['text'] for r in module.make_runs(edited))
 save=svc.save_editor(receipt['archive_id'],{'1':edited})
 assert save['operation']=='updated'
 reopened=svc.open_editor(receipt['archive_id'])
 assert reopened['pages']['1']['body_text'].endswith(marker)
 with fitz.open(target) as d:
  assert marker in module.clean_text(d[0].get_text())
 results.append({'name':name,'isolated_page_edit_save_reopen':'passed','source_page':int(key)})
 print('本文変更・PDF保存・再復元確認: '+name,flush=True)
(root/'アプリ再編集検証.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
