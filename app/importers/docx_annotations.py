"""Extract Word footnotes, endnotes and comments, excluding separator records."""
import xml.etree.ElementTree as ET
import re
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS={'w':W}
KINDS={'footnote':('footnotes','脚注','footnoteReference'),'endnote':('endnotes','文末脚注','endnoteReference'),'comment':('comments','コメント','commentReference')}

def extract_docx_annotations(archive,doc_id):
    roots=[ET.fromstring(archive.read(p)) for p in archive.namelist() if re.fullmatch(r'word/(document|header\d+|footer\d+)\.xml',p)]
    result=[]
    for kind,(part,label,reference) in KINDS.items():
        path=f'word/{part}.xml'
        if path not in archive.namelist():continue
        referenced={e.get(f'{{{W}}}id') for body in roots for e in body.findall(f'.//w:{reference}',NS)}
        if kind=='comment':referenced.update(e.get(f'{{{W}}}id') for body in roots for e in body.findall('.//w:commentRangeStart',NS))
        root=ET.fromstring(archive.read(path))
        definitions={e.get(f'{{{W}}}id'):e for e in root}
        for ref_id,element in definitions.items():
            if ref_id not in referenced or element.get(f'{{{W}}}type') in ('separator','continuationSeparator','continuationNotice'):continue
            paragraphs=[]
            for paragraph in element.findall('.//w:p',NS):
                fragments=[]
                for node in paragraph.iter():
                    if node.tag==f'{{{W}}}t':fragments.append(node.text or '')
                    elif node.tag==f'{{{W}}}tab':fragments.append('\t')
                    elif node.tag in (f'{{{W}}}br',f'{{{W}}}cr'):fragments.append('\n')
                paragraphs.append(''.join(fragments))
            text='\n'.join(paragraphs).strip()
            if not text:continue
            author=element.get(f'{{{W}}}author') if kind=='comment' else None
            if author:text=author+'\n'+text
            result.append({'id':f'ann-{doc_id}-{kind}-{ref_id}','block_id':None,'kind':kind,'anchor_text':f'【{label}{ref_id}】','target_value':text})
    return result
