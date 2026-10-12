from contextlib import closing
from html.parser import HTMLParser
from html import escape
import json, uuid

CSS = {'fontFamily':'font-family','fontSize':'font-size','fontWeight':'font-weight','fontStyle':'font-style','textColor':'color','backgroundColor':'background-color','textDecoration':'text-decoration','underlineStyle':'text-decoration-style','underlineColor':'text-decoration-color','lineHeight':'line-height','textAlign':'text-align','textIndent':'text-indent','paddingInlineStart':'padding-inline-start','marginTop':'margin-top','marginBottom':'margin-bottom'}
TAGS={'p','h1','h2','h3','h4','h5','h6','blockquote','li','div'}

def css_values(style):
    return {css:str(style[key]) for key,css in CSS.items() if style.get(key) is not None}

class StylePatcher(HTMLParser):
    def __init__(self, source, previous, style):
        super().__init__(convert_charrefs=False)
        self.source=source; self.previous=previous; self.style=style; self.edits=[]
        self.offsets=[0]
        for line in source.splitlines(True): self.offsets.append(self.offsets[-1]+len(line))
    def handle_starttag(self,tag,attrs):
        if tag not in TAGS:return
        data=dict(attrs); marker=data.get('data-editor-style-id')
        css={}
        for pair in (data.get('style') or '').split(';'):
            if ':' in pair:
                k,v=pair.split(':',1);css[k.strip().lower()]=v.strip()
        matches=marker==self.style['id']
        if not marker and self.previous:
            expected=css_values(self.previous)
            old_tag=self.previous.get('headingLevel','p')
            # Legacy imported headings have semantic tags but no style ID.
            matches=(self.style['id'] in {f'para-h{i}' for i in range(1,7)} and tag==old_tag)
            if self.style['id']=='para-normal' and tag=='p' and css.get('text-indent','0') in ('0','0px','0em','normal'):matches=True
            if not matches and tag==old_tag and expected:
                matches=all(css.get(k)==v for k,v in expected.items())
        if not matches:return
        for key in CSS.values():css.pop(key,None)
        css.update(css_values(self.style))
        data['style']='; '.join(f'{k}: {v}' for k,v in css.items())
        data['data-editor-style-id']=self.style['id']
        opening='<'+tag+''.join(' '+k+(f'="{escape(v,quote=True)}"' if v is not None else '') for k,v in data.items())+'>'
        line,col=self.getpos();start=self.offsets[line-1]+col
        self.edits.append((start,start+len(self.get_starttag_text()),opening))
    def result(self):
        self.feed(self.source);out=self.source
        for start,end,value in reversed(self.edits):out=out[:start]+value+out[end:]
        return out

def ensure_schema(conn):
    conn.execute('CREATE TABLE IF NOT EXISTS editor_paragraph_styles (style_id TEXT, scope_doc TEXT NOT NULL, style_json TEXT NOT NULL, PRIMARY KEY(style_id,scope_doc))')
    conn.execute('CREATE TABLE IF NOT EXISTS editor_style_backups (operation_id TEXT, block_id TEXT, html_content TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')

def load_styles(service,doc_id=None):
    with closing(service.get_connection()) as conn, conn:
        ensure_schema(conn)
        rows=conn.execute('SELECT style_id,scope_doc,style_json FROM editor_paragraph_styles WHERE scope_doc=\'\' OR scope_doc=? ORDER BY length(scope_doc)',(doc_id or '',)).fetchall()
        styles={row['style_id']:json.loads(row['style_json']) for row in rows}
    return list(styles.values())

def update_style(service,style,previous,scope,doc_id):
    if scope not in ('document','database') or style.get('category')!='paragraph' or not style.get('id'):raise ValueError('段落スタイルと適用範囲を指定してください。')
    if scope=='document' and not doc_id:raise ValueError('アクティブな文書を開いてください。')
    operation=str(uuid.uuid4());changed=set();blocks=0
    with closing(service.get_connection()) as conn, conn:
        ensure_schema(conn);conn.execute('BEGIN IMMEDIATE')
        if scope=='document' and not conn.execute('SELECT 1 FROM documents WHERE id=?',(doc_id,)).fetchone():raise ValueError('文書が見つかりません。')
        rows=conn.execute('SELECT id,document_id,html_content FROM content_blocks'+(' WHERE document_id=?' if scope=='document' else ''),(doc_id,) if scope=='document' else ())
        for row in rows:
            original=row['html_content'] or ''
            updated=StylePatcher(original,previous,style).result()
            if updated==original:continue
            conn.execute('INSERT INTO editor_style_backups(operation_id,block_id,html_content) VALUES(?,?,?)',(operation,row['id'],original))
            conn.execute('UPDATE content_blocks SET html_content=? WHERE id=?',(updated,row['id']))
            changed.add(row['document_id']);blocks+=1
        if scope=='database':conn.execute('DELETE FROM editor_paragraph_styles WHERE style_id=?',(style['id'],))
        conn.execute('INSERT OR REPLACE INTO editor_paragraph_styles VALUES(?,?,?)',(style['id'],doc_id if scope=='document' else '',json.dumps(style,ensure_ascii=False)))
        for document in changed:conn.execute('UPDATE documents SET updated_at=CURRENT_TIMESTAMP WHERE id=?',(document,))
    return {'status':'success','documents':len(changed),'blocks':blocks,'operation_id':operation}
