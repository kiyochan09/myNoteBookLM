import io
import re
import json
import uuid
import html
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import docx
from docx.text.paragraph import Paragraph
from docx.table import Table
from PIL import Image as PILImage

from app.importers.base_importer import BaseImporter


# XML名前空間定義
NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'v': 'urn:schemas-microsoft-com:vml',
    'w14': 'http://schemas.microsoft.com/office/word/2010/wordml',
}

# ハイライト色のカラーマップ
HIGHLIGHT_COLOR_MAP = {
    'yellow': '#ffff00',
    'green': '#00ff00',
    'cyan': '#00ffff',
    'magenta': '#ff00ff',
    'blue': '#0000ff',
    'red': '#ff0000',
    'darkBlue': '#00008b',
    'darkCyan': '#008b8b',
    'darkGreen': '#006400',
    'darkMagenta': '#8b008b',
    'darkRed': '#8b0000',
    'darkYellow': '#808000',
    'darkGray': '#a9a9a9',
    'lightGray': '#d3d3d3',
    'black': '#000000',
}


class DocxImporter(BaseImporter):
    """
    DOCXファイルを解析し、文字修飾（太字・斜体・下線・打消線・上付/下付・文字色・背景色等）、
    ハイパーリンク、ブックマーク、見出し階層、インライン画像、表を忠実に復元して
    ドキュメントバンドルを生成するインポーター。
    """

    def _extract_all_rels(self, zip_archive: zipfile.ZipFile) -> Dict[str, str]:
        """word/_rels/document.xml.rels から全リレーションシップ（画像、リンク等）を抽出"""
        rels_map: Dict[str, str] = {}
        if "word/_rels/document.xml.rels" in zip_archive.namelist():
            try:
                xml_content = zip_archive.read("word/_rels/document.xml.rels")
                root = ET.fromstring(xml_content)
                for rel in root:
                    r_id = rel.get("Id")
                    target = rel.get("Target")
                    if r_id and target:
                        rels_map[r_id] = target
            except Exception as e:
                print(f"[DocxImporter] Error parsing document.xml.rels: {e}")
        return rels_map

    def _extract_styles_map(self, zip_archive: zipfile.ZipFile) -> Dict[str, Dict[str, Any]]:
        """word/styles.xml から各 styleId に対応するスタイル名(name)、アウトラインレベル(outline_level)等を抽出"""
        styles_map: Dict[str, Dict[str, Any]] = {}
        if "word/styles.xml" in zip_archive.namelist():
            try:
                xml_content = zip_archive.read("word/styles.xml")
                root = ET.fromstring(xml_content)
                for style in root.findall(f'.//{{{NS["w"]}}}style'):
                    s_id = style.get(f'{{{NS["w"]}}}styleId')
                    if not s_id:
                        continue
                    s_type = style.get(f'{{{NS["w"]}}}type') or 'paragraph'
                    name_elem = style.find(f'{{{NS["w"]}}}name')
                    s_name = name_elem.get(f'{{{NS["w"]}}}val') if name_elem is not None else ""

                    outline_lvl = None
                    outline_elem = style.find(f'.//{{{NS["w"]}}}outlineLvl')
                    if outline_elem is not None:
                        val = outline_elem.get(f'{{{NS["w"]}}}val')
                        if val and val.isdigit():
                            outline_lvl = int(val) + 1

                    styles_map[s_id] = {
                        "name": s_name,
                        "type": s_type,
                        "outline_level": outline_lvl
                    }
            except Exception as e:
                print(f"[DocxImporter] Error parsing styles.xml: {e}")
        return styles_map

    def _parse_run_element(self, r_elem: ET.Element) -> Tuple[str, str]:
        """
        <w:r> ラン要素を解析し、(プレーンテキスト, 修飾付きHTML) のタプルを返す。
        """
        raw_text_parts = []
        html_text_parts = []

        for child in r_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 't':
                t_text = child.text or ""
                raw_text_parts.append(t_text)
                html_text_parts.append(html.escape(t_text))
            elif tag == 'br':
                raw_text_parts.append("\n")
                html_text_parts.append("<br>")
            elif tag == 'tab':
                raw_text_parts.append("\t")
                html_text_parts.append("&emsp;")
            elif tag == 'sym':
                char_code = child.get(f'{{{NS["w"]}}}char')
                if char_code:
                    try:
                        c = chr(int(char_code, 16))
                        raw_text_parts.append(c)
                        html_text_parts.append(html.escape(c))
                    except Exception:
                        pass

        plain_text = "".join(raw_text_parts)
        if not plain_text and not html_text_parts:
            return "", ""

        inner_html = "".join(html_text_parts)

        rPr = r_elem.find(f'{{{NS["w"]}}}rPr')
        if rPr is None:
            return plain_text, inner_html

        is_bold = False
        is_italic = False
        is_underline = False
        is_strike = False
        vert_align = None
        color_hex = None
        highlight_color = None
        font_size_pt = None
        is_code = False

        for prop in rPr:
            ptag = prop.tag.split('}')[-1] if '}' in prop.tag else prop.tag
            val = prop.get(f'{{{NS["w"]}}}val')

            if ptag in ('b', 'bCs'):
                if val not in ('0', 'false', 'off'):
                    is_bold = True
            elif ptag in ('i', 'iCs'):
                if val not in ('0', 'false', 'off'):
                    is_italic = True
            elif ptag == 'u':
                if val and val != 'none':
                    is_underline = True
            elif ptag in ('strike', 'dstrike'):
                if val not in ('0', 'false', 'off'):
                    is_strike = True
            elif ptag == 'vertAlign':
                if val in ('superscript', 'subscript'):
                    vert_align = val
            elif ptag == 'color':
                if val and val.lower() != 'auto' and re.match(r'^[0-9a-fA-F]{6}$', val):
                    color_hex = f"#{val}"
            elif ptag == 'highlight':
                if val and val.lower() != 'none':
                    highlight_color = HIGHLIGHT_COLOR_MAP.get(val, val)
            elif ptag == 'sz':
                if val and val.isdigit():
                    half_pt = int(val)
                    if half_pt != 21 and half_pt != 22 and half_pt != 24:
                        font_size_pt = round(half_pt / 2, 1)
            elif ptag == 'rStyle':
                if val and any(c in val.lower() for c in ('code', 'htmlcode', 'verbatim')):
                    is_code = True

        res_html = inner_html

        styles = []
        if color_hex:
            styles.append(f"color: {color_hex};")
        if highlight_color:
            styles.append(f"background-color: {highlight_color};")
        if font_size_pt:
            styles.append(f"font-size: {font_size_pt}pt;")

        if styles:
            res_html = f'<span style="{" ".join(styles)}">{res_html}</span>'

        if is_code:
            res_html = f"<code>{res_html}</code>"
        if vert_align == 'superscript':
            res_html = f"<sup>{res_html}</sup>"
        elif vert_align == 'subscript':
            res_html = f"<sub>{res_html}</sub>"
        if is_strike:
            res_html = f"<del>{res_html}</del>"
        if is_underline:
            res_html = f"<u>{res_html}</u>"
        if is_italic:
            res_html = f"<em>{res_html}</em>"
        if is_bold:
            res_html = f"<strong>{res_html}</strong>"

        return plain_text, res_html

    def _parse_paragraph_content(
        self,
        p_elem: ET.Element,
        rels_map: Dict[str, str]
    ) -> Tuple[str, str, Dict[str, Any]]:
        """
        <w:p> 段落要素を走査し、(プレーンテキスト, 修飾付きHTML, 段落メタ情報) を返す。
        ブックマーク、ハイパーリンク、文字修飾を包含。
        """
        plain_parts = []
        html_parts = []
        meta = {
            "align": None,
            "style_name": "",
            "is_list": False,
            "num_level": 0,
            "outline_level": None,
        }

        pPr = p_elem.find(f'{{{NS["w"]}}}pPr')
        if pPr is not None:
            pStyle = pPr.find(f'{{{NS["w"]}}}pStyle')
            if pStyle is not None:
                meta["style_name"] = pStyle.get(f'{{{NS["w"]}}}val') or ""

            jc = pPr.find(f'{{{NS["w"]}}}jc')
            if jc is not None:
                jc_val = jc.get(f'{{{NS["w"]}}}val')
                if jc_val in ('center', 'right', 'both'):
                    meta["align"] = "center" if jc_val == "center" else ("right" if jc_val == "right" else "justify")

            outlineLvl = pPr.find(f'{{{NS["w"]}}}outlineLvl')
            if outlineLvl is not None:
                val = outlineLvl.get(f'{{{NS["w"]}}}val')
                if val and val.isdigit():
                    meta["outline_level"] = int(val) + 1

            numPr = pPr.find(f'{{{NS["w"]}}}numPr')
            if numPr is not None:
                meta["is_list"] = True
                ilvl = numPr.find(f'{{{NS["w"]}}}ilvl')
                if ilvl is not None:
                    lvl_val = ilvl.get(f'{{{NS["w"]}}}val')
                    if lvl_val and lvl_val.isdigit():
                        meta["num_level"] = int(lvl_val)

        for child in p_elem:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

            if tag == 'bookmarkStart':
                bm_name = child.get(f'{{{NS["w"]}}}name')
                if bm_name and bm_name != '_GoBack':
                    html_parts.append(f'<span id="{bm_name}" class="doc-bookmark" title="Bookmark: {bm_name}"></span>')

            elif tag == 'hyperlink':
                r_id = child.get(f'{{{NS["r"]}}}id')
                anchor = child.get(f'{{{NS["w"]}}}anchor')
                target_url = rels_map.get(r_id, "") if r_id else ""

                link_plain_parts = []
                link_html_parts = []

                for sub_r in child.findall(f'.//{{{NS["w"]}}}r'):
                    p_txt, h_txt = self._parse_run_element(sub_r)
                    if p_txt:
                        link_plain_parts.append(p_txt)
                    if h_txt:
                        link_html_parts.append(h_txt)

                link_plain = "".join(link_plain_parts)
                link_inner = "".join(link_html_parts)

                if link_plain or link_inner:
                    plain_parts.append(link_plain)
                    if target_url:
                        html_parts.append(f'<a href="{html.escape(target_url)}" target="_blank" rel="noopener noreferrer">{link_inner}</a>')
                    elif anchor:
                        html_parts.append(f'<a href="#{html.escape(anchor)}" class="doc-internal-link">{link_inner}</a>')
                    else:
                        html_parts.append(f'<a href="javascript:void(0)">{link_inner}</a>')

            elif tag == 'r':
                p_txt, h_txt = self._parse_run_element(child)
                if p_txt:
                    plain_parts.append(p_txt)
                if h_txt:
                    html_parts.append(h_txt)

            elif tag == 'fldSimple':
                instr = child.get(f'{{{NS["w"]}}}instr') or ""
                m_url = re.search(r'HYPERLINK\s+"([^"]+)"', instr)
                field_url = m_url.group(1) if m_url else None

                fld_plain_parts = []
                fld_html_parts = []
                for sub_r in child.findall(f'.//{{{NS["w"]}}}r'):
                    p_txt, h_txt = self._parse_run_element(sub_r)
                    if p_txt:
                        fld_plain_parts.append(p_txt)
                    if h_txt:
                        fld_html_parts.append(h_txt)

                fld_plain = "".join(fld_plain_parts)
                fld_inner = "".join(fld_html_parts)
                if fld_plain or fld_inner:
                    plain_parts.append(fld_plain)
                    if field_url:
                        html_parts.append(f'<a href="{html.escape(field_url)}" target="_blank" rel="noopener noreferrer">{fld_inner}</a>')
                    else:
                        html_parts.append(fld_inner)

        full_plain = "".join(plain_parts).strip()
        full_html = "".join(html_parts)

        return full_plain, full_html, meta

    def parse_file(self, file_path: Path, original_filename: Optional[str] = None) -> Dict[str, Any]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = docx.Document(str(file_path))
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        if original_filename:
            doc_title = Path(original_filename).stem
            source_file_name = Path(original_filename).name
        else:
            clean_name = re.sub(r'^[0-9a-fA-F]{32}_', '', file_path.name)
            doc_title = Path(clean_name).stem
            source_file_name = clean_name

        sections: List[Dict[str, Any]] = []
        blocks: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        figures: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        zip_archive = zipfile.ZipFile(str(file_path), "r")
        rels_map = self._extract_all_rels(zip_archive)
        styles_map = self._extract_styles_map(zip_archive)

        for rId, rel in doc.part.rels.items():
            if rId not in rels_map:
                rels_map[rId] = str(rel.target_ref)

        extracted_media_set = set()

        root_sec_id = f"sec-{uuid.uuid4().hex[:12]}"
        sections.append({
            "id": root_sec_id,
            "parent_id": None,
            "title": doc_title,
            "level": 1,
            "order_idx": 0,
            "page_number": 1
        })

        sec_stack = [(1, root_sec_id)]
        current_sec_id = root_sec_id
        reading_order = 0
        order_idx_counter = 0

        # XML要素の順次走査 (body 内の段落と表)
        for elem in doc.element.body:
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag == 'p':
                p_xml = elem.xml

                # 1. 段落内のインライン画像をチェック
                embed_rids = []
                for m in re.finditer(r'r:embed="([^"]+)"', p_xml):
                    embed_rids.append(m.group(1))
                for m in re.finditer(r'r:id="([^"]+)"', p_xml):
                    embed_rids.append(m.group(1))

                for rId in embed_rids:
                    if rId in rels_map:
                        target_ref = rels_map[rId]
                        clean_ref = target_ref.replace("../", "")
                        zip_entry = f"word/{clean_ref}" if not clean_ref.startswith("word/") else clean_ref

                        if zip_entry in zip_archive.namelist():
                            img_bytes = zip_archive.read(zip_entry)
                            ext = Path(zip_entry).suffix or ".png"
                            saved = self.save_image_bytes(img_bytes, original_ext=ext, prefix="docx")
                            extracted_media_set.add(zip_entry)

                            w, h = 0, 0
                            try:
                                im = PILImage.open(io.BytesIO(img_bytes))
                                w, h = im.size
                            except Exception:
                                pass

                            reading_order += 1
                            fig_id = f"fig-{uuid.uuid4().hex[:12]}"
                            fig_count = len(figures) + 1
                            caption = f"図版 {fig_count} ({saved['file_name']})"

                            figures.append({
                                "id": fig_id,
                                "section_id": current_sec_id,
                                "caption": caption,
                                "file_path": saved["file_path"],
                                "file_hash": saved["file_hash"],
                                "file_size_kb": saved["file_size_kb"],
                                "width": w,
                                "height": h,
                                "reading_order": reading_order,
                                "page_number": 1,
                                "is_custom": False
                            })

                            blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                            fig_html = f'<figure class="figure-embed" data-fig-id="{fig_id}"><div class="fig-actions" contenteditable="false"><button class="fig-action-btn btn-move" onclick="openMoveFigureModal(\'{fig_id}\')" title="見出し/位置を指定して移動"><i class="fa-solid fa-arrows-up-down-left-right"></i> 移動</button><button class="fig-action-btn btn-delete" onclick="deleteFigure(\'{fig_id}\')" title="画像を削除"><i class="fa-solid fa-trash"></i> 削除</button></div><img src="{saved["file_path"]}" alt="{caption}"><figcaption contenteditable="true" oninput="markDirty()">{caption}</figcaption></figure>'
                            blocks.append({
                                "id": blk_id,
                                "section_id": current_sec_id,
                                "block_type": "paragraph",
                                "text_content": f"[図版: {caption}]",
                                "html_content": fig_html,
                                "reading_order": reading_order,
                                "page_number": 1,
                                "bbox_json": None
                            })

                # 2. 段落テキスト・文字修飾・ハイパーリンク・ブックマークの解析
                plain_text, inner_html, meta = self._parse_paragraph_content(elem, rels_map)

                if not plain_text and not inner_html:
                    continue

                reading_order += 1
                style_id = meta["style_name"]
                style_info = styles_map.get(style_id, {})
                style_display_name = style_info.get("name", "")
                style_outline_lvl = style_info.get("outline_level")

                # 見出し判定（スタイルID, スタイル名, アウトラインレベルの総合評価）
                is_heading = False
                h_level = 1

                if meta["outline_level"] is not None:
                    is_heading = True
                    h_level = min(max(meta["outline_level"], 1), 6)
                elif style_outline_lvl is not None:
                    is_heading = True
                    h_level = min(max(style_outline_lvl, 1), 6)
                elif style_id in ('1', '2', '3', '4', '5', '6'):
                    is_heading = True
                    h_level = int(style_id)
                elif re.match(r'^(?:Heading|見出し|Titre|Title|Header|Headline|H)\s*([1-6])$', style_id, re.IGNORECASE):
                    m = re.match(r'^(?:Heading|見出し|Titre|Title|Header|Headline|H)\s*([1-6])$', style_id, re.IGNORECASE)
                    is_heading = True
                    h_level = int(m.group(1))
                elif re.search(r'(?:Heading|見出し|Titre|Title|Header|Headline|章|節|項)\s*([1-6])?', style_display_name, re.IGNORECASE):
                    m = re.search(r'(?:Heading|見出し|Titre|Title|Header|Headline|章|節|項)\s*([1-6])?', style_display_name, re.IGNORECASE)
                    is_heading = True
                    h_level = int(m.group(1)) if m and m.group(1) else 1
                elif style_id.lower() in ('title', 'タイトル', '表題') or style_display_name.lower() in ('title', 'タイトル', '表題'):
                    is_heading = True
                    h_level = 1
                elif style_id.lower() in ('subtitle', 'サブタイトル', '副題') or style_display_name.lower() in ('subtitle', 'サブタイトル', '副題'):
                    is_heading = True
                    h_level = 2

                align_attr = f' style="text-align: {meta["align"]};"' if meta["align"] else ""

                if is_heading:
                    while sec_stack and sec_stack[-1][0] >= h_level:
                        sec_stack.pop()
                    parent_id = sec_stack[-1][1] if sec_stack else root_sec_id

                    order_idx_counter += 1
                    new_sec_id = f"sec-{uuid.uuid4().hex[:12]}"
                    sections.append({
                        "id": new_sec_id,
                        "parent_id": parent_id,
                        "title": plain_text,
                        "level": h_level,
                        "order_idx": order_idx_counter,
                        "page_number": 1
                    })
                    sec_stack.append((h_level, new_sec_id))
                    current_sec_id = new_sec_id

                    blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                    blocks.append({
                        "id": blk_id,
                        "section_id": current_sec_id,
                        "block_type": "heading",
                        "text_content": plain_text,
                        "html_content": f"<h{h_level}{align_attr}>{inner_html}</h{h_level}>",
                        "reading_order": reading_order,
                        "page_number": 1,
                        "bbox_json": None
                    })
                else:
                    blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                    if meta["is_list"]:
                        indent = f' style="margin-left: {(meta["num_level"] + 1) * 20}px;"'
                        p_html = f'<p class="doc-list-item"{indent}>• {inner_html}</p>'
                    else:
                        p_html = f"<p{align_attr}>{inner_html}</p>"

                    blocks.append({
                        "id": blk_id,
                        "section_id": current_sec_id,
                        "block_type": "paragraph",
                        "text_content": plain_text,
                        "html_content": p_html,
                        "reading_order": reading_order,
                        "page_number": 1,
                        "bbox_json": None
                    })

            elif tag == 'tbl':
                t = Table(elem, doc)
                reading_order += 1

                raw_rows: List[List[str]] = []
                html_rows: List[List[str]] = []

                for row in t.rows:
                    row_plain = []
                    row_html = []
                    for cell in row.cells:
                        c_plain_parts = []
                        c_html_parts = []
                        for cell_p in cell.paragraphs:
                            p_txt, h_txt, _ = self._parse_paragraph_content(cell_p._p, rels_map)
                            if p_txt:
                                c_plain_parts.append(p_txt)
                            if h_txt:
                                c_html_parts.append(h_txt)
                        row_plain.append(" ".join(c_plain_parts).strip())
                        row_html.append("<br>".join(c_html_parts) if c_html_parts else "")
                    raw_rows.append(row_plain)
                    html_rows.append(row_html)

                if raw_rows:
                    row_count = len(raw_rows)
                    col_count = max(len(r) for r in raw_rows) if raw_rows else 0
                    headers = raw_rows[0] if raw_rows else []
                    body_rows = raw_rows[1:] if len(raw_rows) > 1 else raw_rows

                    grid_data = {
                        "headers": headers,
                        "rows": raw_rows,
                        "columns": col_count,
                        "row_count": row_count
                    }

                    tbl_id = f"tbl-{uuid.uuid4().hex[:12]}"
                    caption = f"表 ({row_count}行 × {col_count}列)"
                    tables.append({
                        "id": tbl_id,
                        "section_id": current_sec_id,
                        "caption": caption,
                        "row_count": row_count,
                        "col_count": col_count,
                        "grid_json": json.dumps(grid_data, ensure_ascii=False),
                        "reading_order": reading_order,
                        "page_number": 1
                    })

                    # 表HTML（各セル内の文字修飾・リンクを保持）
                    tbl_html_parts = ['<div class="table-embed"><table class="styled-table">']
                    if html_rows:
                        tbl_html_parts.append('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in html_rows[0]) + '</tr></thead>')
                        if len(html_rows) > 1:
                            tbl_html_parts.append('<tbody>')
                            for r in html_rows[1:]:
                                tbl_html_parts.append('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>')
                            tbl_html_parts.append('</tbody>')
                    tbl_html_parts.append('</table></div>')

                    blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                    table_summary = " | ".join(headers) + "\n" + "\n".join([" | ".join(r) for r in body_rows])
                    blocks.append({
                        "id": blk_id,
                        "section_id": current_sec_id,
                        "block_type": "paragraph",
                        "text_content": f"[表: {caption}]\n{table_summary}",
                        "html_content": "".join(tbl_html_parts),
                        "reading_order": reading_order,
                        "page_number": 1,
                        "bbox_json": None
                    })

        # 3. 本文で抽出されなかった残りの画像をフォールバック抽出
        all_media_files = [n for n in zip_archive.namelist() if n.startswith("word/media/")]
        for media_name in all_media_files:
            if media_name not in extracted_media_set:
                img_bytes = zip_archive.read(media_name)
                ext = Path(media_name).suffix or ".png"
                saved = self.save_image_bytes(img_bytes, original_ext=ext, prefix="docx")

                w, h = 0, 0
                try:
                    im = PILImage.open(io.BytesIO(img_bytes))
                    w, h = im.size
                except Exception:
                    pass

                reading_order += 1
                fig_id = f"fig-{uuid.uuid4().hex[:12]}"
                fig_count = len(figures) + 1
                caption = f"図版 {fig_count} ({saved['file_name']})"

                figures.append({
                    "id": fig_id,
                    "section_id": current_sec_id,
                    "caption": caption,
                    "file_path": saved["file_path"],
                    "file_hash": saved["file_hash"],
                    "file_size_kb": saved["file_size_kb"],
                    "width": w,
                    "height": h,
                    "reading_order": reading_order,
                    "page_number": 1,
                    "is_custom": False
                })

                blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                fig_html = f'<figure class="figure-embed" data-fig-id="{fig_id}"><div class="fig-actions" contenteditable="false"><button class="fig-action-btn btn-move" onclick="openMoveFigureModal(\'{fig_id}\')" title="見出し/位置を指定して移動"><i class="fa-solid fa-arrows-up-down-left-right"></i> 移動</button><button class="fig-action-btn btn-delete" onclick="deleteFigure(\'{fig_id}\')" title="画像を削除"><i class="fa-solid fa-trash"></i> 削除</button></div><img src="{saved["file_path"]}" alt="{caption}"><figcaption contenteditable="true" oninput="markDirty()">{caption}</figcaption></figure>'
                blocks.append({
                    "id": blk_id,
                    "section_id": current_sec_id,
                    "block_type": "paragraph",
                    "text_content": f"[図版: {caption}]",
                    "html_content": fig_html,
                    "reading_order": reading_order,
                    "page_number": 1,
                    "bbox_json": None
                })

        zip_archive.close()

        return {
            "document": {
                "id": doc_id,
                "title": doc_title,
                "source_type": "docx",
                "source_filename": source_file_name,
                "total_pages": 1,
                "doc_metadata_json": {
                    "original_path": str(file_path),
                    "file_size_bytes": file_path.stat().st_size
                }
            },
            "sections": sections,
            "blocks": blocks,
            "tables": tables,
            "figures": figures,
            "annotations": annotations
        }
