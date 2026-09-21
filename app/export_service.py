import io
import re
import base64
import json
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from PIL import Image as PILImage

from app.db_service import DBService, BASE_DIR
from app.ocr_pipeline.katakana_corrector import correct_japanese_text


class ExportService:
    """
    MyNotebookLM のドキュメントデータを各種フォーマット
    （Markdown / HTML / DOCX / ODT）で出力・ダウンロード用バイナリを生成するサービス。
    """

    def __init__(self, db_service: Optional[DBService] = None):
        self.db_service = db_service or DBService()

    def get_bundle(self, doc_id: str) -> Dict[str, Any]:
        bundle = self.db_service.get_document_bundle(doc_id)
        if not bundle:
            raise ValueError(f"Document not found: {doc_id}")
        return bundle

    def export_markdown(self, doc_id: str) -> str:
        bundle = self.get_bundle(doc_id)
        doc = bundle["document"]
        blocks = bundle["blocks"]
        tables_map = {t["id"]: t for t in bundle["tables"]}
        figures_map = {f["id"]: f for f in bundle["figures"]}

        lines = [f"# {doc.get('title', 'Document')}\n"]

        for blk in blocks:
            b_type = blk.get("block_type")
            text = blk.get("text_content", "")
            h_content = blk.get("html_content", "")

            if b_type == "heading":
                # 見出しレベル判定 (h1〜h6)
                if "<h1>" in h_content:
                    lines.append(f"\n# {text}\n")
                elif "<h2>" in h_content:
                    lines.append(f"\n## {text}\n")
                elif "<h3>" in h_content:
                    lines.append(f"\n### {text}\n")
                elif "<h4>" in h_content:
                    lines.append(f"\n#### {text}\n")
                else:
                    lines.append(f"\n## {text}\n")
            elif "data-table-id" in h_content:
                # 表のMarkdown化
                import re
                m = re.search(r'data-table-id="([^"]+)"', h_content)
                if m and m.group(1) in tables_map:
                    tbl = tables_map[m.group(1)]
                    grid = tbl.get("grid", {})
                    rows = grid.get("rows", [])
                    if rows:
                        header = rows[0]
                        lines.append("\n| " + " | ".join([str(c).replace("\n", " ") for c in header]) + " |")
                        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
                        for r in rows[1:]:
                            lines.append("| " + " | ".join([str(c).replace("\n", " ") for c in r]) + " |")
                        lines.append("\n")
                else:
                    lines.append(f"\n{text}\n")
            elif "data-fig-id" in h_content:
                import re
                m = re.search(r'data-fig-id="([^"]+)"', h_content)
                if m and m.group(1) in figures_map:
                    fig = figures_map[m.group(1)]
                    lines.append(f"\n![{fig.get('caption', 'Figure')}]({fig.get('file_path', '')})\n")
                else:
                    lines.append(f"\n{text}\n")
            else:
                lines.append(f"\n{text}\n")

        # 脚注
        annotations = bundle.get("annotations", [])
        if annotations:
            lines.append("\n\n---\n### 注釈・脚注\n")
            for idx, ann in enumerate(annotations, start=1):
                lines.append(f"- **【注{idx}】** {ann.get('anchor_text', '')}: {ann.get('target_value', '')}")

        return "\n".join(lines)

    def export_html(self, doc_id: str) -> str:
        bundle = self.get_bundle(doc_id)
        doc = bundle["document"]
        title = doc.get("title", "Document")
        blocks = bundle["blocks"]
        tables_map = {t["id"]: t for t in bundle["tables"]}
        figures_map = {f["id"]: f for f in bundle["figures"]}

        body_html_parts = []
        for blk in blocks:
            h_content = blk.get("html_content", "")
            if "data-table-id" in h_content:
                import re
                m = re.search(r'data-table-id="([^"]+)"', h_content)
                if m and m.group(1) in tables_map:
                    tbl = tables_map[m.group(1)]
                    grid = tbl.get("grid", {})
                    rows = grid.get("rows", [])
                    if rows:
                        t_html = ['<table class="styled-table">']
                        t_html.append('<thead><tr>' + ''.join([f'<th>{c}</th>' for c in rows[0]]) + '</tr></thead>')
                        t_html.append('<tbody>')
                        for r in rows[1:]:
                            t_html.append('<tr>' + ''.join([f'<td>{c}</td>' for c in r]) + '</tr>')
                        t_html.append('</tbody></table>')
                        body_html_parts.append(''.join(t_html))
                    else:
                        body_html_parts.append(h_content)
                else:
                    body_html_parts.append(h_content)
            else:
                body_html_parts.append(h_content)

        ann_html = []
        annotations = bundle.get("annotations", [])
        if annotations:
            ann_html.append('<section class="annotations-section"><hr><h3>注釈・脚注</h3><ul>')
            for idx, ann in enumerate(annotations, start=1):
                ann_html.append(f'<li><strong>【注{idx}】</strong> {ann.get("anchor_text", "")}: {ann.get("target_value", "")}</li>')
            ann_html.append('</ul></section>')

        full_html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} - MyNotebookLM</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", "BIZ UDPGothic", Meiryo, sans-serif;
        line-height: 1.8;
        color: #1e293b;
        background-color: #f8fafc;
        margin: 0;
        padding: 40px 20px;
    }}
    .document-container {{
        max-width: 900px;
        margin: 0 auto;
        background: #ffffff;
        padding: 60px 80px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #e2e8f0;
    }}
    h1 {{ font-size: 2.2rem; color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 12px; margin-top: 0; }}
    h2 {{ font-size: 1.6rem; color: #1e293b; margin-top: 36px; border-left: 4px solid #3b82f6; padding-left: 12px; }}
    h3 {{ font-size: 1.3rem; color: #334155; margin-top: 28px; }}
    p {{ margin: 16px 0; font-size: 1.05rem; word-break: break-word; }}
    .styled-table {{ width: 100%; border-collapse: collapse; margin: 24px 0; font-size: 0.95rem; }}
    .styled-table th, .styled-table td {{ border: 1px solid #cbd5e1; padding: 10px 14px; text-align: left; }}
    .styled-table th {{ background-color: #f1f5f9; font-weight: 600; color: #334155; }}
    .styled-table tr:nth-child(even) {{ background-color: #f8fafc; }}
    figure {{ margin: 24px 0; text-align: center; }}
    figure img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; }}
    figcaption {{ font-size: 0.9rem; color: #64748b; margin-top: 8px; }}
    .annotations-section {{ margin-top: 48px; color: #475569; }}
</style>
</head>
<body>
<div class="document-container">
    {''.join(body_html_parts)}
    {''.join(ann_html)}
</div>
</body>
</html>
"""
        return full_html

    def export_docx(self, doc_id: str) -> io.BytesIO:
        bundle = self.get_bundle(doc_id)
        doc_info = bundle["document"]
        title = doc_info.get("title", "Document")
        blocks = bundle["blocks"]
        tables_map = {t["id"]: t for t in bundle["tables"]}
        figures_map = {f["id"]: f for f in bundle["figures"]}

        docx_doc = docx.Document()
        title_p = docx_doc.add_heading(title, level=0)

        for blk in blocks:
            b_type = blk.get("block_type")
            text = blk.get("text_content", "")
            h_content = blk.get("html_content", "")

            if b_type == "heading":
                lvl = 1
                if "<h1>" in h_content: lvl = 1
                elif "<h2>" in h_content: lvl = 2
                elif "<h3>" in h_content: lvl = 3
                docx_doc.add_heading(text, level=lvl)

            elif "data-table-id" in h_content:
                import re
                m = re.search(r'data-table-id="([^"]+)"', h_content)
                if m and m.group(1) in tables_map:
                    tbl = tables_map[m.group(1)]
                    grid = tbl.get("grid", {})
                    rows = grid.get("rows", [])
                    if rows:
                        t = docx_doc.add_table(rows=len(rows), cols=len(rows[0]))
                        t.style = "Table Grid"
                        for r_idx, row in enumerate(rows):
                            for c_idx, val in enumerate(row):
                                t.cell(r_idx, c_idx).text = str(val)
                else:
                    docx_doc.add_paragraph(text)

            elif "data-fig-id" in h_content:
                import re
                m = re.search(r'data-fig-id="([^"]+)"', h_content)
                if m and m.group(1) in figures_map:
                    fig = figures_map[m.group(1)]
                    fpath = fig.get("file_path", "")
                    if fpath.startswith("/media/"):
                        disk_f = BASE_DIR / "data" / "media" / Path(fpath).name
                        if disk_f.exists():
                            try:
                                docx_doc.add_picture(str(disk_f), width=Inches(5.0))
                                caption_p = docx_doc.add_paragraph(fig.get("caption", ""))
                                caption_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            except Exception:
                                docx_doc.add_paragraph(f"[図版: {fig.get('caption', '')}]")
                else:
                    docx_doc.add_paragraph(text)
            else:
                docx_doc.add_paragraph(text)

        out_stream = io.BytesIO()
        docx_doc.save(out_stream)
        out_stream.seek(0)
        return out_stream

    def export_odt(self, doc_id: str) -> io.BytesIO:
        """
        OpenDocument XMLパッケージを組み立ててODTファイルを生成
        """
        bundle = self.get_bundle(doc_id)
        doc_info = bundle["document"]
        title = doc_info.get("title", "Document")
        blocks = bundle["blocks"]

        content_body_parts = []
        for blk in blocks:
            b_type = blk.get("block_type")
            text = blk.get("text_content", "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            h_content = blk.get("html_content", "")

            if b_type == "heading":
                lvl = "1"
                if "<h2>" in h_content: lvl = "2"
                elif "<h3>" in h_content: lvl = "3"
                content_body_parts.append(f'<text:h text:outline-level="{lvl}">{text}</text:h>')
            else:
                content_body_parts.append(f'<text:p>{text}</text:p>')

        content_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink" office:version="1.2">
<office:body>
<office:text>
<text:h text:outline-level="1">{title}</text:h>
{''.join(content_body_parts)}
</office:text>
</office:body>
</office:document-content>"""

        manifest_xml = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0" manifest:version="1.2">
 <manifest:file-entry manifest:full-path="/" manifest:version="1.2" manifest:media-type="application/vnd.oasis.opendocument.text"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
</manifest:manifest>"""

        out_stream = io.BytesIO()
        with zipfile.ZipFile(out_stream, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
            z.writestr("content.xml", content_xml.encode("utf-8"))
            z.writestr("META-INF/manifest.xml", manifest_xml.encode("utf-8"))

        out_stream.seek(0)
        return out_stream

    def export_ocr_batch_docx(
        self,
        pdf_stem: str,
        options: Optional[Dict[str, Any]] = None,
        memory_pages: Optional[List[Dict[str, Any]]] = None
    ) -> io.BytesIO:
        """
        data/ocr_results/<pdf_stem>/ 配下の全ページOCR修正結果から
        Word (.docx) ドキュメントを一括生成するサービス。
        【完全読み取り専用 (READ-ONLY)】ディスクやDBのデータを一切変更・上書きしません。
        """
        options = options or {}
        merge_cross_page = bool(options.get("merge_cross_page", True))
        insert_page_break = bool(options.get("insert_page_break", False))
        line_char_count = int(options.get("line_char_count") or 0)
        target_page = options.get("target_page")
        page_start = options.get("page_start")
        page_end = options.get("page_end")
        target_pages = options.get("pages")

        # ページフィルタの判定
        allowed_pages = None
        if target_page is not None:
            allowed_pages = {int(target_page)}
        elif target_pages is not None and len(target_pages) > 0:
            allowed_pages = {int(p) for p in target_pages}

        # 1. フォルダとページデータの収集
        clean_stem = Path(pdf_stem).stem
        doc_dir = BASE_DIR / "data" / "ocr_results" / clean_stem
        if not doc_dir.exists():
            # 拡張子なし/ありの両方を探索
            doc_dir = BASE_DIR / "data" / "ocr_results" / pdf_stem

        # メモリ上のページ辞書 (page_number -> data)
        mem_map = {}
        if memory_pages:
            for p in memory_pages:
                p_num = p.get("page_number") or p.get("page")
                if p_num:
                    mem_map[int(p_num)] = p

        # ページデータの集約 (番号順ソート)
        all_pages_data = []
        page_numbers_set = set()

        if doc_dir.exists():
            page_dirs = sorted(list(doc_dir.glob("page_*")))
            for p_dir in page_dirs:
                m = re.search(r"page_(\d+)", p_dir.name)
                p_num = int(m.group(1)) if m else None
                if p_num is not None:
                    # ページフィルタ適用
                    if allowed_pages is not None and p_num not in allowed_pages:
                        continue
                    if page_start is not None and p_num < int(page_start):
                        continue
                    if page_end is not None and p_num > int(page_end):
                        continue

                    page_numbers_set.add(p_num)
                    # メモリ優先 (未保存編集対応)、なければディスクから安全に読み込み (READ-ONLY)
                    if p_num in mem_map:
                        all_pages_data.append((p_num, mem_map[p_num], p_dir))
                    else:
                        pj = p_dir / "page_data.json"
                        if pj.exists():
                            try:
                                with open(pj, "r", encoding="utf-8") as f:
                                    p_data = json.load(f)
                                all_pages_data.append((p_num, p_data, p_dir))
                            except Exception as e:
                                print(f"Warning: error reading {pj}: {e}")

        # メモリにしか存在しない新規ページがあれば追加
        for p_num, p_data in sorted(mem_map.items()):
            p_num = int(p_num)
            if allowed_pages is not None and p_num not in allowed_pages:
                continue
            if page_start is not None and p_num < int(page_start):
                continue
            if page_end is not None and p_num > int(page_end):
                continue
            if p_num not in page_numbers_set:
                all_pages_data.append((p_num, p_data, None))

        all_pages_data.sort(key=lambda x: x[0])

        # 2. Word ドキュメントの初期化
        docx_doc = docx.Document()

        # スタイル・フォント設定 (BIZ UDPGothic / Meiryo / 日本語標準)
        normal_style = docx_doc.styles['Normal']
        normal_font = normal_style.font
        normal_font.name = 'BIZ UDPGothic'
        normal_font.size = Pt(10.5)
        normal_font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)

        # タイトル
        title_text = clean_stem
        if target_page is not None:
            title_text = f"{clean_stem} (P.{target_page})"
        elif len(all_pages_data) == 1 and (page_start == page_end or allowed_pages):
            title_text = f"{clean_stem} (P.{all_pages_data[0][0]})"
        elif page_start is not None and page_end is not None:
            if page_start == page_end:
                title_text = f"{clean_stem} (P.{page_start})"
            else:
                title_text = f"{clean_stem} (P.{page_start}〜P.{page_end})"
        elif target_pages and len(target_pages) > 1:
            title_text = f"{clean_stem} (P.{min(target_pages)}〜P.{max(target_pages)})"

        title_p = docx_doc.add_heading(title_text, level=0)
        title_p.paragraph_format.space_after = Pt(14)

        if not all_pages_data:
            empty_msg = f"（指定されたページ{' P.' + str(target_page) if target_page is not None else ''}のOCRデータはありません）"
            docx_doc.add_paragraph(empty_msg)

        # 3. ページまたぎ結合の準備と各ページ要素の出力
        all_footnotes = []

        # ページごとの本文段落リスト
        doc_elements = []

        for p_num, p_data, p_dir in all_pages_data:
            # 見出し
            headings = p_data.get("headings", [])
            for h in headings:
                h_title = h.get("title") if isinstance(h, dict) else str(h)
                h_level = h.get("level", 1) if isinstance(h, dict) else 1
                h_title = re.sub(r"^\[P\d+\]\s*", "", str(h_title)).strip()
                if h_title:
                    doc_elements.append({
                        "type": "heading",
                        "text": correct_japanese_text(h_title),
                        "level": h_level,
                        "page": p_num
                    })

            # 本文
            body_text = p_data.get("body_text", "")
            if not body_text:
                body_regions = [r for r in p_data.get("regions", []) if r.get("type") in ["body", "paragraph", None]]
                body_regions.sort(key=lambda r: r.get("reading_order", 0))
                body_text = "\n".join([r.get("text", "") for r in body_regions if r.get("text")])

            body_text = re.sub(r"^=== ページ \d+ ===\s*", "", body_text).strip()

            if body_text:
                # 段落単位 (2つ以上の連続改行) で分割
                raw_para_blocks = [p.strip() for p in re.split(r'\n{2,}', body_text) if p.strip()]
                clean_paras = []
                for p_block in raw_para_blocks:
                    # 段落内の単一改行（ソフト改行）を適切に連結整形
                    # 1. 日本語/全角文字同士の改行はそのまま直接連結
                    p_clean = re.sub(r'([^\x00-\x7F])\n+([^\x00-\x7F])', r'\1\2', p_block)
                    # 2. 欧文単語同士の改行は半角スペースで連結
                    p_clean = re.sub(r'([a-zA-Z0-9])\n+([a-zA-Z0-9])', r'\1 \2', p_clean)
                    # 3. その他残った改行を自然に除去
                    p_clean = re.sub(r'\n+', '', p_clean).strip()
                    p_clean = correct_japanese_text(p_clean)
                    if p_clean:
                        clean_paras.append(p_clean)

                for p_idx, p_str in enumerate(clean_paras):
                    doc_elements.append({
                        "type": "para",
                        "text": p_str,
                        "page": p_num,
                        "is_first_para_of_page": (p_idx == 0),
                        "is_last_para_of_page": (p_idx == len(clean_paras) - 1)
                    })

            # 表
            tables = p_data.get("tables", [])
            for tbl in tables:
                doc_elements.append({
                    "type": "table",
                    "data": tbl,
                    "page": p_num
                })

            # 図版
            figures = p_data.get("figures", [])
            for fig in figures:
                doc_elements.append({
                    "type": "figure",
                    "data": fig,
                    "page": p_num,
                    "page_dir": p_dir
                })

            # 注釈
            footnotes = p_data.get("footnotes", [])
            if not footnotes:
                fn_regions = [r for r in p_data.get("regions", []) if r.get("type") == "footnote"]
                footnotes = [r.get("text", "") for r in fn_regions if r.get("text")]
            for fn in footnotes:
                if isinstance(fn, dict):
                    fn_str = fn.get("text", "")
                else:
                    fn_str = str(fn)
                fn_str = re.sub(r"^\[P\d+\]\s*", "", fn_str).strip()
                if fn_str:
                    all_footnotes.append((p_num, correct_japanese_text(fn_str)))

        # 4. ページまたぎ結合処理 (merge_cross_page)
        merged_elements = []
        terminal_punctuations = ("。", "！", "？", "!", "?", "」", "』", "）", ")", "…", "―")

        for el in doc_elements:
            if merge_cross_page and el["type"] == "para" and el.get("is_first_para_of_page"):
                # 前の要素が段落であり、前ページの末尾段落で、句点で終わっていないかハイフンで終わっている場合
                if merged_elements and merged_elements[-1]["type"] == "para" and merged_elements[-1].get("is_last_para_of_page"):
                    prev_text = merged_elements[-1]["text"].rstrip()
                    if prev_text and not prev_text.endswith(terminal_punctuations):
                        if prev_text.endswith("-"):
                            merged_elements[-1]["text"] = prev_text[:-1] + el["text"]
                        elif re.search(r'[a-zA-Z0-9]$', prev_text) and re.match(r'^[a-zA-Z0-9]', el["text"]):
                            merged_elements[-1]["text"] = prev_text + " " + el["text"]
                        else:
                            merged_elements[-1]["text"] = prev_text + el["text"]
                        continue

            merged_elements.append(el)

        # 5. Word文書への書き出し
        current_rendered_page = None

        for el in merged_elements:
            el_page = el.get("page")

            # 改ページ処理 (insert_page_break)
            if insert_page_break and current_rendered_page is not None and el_page != current_rendered_page:
                docx_doc.add_page_break()
            current_rendered_page = el_page

            if el["type"] == "heading":
                lvl = min(max(el.get("level", 1), 1), 3)
                hp = docx_doc.add_heading(el["text"], level=lvl)
                hp.paragraph_format.space_before = Pt(12)
                hp.paragraph_format.space_after = Pt(4)

            elif el["type"] == "para":
                p_text = el["text"].strip()
                if p_text:
                    p = docx_doc.add_paragraph(p_text)
                    p.paragraph_format.line_spacing = 1.25
                    p.paragraph_format.space_after = Pt(4)

            elif el["type"] == "table":
                tbl = el["data"]
                caption = tbl.get("name") or f"表 (P.{el_page})"
                rows = tbl.get("rows", [])
                if rows and isinstance(rows, list):
                    cp = docx_doc.add_paragraph(f"📊 {caption}")
                    cp.paragraph_format.space_before = Pt(8)
                    cp.paragraph_format.space_after = Pt(2)
                    if cp.runs:
                        cp.runs[0].bold = True

                    col_cnt = max(len(r) for r in rows if isinstance(r, list)) if rows else 1
                    t = docx_doc.add_table(rows=len(rows), cols=col_cnt)
                    t.style = "Table Grid"
                    for r_idx, row in enumerate(rows):
                        if not isinstance(row, list): continue
                        for c_idx, val in enumerate(row):
                            if c_idx < col_cnt:
                                cell = t.cell(r_idx, c_idx)
                                cell.text = str(val if val is not None else "")
                                if r_idx == 0:
                                    for run in cell.paragraphs[0].runs:
                                        run.bold = True
                    docx_doc.add_paragraph().paragraph_format.space_after = Pt(6)

            elif el["type"] == "figure":
                fig = el["data"]
                caption = fig.get("name") or f"図版 (P.{el_page})"
                p_dir = el.get("page_dir")
                img_stream = None

                # 画像データの取得 (Base64 または ディスクファイル)
                b64 = fig.get("image_base64", "")
                if b64 and isinstance(b64, str) and b64.startswith("data:image"):
                    try:
                        _, encoded = b64.split(",", 1) if "," in b64 else ("", b64)
                        img_stream = io.BytesIO(base64.b64decode(encoded))
                    except Exception:
                        pass

                if not img_stream:
                    if p_dir:
                        fname = fig.get("file_name", "")
                        if fname and (p_dir / fname).exists():
                            img_stream = str(p_dir / fname)
                        else:
                            fig_dir = p_dir.parent / "figures"
                            if fig_dir.exists():
                                clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(fig.get("id", "")))
                                matches = list(fig_dir.glob(f"*{clean_id}*"))
                                if matches and matches[0].exists():
                                    img_stream = str(matches[0])
                    elif doc_dir.exists():
                        fig_dir = doc_dir / "figures"
                        page_specific_dir = doc_dir / f"page_{el_page:04d}"
                        fname = fig.get("file_name", "")
                        if fname and (page_specific_dir / fname).exists():
                            img_stream = str(page_specific_dir / fname)
                        elif fig_dir.exists():
                            clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(fig.get("id", "")))
                            matches = list(fig_dir.glob(f"*{clean_id}*"))
                            if matches and matches[0].exists():
                                img_stream = str(matches[0])

                if img_stream:
                    try:
                        docx_doc.add_paragraph().paragraph_format.space_before = Pt(6)
                        docx_doc.add_picture(img_stream, width=Inches(5.2))
                        last_p = docx_doc.paragraphs[-1]
                        last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        
                        cap_p = docx_doc.add_paragraph(f"🖼️ {caption}")
                        cap_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        cap_p.paragraph_format.space_after = Pt(8)
                        if cap_p.runs:
                            cap_p.runs[0].font.size = Pt(9.5)
                            cap_p.runs[0].font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
                    except Exception as ex:
                        print(f"Error embedding picture: {ex}")
                        docx_doc.add_paragraph(f"[図版: {caption}]")

        # 6. 注釈一覧セクション (文末にまとめて配置)
        if all_footnotes:
            docx_doc.add_page_break()
            fn_head = docx_doc.add_heading("注釈", level=1)
            fn_head.paragraph_format.space_before = Pt(14)
            fn_head.paragraph_format.space_after = Pt(8)

            for p_num, fn_text in all_footnotes:
                fn_p = docx_doc.add_paragraph(fn_text)
                fn_p.paragraph_format.line_spacing = 1.15
                fn_p.paragraph_format.space_after = Pt(4)

        # 7. バイナリストリーム返却
        out_stream = io.BytesIO()
        docx_doc.save(out_stream)
        out_stream.seek(0)
        return out_stream

