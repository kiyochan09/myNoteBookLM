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

from app.ocr_footnotes import WordFootnotes, note_id
from app.db_service import DBService
from app.paths import APP_PATHS
BASE_DIR = APP_PATHS.root
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
        from app.rich_docx import export_rich_docx
        return export_rich_docx(self.get_bundle(doc_id), BASE_DIR)

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
        doc_dir = APP_PATHS.ocr_results / clean_stem
        if not doc_dir.exists():
            # 拡張子なし/ありの両方を探索
            doc_dir = APP_PATHS.ocr_results / pdf_stem

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
        word_notes = WordFootnotes(docx_doc, all_pages_data, doc_dir, memory_pages)
        all_footnotes = []

        # ページごとの本文段落リスト
        doc_elements = []

        for p_num, p_data, p_dir in all_pages_data:
            # 見出しタブの編集を優先。同じ領域の旧 OCR 文字は別見出しにしない。
            def heading_title(value):
                value = re.sub(r"^\[P\d+\]\s*", "", str(value or "")).strip()
                return correct_japanese_text(re.sub(r"^【(?:大|中|小)?見出し】\s*", "", value).strip())

            headings_raw = p_data.get("headings", [])
            heading_regions = [r for r in p_data.get("regions", []) if r.get("type") == "heading"]
            region_headings = {}
            matched_records = set()
            for index, h in enumerate(headings_raw):
                title = heading_title((h.get("title") if "title" in h else h.get("text")) if isinstance(h, dict) else h)
                region_key = h.get("region_id", h.get("id")) if isinstance(h, dict) else None
                bbox = h.get("bbox") if isinstance(h, dict) else None
                candidates = [i for i, r in enumerate(heading_regions) if i not in region_headings and (
                    (region_key is not None and r.get("id") == region_key) or
                    (region_key is None and bbox is not None and bbox == [r.get("x"), r.get("y"), r.get("x", 0)+r.get("w", 0), r.get("y", 0)+r.get("h", 0)]))]
                if not candidates and region_key is None:
                    candidates = [i for i, r in enumerate(heading_regions) if i not in region_headings and heading_title(r.get("text")) == title]
                if len(candidates) == 1:
                    region_headings[candidates[0]] = h
                    matched_records.add(index)
            # 識別子がない旧データも、残る組が一対だけなら安全に対応できる。
            remaining_regions = [i for i in range(len(heading_regions)) if i not in region_headings]
            remaining_records = [i for i in range(len(headings_raw)) if i not in matched_records
                and (not isinstance(headings_raw[i], dict) or not headings_raw[i].get("region_id", headings_raw[i].get("id")))]
            if len(remaining_regions) == len(remaining_records) == 1:
                region_headings[remaining_regions[0]] = headings_raw[remaining_records[0]]

            headings_map, heading_aliases, region_titles = {}, {}, {}
            for h in headings_raw:
                title = heading_title((h.get("title") if "title" in h else h.get("text")) if isinstance(h, dict) else h)
                if title:
                    headings_map[title] = h.get("level", 1) if isinstance(h, dict) else 1
            for i, r in enumerate(heading_regions):
                old = heading_title(r.get("text"))
                if i in region_headings:
                    h = region_headings[i]
                    title = heading_title((h.get("title") if "title" in h else h.get("text")) if isinstance(h, dict) else h)
                    if old and old != title:
                        heading_aliases[old] = title
                else:
                    title = old
                    if title:
                        headings_map.setdefault(title, 2)
                region_titles[id(r)] = title
            emitted_headings = set()
            page_elements = []

            # 2. 本文テキストの取得: 真実のソース（Source of Truth）はユーザーが編集・保存した body_text
            b_text = p_data.get("body_text", "")
            if not b_text and p_dir:
                txt_file = p_dir / "body_reading_order.txt"
                if txt_file.exists():
                    try:
                        b_text = txt_file.read_text(encoding="utf-8")
                    except Exception:
                        pass

            b_text = re.sub(r"^=== ページ \d+ ===\s*", "", (b_text or "")).strip()

            japanese_particles = set(["の", "に", "を", "で", "が", "と", "は", "へ", "から", "まで", "など", "より", "やら"])

            if b_text:
                # body_text を元に段落・見出しを構築
                # 改行（\n）で区切られた各行を尊重し、ページ内で勝手に前後の文に接続（結合）させない
                raw_lines = [ln.strip() for ln in b_text.split('\n') if ln.strip()]
                for ln in raw_lines:
                    # 見出し判定 (マーカー)
                    m_midashi = re.match(r"^【(?:大|中|小)?見出し】\s*(.+)$", ln)
                    if m_midashi:
                        m_title = m_midashi.group(1).strip()
                        clean_m = heading_aliases.get(heading_title(m_title), heading_title(m_title))
                        if not clean_m or clean_m in emitted_headings:
                            continue
                        h_lvl = headings_map.get(clean_m, headings_map.get(m_title, 2))
                        page_elements.append({
                            "type": "heading",
                            "text": clean_m,
                            "level": h_lvl,
                            "page": p_num
                        })
                        emitted_headings.add(clean_m)
                        emitted_headings.add(m_title)
                        continue

                    # 見出し判定 (辞書マッチ)
                    clean_ln = correct_japanese_text(ln)
                    if heading_title(ln) in heading_aliases:
                        clean_ln = heading_aliases[heading_title(ln)]
                        if not clean_ln or clean_ln in emitted_headings:
                            continue
                    if clean_ln in headings_map and (len(ln) < 40):
                        if clean_ln in emitted_headings:
                            continue
                        h_lvl = headings_map.get(clean_ln, headings_map.get(ln, 2))
                        page_elements.append({
                            "type": "heading",
                            "text": clean_ln,
                            "level": h_lvl,
                            "page": p_num
                        })
                        emitted_headings.add(clean_ln)
                        emitted_headings.add(ln)
                        continue

                    # 通常段落: 改行されている各行は独立した段落として保持（前後の文へ勝手に接続しない）
                    p_clean = correct_japanese_text(clean_ln)
                    if p_clean:
                        page_elements.append({
                            "type": "para",
                            "text": p_clean,
                            "page": p_num
                        })
            elif p_data.get("regions"):
                # body_text が一切ない場合のみ、regions にフォールバック
                sorted_regs = sorted(p_data.get("regions", []), key=lambda r: r.get("reading_order", 0))
                for r in sorted_regs:
                    rtype = r.get("type", "body")
                    rtext = r.get("text", "").strip()
                    if not rtext:
                        continue
                    if rtype == "heading":
                        clean_h = re.sub(r"^\[P\d+\]\s*", "", rtext).strip()
                        clean_h = re.sub(r"^【(?:大|中|小)?見出し】\s*", "", clean_h).strip()
                        clean_h = region_titles.get(id(r), heading_title(clean_h))
                        if not clean_h or clean_h in emitted_headings:
                            continue
                        h_lvl = headings_map.get(clean_h, headings_map.get(rtext, 2))
                        page_elements.append({
                            "type": "heading",
                            "text": clean_h,
                            "level": h_lvl,
                            "page": p_num
                        })
                        emitted_headings.add(clean_h)
                        emitted_headings.add(rtext.strip())
                    elif rtype in ["body", "paragraph", None]:
                        lines = [ln.strip() for ln in rtext.split('\n') if ln.strip()]
                        for ln in lines:
                            p_clean = correct_japanese_text(ln)
                            if p_clean:
                                page_elements.append({
                                    "type": "para",
                                    "text": p_clean,
                                    "page": p_num
                                })

            # 本文にない見出しは、領域の読み順で本文の間に配置する。
            # 保存時に本文が最初の領域へ集約された旧データは OCR 行で境界を復元。
            def normalized(value):
                value = correct_japanese_text(str(value or ""))
                value = re.sub(r"⟦注:[^⟧]+⟧", "", value)
                return re.sub(r"\s+", "", value)

            stream = "\n".join(el["text"] for el in page_elements)
            compact, offsets = [], []
            token_spans = [m.span() for m in re.finditer(r"⟦注:[^⟧]+⟧", stream)]
            for pos, char in enumerate(stream):
                if not char.isspace() and not any(a <= pos < b for a, b in token_spans):
                    compact.append(char)
                    offsets.append(pos)
            compact = "".join(compact)

            def anchor(value, after=False):
                value = normalized(value)
                if not value:
                    return None
                # 短い一致は複数箇所に現れるため、位置が一意のときだけ使用。
                for length in (len(value), 24, 16, 10):
                    if length > len(value):
                        continue
                    piece = value[-length:] if after else value[:length]
                    at = compact.find(piece)
                    if at >= 0 and compact.find(piece, at + 1) < 0:
                        return offsets[at + len(piece) - 1] + 1 if after else offsets[at]
                return None

            def order(region):
                value = region.get("reading_order")
                return value if isinstance(value, (int, float)) else 999

            ordered = sorted(p_data.get("regions", []), key=order)
            body_types = ("body", "paragraph", None)

            def boundary(region, after=False):
                x, y = region.get("x", 0), region.get("y", 0)
                w, h = region.get("w", 0), region.get("h", 0)
                lines = []
                for line in p_data.get("ocr_lines", []):
                    cx = line.get("x", 0) + line.get("w", 0) / 2
                    cy = line.get("y", 0) + line.get("h", 0) / 2
                    if w > 0 and h > 0 and x <= cx <= x+w and y <= cy <= y+h:
                        lines.append(line)
                # OCR 行の本文内位置を使う。行配列の並びには依存しない。
                matches = [point for line in lines
                           if (point := anchor(line.get("text"), after)) is not None]
                if matches:
                    return max(matches) if after else min(matches)
                value = region.get("source_text") or region.get("text", "")
                value = re.sub(r"^=== ページ \d+ ===\s*", "", str(value))
                # 本文全体を持つ領域は内部の境界を示さない。
                if normalized(value) == normalized(b_text) and len([
                    r for r in ordered if r.get("type") in body_types]) > 1:
                    return None
                return anchor(value, after)

            insertions = []
            seen = {correct_japanese_text(title) for title in emitted_headings}
            for title, level in headings_map.items():
                title = correct_japanese_text(title)
                if title in seen:
                    continue
                seen.add(title)
                record = next((h for h in headings_raw if isinstance(h, dict)
                               and normalized(re.sub(r"^【(?:大|中|小)?見出し】\s*", "",
                                   re.sub(r"^\[P\d+\]\s*", "", str(h.get("title") or h.get("text") or "")))) == normalized(title)), {})
                region_index = next((i for i, r in enumerate(ordered)
                    if r.get("type") == "heading" and (
                        (record.get("region_id", record.get("id")) is not None and
                         r.get("id") == record.get("region_id", record.get("id"))) or
                        normalized(region_titles.get(id(r), r.get("text", ""))) == normalized(title))), None)
                point = None
                if region_index is not None:
                    for r in ordered[region_index+1:]:
                        if r.get("type") in body_types:
                            point = boundary(r)
                            if point is not None:
                                break
                    if point is None:
                        for r in reversed(ordered[:region_index]):
                            if r.get("type") in body_types:
                                point = boundary(r, after=True)
                                if point is not None:
                                    break
                if point is None:
                    point = len(stream) if region_index is not None and any(
                        r.get("type") in body_types for r in ordered[:region_index]) else 0
                insertions.append((point, region_index if region_index is not None else 999 + len(insertions),
                    {"type": "heading", "text": title, "level": level, "page": p_num}))

            # 編集済み本文をそのまま分割する。領域の古い本文で上書きしない。
            for point, _, heading in sorted(insertions, key=lambda item: (item[0], item[1]), reverse=True):
                cursor = 0
                for i, el in enumerate(page_elements):
                    end = cursor + len(el["text"])
                    if point <= end:
                        cut = max(0, point - cursor)
                        if el["type"] == "para" and 0 < cut < len(el["text"]):
                            left, right = el["text"][:cut].rstrip(), el["text"][cut:].lstrip()
                            page_elements[i:i+1] = ([{**el, "text": left}] if left else []) + [heading] + ([{**el, "text": right}] if right else [])
                        else:
                            page_elements.insert(i + (1 if cut > 0 else 0), heading)
                        break
                    cursor = end + 1
                else:
                    page_elements.append(heading)

            # ページ内最初と最後の段落にフラグを設定（ページまたぎ結合用）
            para_indices = [idx for idx, el in enumerate(page_elements) if el["type"] == "para"]
            if para_indices:
                page_elements[para_indices[0]]["is_first_para_of_page"] = True
                page_elements[para_indices[-1]]["is_last_para_of_page"] = True

            doc_elements.extend(page_elements)

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
            for fn_index, fn in enumerate(footnotes):
                if (p_num, note_id(fn, fn_index)) in word_notes.bound:
                    continue
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
                lvl = min(max(el.get("level", 1), 1), 9)
                hp = docx_doc.add_heading("", level=lvl)
                word_notes.write(hp, el["text"])
                hp.paragraph_format.space_before = Pt(12)
                hp.paragraph_format.space_after = Pt(4)

            elif el["type"] == "para":
                p_text = el["text"].strip()
                if p_text:
                    p = docx_doc.add_paragraph()
                    word_notes.write(p, p_text)
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
                caption = fig.get("caption") or fig.get("name") or f"図版 (P.{el_page})"
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
                        section = docx_doc.sections[-1]
                        max_width = min(Inches(5.2), section.page_width - section.left_margin - section.right_margin)
                        max_height = section.page_height - section.top_margin - section.bottom_margin - Inches(0.8)
                        with PILImage.open(img_stream) as source_image:
                            pixel_width, pixel_height = source_image.size
                        if hasattr(img_stream, "seek"):
                            img_stream.seek(0)
                        display_width = min(max_width, int(max_height * pixel_width / pixel_height))
                        picture = docx_doc.add_picture(img_stream, width=display_width)
                        picture._inline.docPr.set("title", "OCRFigureCaption")
                        picture._inline.docPr.set("descr", caption)
                        last_p = docx_doc.paragraphs[-1]
                        last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        last_p.paragraph_format.keep_with_next = True
                        
                        if "OCRFigureCaption" not in docx_doc.styles:
                            from docx.enum.style import WD_STYLE_TYPE
                            docx_doc.styles.add_style("OCRFigureCaption", WD_STYLE_TYPE.PARAGRAPH)
                        cap_p = docx_doc.add_paragraph(f"🖼️ {caption}", style="OCRFigureCaption")
                        cap_p.paragraph_format.keep_together = True
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

