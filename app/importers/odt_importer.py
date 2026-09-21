import io
import re
import json
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image as PILImage

from app.importers.base_importer import BaseImporter


class OdtImporter(BaseImporter):
    """
    ODT (OpenDocument Text) ファイルを解凍・パースし、
    インライン画像、見出しツリー、本文、表を抽出してドキュメントバンドルを生成するインポーター。
    """

    NS = {
        "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
        "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
        "xlink": "http://www.w3.org/1999/xlink",
        "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
        "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
        "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0"
    }

    def _extract_all_text(self, elem: ET.Element) -> str:
        texts = []
        if elem.text:
            texts.append(elem.text)
        for child in elem:
            texts.append(self._extract_all_text(child))
            if child.tail:
                texts.append(child.tail)
        return "".join(texts)

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        doc_title = file_path.stem

        sections: List[Dict[str, Any]] = []
        blocks: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        figures: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

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

        extracted_pics_set = set()

        with zipfile.ZipFile(str(file_path), "r") as z:
            content_xml = z.read("content.xml")
            root = ET.fromstring(content_xml)
            body = root.find("office:body", self.NS)
            text_body = body.find("office:text", self.NS) if body is not None else None

            if text_body is not None:
                for elem in text_body:
                    tag = elem.tag

                    # 1. 見出し: <text:h>
                    if tag.endswith("h"):
                        outline_lvl = elem.attrib.get(f"{{{self.NS['text']}}}outline-level", "1")
                        try:
                            h_level = int(outline_lvl)
                        except ValueError:
                            h_level = 1

                        h_text = self._extract_all_text(elem).strip()
                        if not h_text:
                            continue

                        reading_order += 1
                        while sec_stack and sec_stack[-1][0] >= h_level:
                            sec_stack.pop()
                        parent_id = sec_stack[-1][1] if sec_stack else root_sec_id

                        order_idx_counter += 1
                        new_sec_id = f"sec-{uuid.uuid4().hex[:12]}"
                        sections.append({
                            "id": new_sec_id,
                            "parent_id": parent_id,
                            "title": h_text,
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
                            "text_content": h_text,
                            "html_content": f"<h{h_level}>{h_text}</h{h_level}>",
                            "reading_order": reading_order,
                            "page_number": 1,
                            "bbox_json": None
                        })

                    # 2. 段落: <text:p> (インライン画像も検出)
                    elif tag.endswith("p"):
                        # 段落内の画像 <draw:image> 検出
                        draw_images = elem.findall(".//draw:image", self.NS)
                        for d_img in draw_images:
                            href = d_img.attrib.get(f"{{{self.NS['xlink']}}}href", "")
                            if href and href in z.namelist():
                                img_bytes = z.read(href)
                                ext = Path(href).suffix or ".png"
                                saved = self.save_image_bytes(img_bytes, original_ext=ext, prefix="odt")
                                extracted_pics_set.add(href)

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
                                blocks.append({
                                    "id": blk_id,
                                    "section_id": current_sec_id,
                                    "block_type": "paragraph",
                                    "text_content": f"[図版: {caption}]",
                                    "html_content": f'<figure class="figure-embed" data-fig-id="{fig_id}"><img src="{saved["file_path"]}" alt="{caption}"><figcaption>{caption}</figcaption></figure>',
                                    "reading_order": reading_order,
                                    "page_number": 1,
                                    "bbox_json": None
                                })

                        p_text = self._extract_all_text(elem).strip()
                        if p_text:
                            reading_order += 1
                            blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                            html = f"<p>{p_text}</p>"
                            blocks.append({
                                "id": blk_id,
                                "section_id": current_sec_id,
                                "block_type": "paragraph",
                                "text_content": p_text,
                                "html_content": html,
                                "reading_order": reading_order,
                                "page_number": 1,
                                "bbox_json": None
                            })

                    # 3. 表: <table:table>
                    elif tag.endswith("table"):
                        reading_order += 1
                        raw_rows: List[List[str]] = []
                        for row_elem in elem.findall("table:table-row", self.NS):
                            row_cells = []
                            for cell_elem in row_elem.findall("table:table-cell", self.NS):
                                cell_text = self._extract_all_text(cell_elem).strip()
                                row_cells.append(cell_text)
                            if row_cells:
                                raw_rows.append(row_cells)

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

                            blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                            table_summary = " | ".join(headers) + "\n" + "\n".join([" | ".join(r) for r in body_rows])
                            blocks.append({
                                "id": blk_id,
                                "section_id": current_sec_id,
                                "block_type": "paragraph",
                                "text_content": f"[表: {caption}]\n{table_summary}",
                                "html_content": f'<div class="table-embed" data-table-id="{tbl_id}"><strong>📊 {caption}</strong></div>',
                                "reading_order": reading_order,
                                "page_number": 1,
                                "bbox_json": None
                            })

            # 4. 未抽出の Pictures/ 配下の画像フォールバック
            pic_names = [n for n in z.namelist() if n.startswith("Pictures/") and not n.endswith("/")]
            for pic_name in pic_names:
                if pic_name not in extracted_pics_set:
                    img_bytes = z.read(pic_name)
                    ext = Path(pic_name).suffix or ".png"
                    saved = self.save_image_bytes(img_bytes, original_ext=ext, prefix="odt")

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
                    blocks.append({
                        "id": blk_id,
                        "section_id": current_sec_id,
                        "block_type": "paragraph",
                        "text_content": f"[図版: {caption}]",
                        "html_content": f'<figure class="figure-embed" data-fig-id="{fig_id}"><img src="{saved["file_path"]}" alt="{caption}"><figcaption>{caption}</figcaption></figure>',
                        "reading_order": reading_order,
                        "page_number": 1,
                        "bbox_json": None
                    })

        return {
            "document": {
                "id": doc_id,
                "title": doc_title,
                "source_type": "odt",
                "source_filename": file_path.name,
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
