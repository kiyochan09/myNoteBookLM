import re
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
from PIL import Image as PILImage
import io

from app.importers.base_importer import BaseImporter


class PdfImporter(BaseImporter):
    """
    PDFファイルを解析し、デジタルPDFの多段組整序、画像抽出、見出し推定、段落結合を行って
    MyNotebookLM の正規化ドキュメントバンドルを生成するインポーター。
    """

    def parse_file(self, file_path: Path, original_filename: Optional[str] = None) -> Dict[str, Any]:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = fitz.open(str(file_path))
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        if original_filename:
            doc_title = Path(original_filename).stem
            source_file_name = Path(original_filename).name
        else:
            clean_name = re.sub(r'^[0-9a-fA-F]{32}_', '', file_path.name)
            doc_title = Path(clean_name).stem
            source_file_name = clean_name
        total_pages = len(doc)

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

        # 全ページを走査
        for pno in range(total_pages):
            page_num = pno + 1
            page = doc[pno]
            rect = page.rect
            page_w = rect.width
            page_h = rect.height

            # 上下マージン（柱・ノンブル）の閾値
            top_margin = page_h * 0.08
            bottom_margin = page_h * 0.92

            page_dict = page.get_text("dict")
            raw_blocks = page_dict.get("blocks", [])

            extracted_items = []

            for b in raw_blocks:
                b_type = b.get("type", 0)
                bbox = b.get("bbox", [0, 0, 0, 0])
                y0, y1 = bbox[1], bbox[3]

                if y1 < top_margin or y0 > bottom_margin:
                    continue

                if b_type == 0:  # テキストブロック
                    lines_text = []
                    max_font_size = 0.0
                    is_bold = False

                    for line in b.get("lines", []):
                        line_str = "".join([span.get("text", "") for span in line.get("spans", [])])
                        if line_str.strip():
                            lines_text.append(line_str.strip())
                        for span in line.get("spans", []):
                            sz = span.get("size", 10.0)
                            if sz > max_font_size:
                                max_font_size = sz
                            flags = span.get("flags", 0)
                            if flags & 2 or "Bold" in span.get("font", ""):
                                is_bold = True

                    if lines_text:
                        full_text = ""
                        for l in lines_text:
                            if not full_text:
                                full_text = l
                            else:
                                if full_text.endswith(("。", "．", ".", "!", "?", "！", "？", ":", "：")):
                                    full_text += "\n" + l
                                else:
                                    full_text += l

                        extracted_items.append({
                            "kind": "text",
                            "bbox": bbox,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "text": full_text.strip(),
                            "font_size": max_font_size,
                            "is_bold": is_bold
                        })

                elif b_type == 1:  # 画像ブロック
                    img_bytes = b.get("image")
                    ext = b.get("ext", "png")
                    if img_bytes:
                        extracted_items.append({
                            "kind": "image",
                            "bbox": bbox,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "img_bytes": img_bytes,
                            "ext": ext
                        })

            # PyMuPDF の get_images でも補完チェック
            page_images = page.get_images(full=True)
            if not any(it["kind"] == "image" for it in extracted_items) and page_images:
                for img_info in page_images:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    img_bytes = base_image.get("image")
                    ext = base_image.get("ext", "png")
                    if img_bytes:
                        extracted_items.append({
                            "kind": "image",
                            "bbox": [0, page_h * 0.5, page_w, page_h * 0.8],
                            "x0": 0,
                            "y0": page_h * 0.5,
                            "img_bytes": img_bytes,
                            "ext": ext
                        })

            # 多段組判定
            mid_x = page_w / 2.0
            left_col = [it for it in extracted_items if it["x0"] < mid_x and it["bbox"][2] < (mid_x + 50)]
            right_col = [it for it in extracted_items if it["x0"] >= (mid_x - 50)]
            full_col = [it for it in extracted_items if it not in left_col and it not in right_col]

            if len(left_col) >= 2 and len(right_col) >= 2:
                left_col.sort(key=lambda it: it["y0"])
                right_col.sort(key=lambda it: it["y0"])
                full_col.sort(key=lambda it: it["y0"])
                sorted_items = sorted(full_col, key=lambda it: it["y0"]) + left_col + right_col
            else:
                sorted_items = sorted(extracted_items, key=lambda it: it["y0"])

            # アイテムの登録
            for it in sorted_items:
                if it["kind"] == "image":
                    saved = self.save_image_bytes(it["img_bytes"], original_ext=it["ext"], prefix=f"p{page_num}")
                    fig_id = f"fig-{uuid.uuid4().hex[:12]}"
                    fig_count = len(figures) + 1
                    caption = f"P.{page_num} 図版 {fig_count}"
                    reading_order += 1

                    w = int(it["bbox"][2] - it["bbox"][0]) or 400
                    h = int(it["bbox"][3] - it["bbox"][1]) or 300

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
                        "page_number": page_num,
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
                        "page_number": page_num,
                        "bbox_json": json.dumps(it["bbox"])
                    })

                elif it["kind"] == "text":
                    text = it["text"]
                    if not text:
                        continue

                    reading_order += 1
                    font_sz = it.get("font_size", 10.0)
                    is_bold = it.get("is_bold", False)

                    is_heading = False
                    h_level = 1

                    heading_pats = [
                        (r"^(?:第[0-9一二三四五六七八九十]+章|[0-9]+\.[0-9]+|[0-9]+章)", 1),
                        (r"^[0-9]+\.[0-9]+\.[0-9]+", 2),
                        (r"^[■◆●★・]\s*", 2)
                    ]
                    for pat, lvl in heading_pats:
                        if re.match(pat, text):
                            is_heading = True
                            h_level = lvl
                            break

                    if not is_heading and (font_sz >= 15.0 or (font_sz >= 13.0 and is_bold and len(text) < 50)):
                        is_heading = True
                        h_level = 2 if font_sz < 16.0 else 1

                    if is_heading:
                        while sec_stack and sec_stack[-1][0] >= h_level:
                            sec_stack.pop()
                        parent_id = sec_stack[-1][1] if sec_stack else root_sec_id

                        order_idx_counter += 1
                        new_sec_id = f"sec-{uuid.uuid4().hex[:12]}"
                        sections.append({
                            "id": new_sec_id,
                            "parent_id": parent_id,
                            "title": text[:80],
                            "level": h_level,
                            "order_idx": order_idx_counter,
                            "page_number": page_num
                        })
                        sec_stack.append((h_level, new_sec_id))
                        current_sec_id = new_sec_id

                        blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                        blocks.append({
                            "id": blk_id,
                            "section_id": current_sec_id,
                            "block_type": "heading",
                            "text_content": text,
                            "html_content": f"<h{h_level}>{text}</h{h_level}>",
                            "reading_order": reading_order,
                            "page_number": page_num,
                            "bbox_json": json.dumps(it["bbox"])
                        })
                    else:
                        blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                        html = f"<p>{text.replace(chr(10), '<br>')}</p>"
                        blocks.append({
                            "id": blk_id,
                            "section_id": current_sec_id,
                            "block_type": "paragraph",
                            "text_content": text,
                            "html_content": html,
                            "reading_order": reading_order,
                            "page_number": page_num,
                            "bbox_json": json.dumps(it["bbox"])
                        })

        doc.close()

        return {
            "document": {
                "id": doc_id,
                "title": doc_title,
                "source_type": "pdf_digital",
                "source_filename": source_file_name,
                "total_pages": total_pages,
                "doc_metadata_json": {
                    "original_path": str(file_path),
                    "total_pages": total_pages,
                    "file_size_bytes": file_path.stat().st_size
                }
            },
            "sections": sections,
            "blocks": blocks,
            "tables": tables,
            "figures": figures,
            "annotations": annotations
        }
