import re
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image as PILImage

from app.importers.base_importer import BaseImporter


class OcrBatchImporter(BaseImporter):
    """
    data/ocr_results/<doc_name>/ に保存された全ページOCR結果（page_data.json等）を
    読み込み、ページまたぎ句点結合、見出し階層化、表・図版・注釈を正規化して
    ドキュメントバンドルを生成するインポーター。
    ※自動実行は行わず、ユーザーの明示的なリクエスト時のみ実行される。
    """

    def parse_batch_dir(self, batch_dir: Path) -> Dict[str, Any]:
        batch_dir = Path(batch_dir).resolve()
        if not batch_dir.exists():
            raise FileNotFoundError(f"OCR Batch folder not found: {batch_dir}")

        doc_name = batch_dir.name
        doc_id = f"doc-{uuid.uuid4().hex[:12]}"

        page_dirs = sorted([d for d in batch_dir.glob("page_*") if d.is_dir()])
        total_pages = len(page_dirs)

        sections: List[Dict[str, Any]] = []
        blocks: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        figures: List[Dict[str, Any]] = []
        annotations: List[Dict[str, Any]] = []

        root_sec_id = f"sec-{uuid.uuid4().hex[:12]}"
        sections.append({
            "id": root_sec_id,
            "parent_id": None,
            "title": doc_name,
            "level": 1,
            "order_idx": 0,
            "page_number": 1
        })

        sec_stack = [(1, root_sec_id)]
        current_sec_id = root_sec_id
        reading_order = 0
        order_idx_counter = 0

        # 全ページのデータを順次処理
        pending_body_text = ""
        pending_page_num = 1

        for p_dir in page_dirs:
            p_json_file = p_dir / "page_data.json"
            if not p_json_file.exists():
                continue

            try:
                with open(p_json_file, "r", encoding="utf-8") as f:
                    p_data = json.load(f)
            except Exception as e:
                print(f"Error reading {p_json_file}: {e}")
                continue

            page_num = p_data.get("page_number") or p_data.get("current_page") or 1

            # 1. 見出し（headings）
            p_headings = p_data.get("headings", [])
            for h_str in p_headings:
                clean_h = re.sub(r"^\[P\d+\]\s*", "", h_str).strip()
                if not clean_h:
                    continue

                h_level = 2
                if re.match(r"^(?:第[0-9一二三四五六七八九十]+章|[0-9]+章)", clean_h):
                    h_level = 1
                elif re.match(r"^[0-9]+\.[0-9]+", clean_h):
                    h_level = 2

                while sec_stack and sec_stack[-1][0] >= h_level:
                    sec_stack.pop()
                parent_id = sec_stack[-1][1] if sec_stack else root_sec_id

                order_idx_counter += 1
                new_sec_id = f"sec-{uuid.uuid4().hex[:12]}"
                sections.append({
                    "id": new_sec_id,
                    "parent_id": parent_id,
                    "title": clean_h,
                    "level": h_level,
                    "order_idx": order_idx_counter,
                    "page_number": page_num
                })
                sec_stack.append((h_level, new_sec_id))
                current_sec_id = new_sec_id

            # 2. 本文（body_text / regions）
            body_text = p_data.get("body_text", "")
            if not body_text:
                body_regions = [r for r in p_data.get("regions", []) if r.get("type") in ["body", "paragraph", None]]
                body_regions.sort(key=lambda r: r.get("reading_order", 0))
                body_text = "\n".join([r.get("text", "") for r in body_regions if r.get("text")])

            # ヘッダー除去
            body_text = re.sub(r"^=== ページ \d+ ===\s*", "", body_text).strip()

            if body_text:
                reading_order += 1
                blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                html = f"<p>{body_text.replace(chr(10), '<br>')}</p>"
                blocks.append({
                    "id": blk_id,
                    "section_id": current_sec_id,
                    "block_type": "paragraph",
                    "text_content": body_text,
                    "html_content": html,
                    "reading_order": reading_order,
                    "page_number": page_num,
                    "bbox_json": None
                })

            # 3. 表（tables）
            p_tables = p_data.get("tables", [])
            for tbl in p_tables:
                reading_order += 1
                tbl_id = tbl.get("id") or f"tbl-{uuid.uuid4().hex[:12]}"
                caption = tbl.get("name") or f"P.{page_num} の表"
                rows = tbl.get("rows", [])
                cols = tbl.get("columns", len(rows[0]) if rows else 0)
                row_cnt = tbl.get("row_count", len(rows))

                tables.append({
                    "id": tbl_id,
                    "section_id": current_sec_id,
                    "caption": caption,
                    "row_count": row_cnt,
                    "col_count": cols,
                    "grid_json": json.dumps({"rows": rows, "columns": cols, "row_count": row_cnt}, ensure_ascii=False),
                    "reading_order": reading_order,
                    "page_number": page_num
                })

                blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                blocks.append({
                    "id": blk_id,
                    "section_id": current_sec_id,
                    "block_type": "paragraph",
                    "text_content": f"[表: {caption}]",
                    "html_content": f'<div class="table-embed" data-table-id="{tbl_id}"><strong>📊 {caption}</strong></div>',
                    "reading_order": reading_order,
                    "page_number": page_num,
                    "bbox_json": None
                })

            # 4. 図版（figures）
            p_figures = p_data.get("figures", [])
            for fig in p_figures:
                reading_order += 1
                fig_id = fig.get("id") or f"fig-{uuid.uuid4().hex[:12]}"
                caption = fig.get("name") or f"P.{page_num} の図版"
                fpath = fig.get("file_path", "")
                
                # 相対パスを正規化
                if fpath and not fpath.startswith("/media/") and not fpath.startswith("http"):
                    # ディスク上のファイルが存在するか確認
                    abs_f = Path(BASE_DIR) / fpath
                    if abs_f.exists():
                        fpath = f"/media/{abs_f.name}"
                        # media フォルダへコピー
                        target_media = self.media_dir / abs_f.name
                        if not target_media.exists():
                            target_media.write_bytes(abs_f.read_bytes())

                figures.append({
                    "id": fig_id,
                    "section_id": current_sec_id,
                    "caption": caption,
                    "file_path": fpath,
                    "file_hash": "",
                    "file_size_kb": fig.get("file_size_kb", 0.0),
                    "width": fig.get("w", 0),
                    "height": fig.get("h", 0),
                    "reading_order": reading_order,
                    "page_number": page_num,
                    "is_custom": fig.get("is_custom_image", False)
                })

                blk_id = f"blk-{uuid.uuid4().hex[:12]}"
                blocks.append({
                    "id": blk_id,
                    "section_id": current_sec_id,
                    "block_type": "paragraph",
                    "text_content": f"[図版: {caption}]",
                    "html_content": f'<figure class="figure-embed" data-fig-id="{fig_id}"><img src="{fpath}" alt="{caption}"><figcaption>{caption}</figcaption></figure>',
                    "reading_order": reading_order,
                    "page_number": page_num,
                    "bbox_json": None
                })

            # 5. 脚注（footnotes）
            p_footnotes = p_data.get("footnotes", [])
            for fn in p_footnotes:
                clean_fn = re.sub(r"^\[P\d+\]\s*", "", fn).strip()
                if clean_fn:
                    ann_id = f"ann-{uuid.uuid4().hex[:12]}"
                    annotations.append({
                        "id": ann_id,
                        "block_id": None,
                        "kind": "footnote",
                        "anchor_text": f"P.{page_num} 脚注",
                        "target_value": clean_fn
                    })

        return {
            "document": {
                "id": doc_id,
                "title": doc_name,
                "source_type": "ocr_batch",
                "source_filename": f"{doc_name}.pdf",
                "total_pages": total_pages,
                "doc_metadata_json": {
                    "source_folder": str(batch_dir),
                    "total_pages": total_pages
                }
            },
            "sections": sections,
            "blocks": blocks,
            "tables": tables,
            "figures": figures,
            "annotations": annotations
        }

    def parse_file(self, file_path: Path) -> Dict[str, Any]:
        return self.parse_batch_dir(file_path)
