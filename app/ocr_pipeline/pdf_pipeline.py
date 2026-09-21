import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List

from app.core.db import get_db_connection
from app.ocr_pipeline.pdf_renderer import PdfRenderer
from app.ocr_pipeline.table_extractor import TableExtractor
from app.ocr_pipeline.figure_extractor import FigureExtractor
from app.ocr_pipeline.block_classifier import BlockClassifier


class PdfPipeline:
    """PDFを取り込み、文書構造化解析を行って共通SQLiteデータベースに登録する統合パイプライン"""

    def __init__(self, db_path: Optional[Path] = None, media_dir: Optional[Path] = None):
        self.db_path = db_path
        self.media_dir = media_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "media")
        self.renderer = PdfRenderer(output_dir=self.media_dir)
        self.table_extractor = TableExtractor()
        self.figure_extractor = FigureExtractor(media_dir=self.media_dir)
        self.classifier = BlockClassifier()

    def process_pdf(self, pdf_path: str | Path, notebook_id: str = "nb-default", ocr_json_path: Optional[str | Path] = None, max_pages: Optional[int] = None) -> Dict[str, Any]:
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc_id = f"doc-{uuid.uuid4().hex[:12]}"
        doc_title = pdf_path.stem

        # 1. ページレンダリング & ブロック抽出
        pages_data = self.renderer.render_and_extract(pdf_path, doc_id, ocr_json_path=ocr_json_path, max_pages=max_pages)

        conn = get_db_connection(self.db_path)
        cur = conn.cursor()

        stats = {
            "document_id": doc_id,
            "title": doc_title,
            "page_count": len(pages_data),
            "tree_nodes_count": 0,
            "body_blocks_count": 0,
            "tables_count": 0,
            "figures_count": 0,
            "footnotes_count": 0,
            "pages": []
        }

        try:
            # documents レコード作成
            cur.execute(
                """
                INSERT INTO documents (id, title, file_path, source_type, page_count, metadata_json)
                VALUES (?, ?, ?, 'pdf_ocr', ?, ?)
                """,
                (doc_id, doc_title, str(pdf_path), len(pages_data), json.dumps({"filename": pdf_path.name}))
            )

            # ルートノード (KaisoRエディタのドキュメントノード)
            root_node_id = f"node-{uuid.uuid4().hex[:12]}"
            cur.execute(
                """
                INSERT INTO tree_nodes (id, notebook_id, parent_id, title, node_type, sort_order)
                VALUES (?, ?, NULL, ?, 'rich', 0)
                """,
                (root_node_id, notebook_id, doc_title)
            )
            stats["tree_nodes_count"] += 1

            node_stack = [(0, root_node_id)]
            current_node_id = root_node_id
            sort_order_counter = 0
            global_reading_order = 0

            # ページ毎の処理
            for p_info in pages_data:
                p_no = p_info["page_number"]
                page_id = f"pg-{uuid.uuid4().hex[:12]}"
                pw = p_info["width"]
                ph = p_info["height"]

                cur.execute(
                    """
                    INSERT INTO pages (id, document_id, page_number, width, height, image_path)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (page_id, doc_id, p_no, pw, ph, p_info["image_path"])
                )

                page_debug_blocks = []
                raw_blocks = p_info["blocks"]

                # 2. 表の検出と抽出
                table_regions = self.table_extractor.detect_table_regions(p_info["image_path"])
                extracted_tables = []
                table_consumed_boxes = []

                for tr in table_regions:
                    tbl_data = self.table_extractor.extract_table_from_blocks(tr, raw_blocks)
                    if tbl_data:
                        extracted_tables.append(tbl_data)
                        table_consumed_boxes.append(tr["box"])

                # 3. 画像ブロックの検出とクロップ
                image_blocks = [b for b in raw_blocks if b.get("type") == "image"]
                extracted_figures = []

                for fig_idx, ib in enumerate(image_blocks, start=1):
                    # 極端に小さい画像（飾りアイコンなど）は除外
                    bw = ib["box"][2] - ib["box"][0]
                    bh = ib["box"][3] - ib["box"][1]
                    if bw > 50 and bh > 50:
                        crop_path = self.figure_extractor.crop_and_save(p_info["image_path"], ib["box"], doc_id, f"p{p_no}_{fig_idx}")
                        cap = self.figure_extractor.find_caption_near_box(ib["box"], raw_blocks) or f"図 (p.{p_no}-{fig_idx})"
                        extracted_figures.append({
                            "box": ib["box"],
                            "image_path": crop_path,
                            "caption": cap
                        })
                        table_consumed_boxes.append(ib["box"])

                # 4. 表や画像に含まれないテキストブロックの分類
                remaining_blocks = []
                for b in raw_blocks:
                    if b.get("type") == "image":
                        continue
                    # 表の内部に含まれているか判定
                    in_table = False
                    bx = b["box"]
                    cx = (bx[0] + bx[2]) / 2
                    cy = (bx[1] + bx[3]) / 2
                    for tb in table_consumed_boxes:
                        if tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]:
                            in_table = True
                            break
                    if not in_table:
                        remaining_blocks.append(b)

                classified_blocks = self.classifier.classify_and_order(remaining_blocks, pw, ph)

                # 5. 各ブロックをDBに格納
                # 5.1 テキストブロック (見出し・本文・脚注)
                for b in classified_blocks:
                    global_reading_order += 1
                    b_type = b["block_type"]
                    text = b["text"]
                    bbox_json = json.dumps({"x": b["box"][0], "y": b["box"][1], "w": b["box"][2] - b["box"][0], "h": b["box"][3] - b["box"][1]})

                    # 見出しの場合: 新規階層ツリーノードを生成
                    if b_type.startswith("h"):
                        lvl = int(b_type[1])
                        while node_stack and node_stack[-1][0] >= lvl:
                            node_stack.pop()
                        parent_id = node_stack[-1][1] if node_stack else root_node_id

                        sort_order_counter += 1
                        new_node_id = f"node-{uuid.uuid4().hex[:12]}"
                        cur.execute(
                            """
                            INSERT INTO tree_nodes (id, notebook_id, parent_id, title, node_type, sort_order)
                            VALUES (?, ?, ?, ?, 'rich', ?)
                            """,
                            (new_node_id, notebook_id, parent_id, text[:60], sort_order_counter)
                        )
                        node_stack.append((lvl, new_node_id))
                        current_node_id = new_node_id
                        stats["tree_nodes_count"] += 1

                    block_id = f"blk-{uuid.uuid4().hex[:12]}"
                    html = f"<{b_type}>{text}</{b_type}>" if b_type.startswith("h") else f"<p>{text}</p>"

                    cur.execute(
                        """
                        INSERT INTO content_blocks (id, document_id, page_id, node_id, block_type, reading_order, text_content, html_content, bbox_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (block_id, doc_id, page_id, current_node_id, b_type, global_reading_order, text, html, bbox_json)
                    )

                    if b_type == "body":
                        stats["body_blocks_count"] += 1
                    elif b_type == "footnote":
                        stats["footnotes_count"] += 1

                    page_debug_blocks.append({
                        "id": block_id,
                        "type": b_type,
                        "box": b["box"],
                        "text": text
                    })

                # 5.2 表の格納
                for tbl in extracted_tables:
                    global_reading_order += 1
                    block_id = f"blk-{uuid.uuid4().hex[:12]}"
                    tbl_id = f"tbl-{uuid.uuid4().hex[:12]}"
                    bbox_json = json.dumps({"x": tbl["box"][0], "y": tbl["box"][1], "w": tbl["box"][2] - tbl["box"][0], "h": tbl["box"][3] - tbl["box"][1]})
                    tbl_text = " | ".join(tbl["headers"]) + "\n" + "\n".join([" | ".join([c["value"] for c in r]) for r in tbl["matrix"]])

                    cur.execute(
                        """
                        INSERT INTO content_blocks (id, document_id, page_id, node_id, block_type, reading_order, text_content, html_content, bbox_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (block_id, doc_id, page_id, current_node_id, "table", global_reading_order, tbl_text, f'<div id="{tbl_id}" class="table-placeholder"></div>', bbox_json)
                    )

                    cur.execute(
                        """
                        INSERT INTO tables (id, block_id, row_count, col_count, headers_json, matrix_json, has_header_row)
                        VALUES (?, ?, ?, ?, ?, ?, 1)
                        """,
                        (tbl_id, block_id, tbl["row_count"], tbl["col_count"], json.dumps(tbl["headers"], ensure_ascii=False), json.dumps(tbl["matrix"], ensure_ascii=False))
                    )
                    stats["tables_count"] += 1
                    page_debug_blocks.append({"id": block_id, "type": "table", "box": tbl["box"], "text": f"Table ({tbl['row_count']}x{tbl['col_count']})"})

                # 5.3 図版の格納
                for fig in extracted_figures:
                    global_reading_order += 1
                    block_id = f"blk-{uuid.uuid4().hex[:12]}"
                    fig_id = f"fig-{uuid.uuid4().hex[:12]}"
                    bbox_json = json.dumps({"x": fig["box"][0], "y": fig["box"][1], "w": fig["box"][2] - fig["box"][0], "h": fig["box"][3] - fig["box"][1]})

                    cur.execute(
                        """
                        INSERT INTO content_blocks (id, document_id, page_id, node_id, block_type, reading_order, text_content, html_content, bbox_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (block_id, doc_id, page_id, current_node_id, "figure", global_reading_order, fig["caption"], f'<figure id="{fig_id}"><img src="{fig["image_path"]}"><figcaption>{fig["caption"]}</figcaption></figure>', bbox_json)
                    )

                    cur.execute(
                        """
                        INSERT INTO figures (id, block_id, figure_type, image_path, caption, anchor_id)
                        VALUES (?, ?, 'figure', ?, ?, ?)
                        """,
                        (fig_id, block_id, fig["image_path"], fig["caption"], fig_id)
                    )
                    stats["figures_count"] += 1
                    page_debug_blocks.append({"id": block_id, "type": "figure", "box": fig["box"], "text": fig["caption"]})

                stats["pages"].append({
                    "page_number": p_no,
                    "image_path": p_info["image_path"],
                    "width": pw,
                    "height": ph,
                    "blocks": page_debug_blocks
                })

            conn.commit()
            return stats

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
