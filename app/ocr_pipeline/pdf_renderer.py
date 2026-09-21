import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz  # PyMuPDF


class PdfRenderer:
    """PDFの各ページを高解像度画像としてレンダリングし、埋め込みテキストや画像情報を取得するクラス"""

    def __init__(self, output_dir: Optional[Path] = None, dpi: int = 200):
        self.output_dir = output_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "media")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def render_and_extract(
        self,
        pdf_path: str | Path,
        doc_id: str,
        ocr_json_path: Optional[str | Path] = None,
        max_pages: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path).resolve()
        doc = fitz.open(str(pdf_path))
        pages_info = []

        total = len(doc) if max_pages is None else min(len(doc), max_pages)

        # 外部OCR JSON (NDLOCR-Lite出力形式) が指定されているか確認
        ocr_data = None
        if ocr_json_path and Path(ocr_json_path).exists():
            with open(ocr_json_path, "r", encoding="utf-8") as f:
                ocr_data = json.load(f)

        for pno in range(total):
            page = doc[pno]
            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_filename = f"{doc_id}_page_{pno + 1}.png"
            img_path = self.output_dir / img_filename
            pix.save(str(img_path))

            extracted_blocks = []

            # 1. 外部OCR JSONがある場合
            if ocr_data and "blocks" in ocr_data:
                for b in ocr_data["blocks"]:
                    box = b.get("box", [0, 0, 100, 30])
                    extracted_blocks.append({
                        "id": b.get("id", len(extracted_blocks) + 1),
                        "type": "text",
                        "text": b.get("text", "").strip(),
                        "box": box
                    })
            else:
                # 2. PyMuPDF によるテキストブロック抽出
                raw_blocks = page.get_text("blocks")
                for b in raw_blocks:
                    x0, y0, x1, y1, text, bno, btype = b
                    box = [int(x0 * zoom), int(y0 * zoom), int(x1 * zoom), int(y1 * zoom)]
                    txt = text.strip() if isinstance(text, str) else ""
                    if txt or btype == 1:
                        extracted_blocks.append({
                            "id": bno,
                            "type": "text" if btype == 0 else "image",
                            "text": txt,
                            "box": box
                        })

                # 3. ページ内の埋め込み画像オブジェクトも抽出
                img_list = page.get_images(full=True)
                for img_idx, img_info in enumerate(img_list, start=1):
                    xref = img_info[0]
                    rects = page.get_image_rects(xref)
                    for r in rects:
                        box = [int(r.x0 * zoom), int(r.y0 * zoom), int(r.x1 * zoom), int(r.y1 * zoom)]
                        # 既存ブロックとの重複を避ける
                        extracted_blocks.append({
                            "id": 1000 + img_idx,
                            "type": "image",
                            "text": f"Embedded Image {img_idx}",
                            "box": box
                        })

            pages_info.append({
                "page_number": pno + 1,
                "width": pix.width,
                "height": pix.height,
                "image_path": str(img_path),
                "blocks": extracted_blocks
            })

        doc.close()
        return pages_info
