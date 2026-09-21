from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import re


class FigureExtractor:
    """図版・写真・地図の検出およびクロップ保存、キャプション紐付けを行うモジュール"""

    def __init__(self, media_dir: Optional[Path] = None):
        self.media_dir = media_dir or (Path(__file__).resolve().parent.parent.parent / "data" / "media")
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def crop_and_save(self, page_img_path: str | Path, box: List[int], doc_id: str, fig_idx: int) -> str:
        """指定矩形領域を画像として切り出して保存"""
        with Image.open(page_img_path) as img:
            x1, y1, x2, y2 = box
            # 範囲制限
            w, h = img.size
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            cropped = img.crop((x1, y1, x2, y2))
            out_filename = f"{doc_id}_crop_{fig_idx}.png"
            out_path = self.media_dir / out_filename
            cropped.save(out_path, format="PNG")
            return str(out_path)

    def find_caption_near_box(self, box: List[int], blocks: List[Dict[str, Any]]) -> Optional[str]:
        """図版領域の直下または直上にある「図1」「Figure 2」「地図」等のキャプションを探索"""
        x1, y1, x2, y2 = box
        caption_candidates = []

        caption_pattern = re.compile(r"^(?:図|Fig(?:ure)?|表|Table|地図|Map)\s*[\d\.\-\:]+", re.IGNORECASE)

        for b in blocks:
            bx = b.get("box", [])
            if len(bx) != 4:
                continue
            text = b.get("text", "").strip()
            if not text:
                continue

            # Y方向の近接度判定 (直下または直上 100px以内)
            bx1, by1, bx2, by2 = bx
            dist_bottom = abs(by1 - y2)
            dist_top = abs(y1 - by2)

            if dist_bottom < 120 or dist_top < 120:
                # パターンマッチまたは短文
                if caption_pattern.search(text) or len(text) < 60:
                    score = min(dist_bottom, dist_top)
                    caption_candidates.append((score, text))

        if caption_candidates:
            caption_candidates.sort(key=lambda c: c[0])
            return caption_candidates[0][1]

        return None
