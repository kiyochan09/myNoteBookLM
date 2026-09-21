import re
from typing import List, Dict, Any, Tuple


class BlockClassifier:
    """PDFページ内のテキストブロックを幾何情報とテキスト特徴から見出し・本文・脚注等に分類するモジュール"""

    def __init__(self):
        pass

    def classify_and_order(self, blocks: List[Dict[str, Any]], page_width: int, page_height: int) -> List[Dict[str, Any]]:
        """ブロックの分類と段組みを考慮した読み順ソート"""
        classified = []

        # 1. ページヘッダー・フッター・ページ番号の除外判定
        for b in blocks:
            box = b.get("box", [])
            if len(box) != 4:
                continue

            x1, y1, x2, y2 = box
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            text = b.get("text", "").strip()

            if not text:
                continue

            # ページ番号
            clean_num = text.replace(" ", "").replace("　", "")
            if len(clean_num) <= 5 and clean_num.isdigit():
                continue

            # 上端ヘッダー / 下端フッター (極端に端にある単行)
            top_ratio = y1 / page_height
            bottom_ratio = y2 / page_height
            if (top_ratio < 0.05 or bottom_ratio > 0.95) and len(text) < 40:
                continue

            # 脚注判定 (ページ下部 15% 以内にあり、注釈マーカーで始まる)
            if bottom_ratio > 0.82 and re.match(r"^(?:[\*\†\‡\(\[\（]?[0-9一二三四五]+[\)\]\）\.]?|注|※)", text):
                classified.append({
                    **b,
                    "category": "footnote",
                    "block_type": "footnote"
                })
                continue

            # 見出し判定 (行の高さ、短文、見出し番号パターン)
            is_heading = False
            heading_level = "h2"

            heading_patterns = [
                (r"^(?:第[0-9一二三四五六七八九十]+章|[0-9]+\.[0-9]+(?:\.[0-9]+)?|[0-9]+[\.\s])\s+", "h1"),
                (r"^(?:[A-Z0-9IVX]+\.\s+|[0-9]+\.[0-9]+)\s*", "h2"),
                (r"^(?:\([0-9]+\)|[a-z]\))\s*", "h3")
            ]

            for pat, lvl in heading_patterns:
                if re.match(pat, text):
                    is_heading = True
                    heading_level = lvl
                    break

            # フォントサイズ/高さが平均より大きい、または短く強調されている場合
            if not is_heading and len(text) < 40 and not text.endswith(("。", ".", "、", ",")):
                if h > 28 or (w < page_width * 0.5 and h > 22):
                    is_heading = True
                    heading_level = "h2"

            if is_heading:
                classified.append({
                    **b,
                    "category": "heading",
                    "block_type": heading_level
                })
            else:
                # 通常本文
                classified.append({
                    **b,
                    "category": "body",
                    "block_type": "body"
                })

        # 2. 段組みを考慮した読み順ソート (Reading Order)
        # 2段組み（カラム分割）の判定
        mid_x = page_width / 2.0
        left_col = []
        right_col = []
        full_width = []

        for b in classified:
            x1, y1, x2, y2 = b["box"]
            cx = (x1 + x2) / 2.0
            bw = x2 - x1

            if bw > page_width * 0.65:
                full_width.append(b)
            elif cx < mid_x:
                left_col.append(b)
            else:
                right_col.append(b)

        # 2段組みが存在する場合: 左カラム上から下 → 右カラム上から下
        if len(left_col) > 2 and len(right_col) > 2:
            left_col.sort(key=lambda b: b["box"][1])
            right_col.sort(key=lambda b: b["box"][1])
            # 全幅ブロックとマージ
            all_ordered = []
            # Y座標でインターリーブ
            col_items = left_col + right_col
            col_items.sort(key=lambda b: (0 if b["box"][0] < mid_x else 1, b["box"][1]))
            ordered = col_items
        else:
            # 1段組み通常ソート (上から下)
            classified.sort(key=lambda b: (b["box"][1], b["box"][0]))
            ordered = classified

        return ordered
