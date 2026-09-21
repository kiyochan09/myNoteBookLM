from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


class TableExtractor:
    """ページ画像とブロックから表領域および罫線セルを抽出し、行列マトリクスに変換するモジュール"""

    def __init__(self):
        pass

    def detect_table_regions(self, img_path: str | Path) -> List[Dict[str, Any]]:
        """OpenCVを用いた罫線・表領域の検出"""
        p_str = str(img_path)
        img = None
        try:
            with open(p_str, "rb") as f:
                buf = f.read()
            img = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return []

        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二値化 (白背景・黒文字/黒罫線)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        h, w = thresh.shape
        # 水平・垂直カーネル
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 40), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 40)))

        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

        table_mask = cv2.add(h_lines, v_lines)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        table_boxes = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            # 一定以上のサイズ（小さなアイコンやノイズを除外）
            if cw > w * 0.25 and ch > h * 0.05:
                table_boxes.append({
                    "box": [x, y, x + cw, y + ch],
                    "width": cw,
                    "height": ch
                })

        return table_boxes

    def extract_table_from_blocks(self, table_box: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """表矩形に含まれるテキストブロックを行列セルに再構成"""
        tx1, ty1, tx2, ty2 = table_box["box"]
        inner_blocks = []

        for b in blocks:
            bx = b.get("box", [])
            if len(bx) != 4:
                continue
            cx = (bx[0] + bx[2]) / 2
            cy = (bx[1] + bx[3]) / 2
            if tx1 <= cx <= tx2 and ty1 <= cy <= ty2:
                inner_blocks.append(b)

        if not inner_blocks:
            return None

        # Y座標でソートして行グループ化
        inner_blocks.sort(key=lambda b: (b["box"][1], b["box"][0]))
        row_groups = []
        current_row = []
        y_threshold = 15

        for b in inner_blocks:
            if not current_row:
                current_row.append(b)
                continue
            prev_y = current_row[-1]["box"][1]
            if abs(b["box"][1] - prev_y) <= y_threshold:
                current_row.append(b)
            else:
                row_groups.append(current_row)
                current_row = [b]
        if current_row:
            row_groups.append(current_row)

        if len(row_groups) < 2:
            return None

        # 各行をX座標順にソート
        matrix_rows = []
        for rg in row_groups:
            rg.sort(key=lambda b: b["box"][0])
            matrix_rows.append([b.get("text", "") for b in rg])

        max_cols = max(len(r) for r in matrix_rows)
        headers = matrix_rows[0]
        while len(headers) < max_cols:
            headers.append(f"列{len(headers) + 1}")

        body_matrix = []
        for r in matrix_rows[1:]:
            padded = r + [""] * (max_cols - len(r))
            body_matrix.append([{"value": val} for val in padded])

        return {
            "row_count": len(matrix_rows),
            "col_count": max_cols,
            "headers": headers,
            "matrix": body_matrix,
            "box": table_box["box"]
        }
