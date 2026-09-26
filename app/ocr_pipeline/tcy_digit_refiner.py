import os
import sys
import re
import math
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

class TcyDigitRefiner:
    """
    縦中横（TCY: Tate-Chū-Yoko）2桁数字（10〜99）および1桁数字（0〜9）の精密照合・補正エンジン。
    OCR（NDLOCR/WinOCR）が空白・脱落・誤認識（「切」「!」「|」「1」等）として出力する前に、
    登録済み画像テンプレート（0〜99）およびフォントレンダリングパターンと高精度テンプレートマッチング（NCC）
    を行い、文脈（「〜代」「〜歳」「〜年」「〜月」「〜日」「〜世紀」「〜章」「〜人」等）および行内相対座標に基づき自動補正する。
    """
    _instance = None

    # 方針①・方針②の幾何・インク判定パラメータ閾値
    EXCESS_INK_RATIO_THRESH = 0.350
    EXCESS_INK_DEADZONE = 0.150
    SINGLE_DIGIT_MIN_SCORE = 0.885
    HORIZONTAL_GAP_RATIO = 0.40

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.single_templates = {}
        self.tcy_2digit_templates = {}
        self._load_font_templates()
        self._load_disk_templates()

    def _load_font_templates(self):
        font_dir = Path("C:/Windows/Fonts")
        font_files = [
            "msmincho.ttc", "msgothic.ttc", "meiryo.ttc",
            "arial.ttf", "yumin.ttf", "times.ttf"
        ]
        for fname in font_files:
            fpath = font_dir / fname
            if not fpath.exists():
                continue
            for sz in [24, 28, 32, 36, 40]:
                try:
                    font = ImageFont.truetype(str(fpath), sz)
                    for d in "0123456789":
                        im = Image.new("L", (50, 50), 255)
                        draw = ImageDraw.Draw(im)
                        bbox = draw.textbbox((0, 0), d, font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        if w <= 0 or h <= 0:
                            continue
                        draw.text(((50 - w) // 2, (50 - h) // 2), d, font=font, fill=0)
                        np_im = np.array(im)
                        _, bin_t = cv2.threshold(np_im, 200, 255, cv2.THRESH_BINARY_INV)
                        pts = cv2.findNonZero(bin_t)
                        if pts is not None:
                            tx, ty, tw, th = cv2.boundingRect(pts)
                            self.single_templates.setdefault(d, []).append(np_im[ty:ty+th, tx:tx+tw])

                    for n in range(0, 100):
                        s = f"{n:02d}" if n < 10 else str(n)
                        im = Image.new("L", (60, 60), 255)
                        draw = ImageDraw.Draw(im)
                        bbox = draw.textbbox((0, 0), s, font=font)
                        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                        if w <= 0 or h <= 0:
                            continue
                        draw.text(((60 - w) // 2, (60 - h) // 2), s, font=font, fill=0)
                        np_im = np.array(im)
                        _, bin_t = cv2.threshold(np_im, 200, 255, cv2.THRESH_BINARY_INV)
                        pts = cv2.findNonZero(bin_t)
                        if pts is not None:
                            tx, ty, tw, th = cv2.boundingRect(pts)
                            self.tcy_2digit_templates.setdefault(n, []).append(np_im[ty:ty+th, tx:tx+tw])
                except Exception:
                    pass

    def _load_disk_templates(self):
        candidate_dirs = [
            Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\config\tcy_templates"),
            Path(r"C:\Users\natur\.gemini\antigravity\scratch\knowledge_base_system\data\tcy_templates"),
        ]
        for tdir in candidate_dirs:
            if not tdir.exists():
                continue
            for p in tdir.glob("*.png"):
                raw_num = p.stem.split("_")[0]
                try:
                    num = int(raw_num)
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        _, bin_t = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
                        pts = cv2.findNonZero(bin_t)
                        if pts is not None:
                            tx, ty, tw, th = cv2.boundingRect(pts)
                            crop = img[ty:ty+th, tx:tx+tw]
                            if len(raw_num) == 1:
                                self.single_templates.setdefault(str(num), []).append(crop)
                            else:
                                self.tcy_2digit_templates.setdefault(num, []).append(crop)
                except Exception:
                    pass

    def match_1digit_patch_detailed(self, patch_gray: np.ndarray, ch: int = None) -> Tuple[Optional[str], float, Optional[Tuple[int, int]], Optional[np.ndarray]]:
        """
        単一パッチを登録済み1桁数字（0〜9）テンプレートと高精度照合（NCC）し、
        ベストマッチ数字、スコア、最適位置、および最適スケーリングテンプレートを返却する。
        """
        if patch_gray is None or patch_gray.size == 0:
            return None, 0.0, None, None
        ph, pw = patch_gray.shape
        if ph < 8 or pw < 4:
            return None, 0.0, None, None

        if np.mean(patch_gray) < 100:
            patch_gray = 255 - patch_gray

        best_digit = None
        best_s = -1.0
        best_pos = None
        best_scaled_t = None

        base_h = ch if ch is not None else ph

        for d, t_list in self.single_templates.items():
            for t in t_list:
                th, tw = t.shape
                for scale_h in [0.80, 0.90, 1.00, 1.10, 1.20]:
                    target_h = int(base_h * scale_h)
                    if target_h < 8 or target_h >= ph:
                        continue
                    target_w = max(4, int(tw * (target_h / th)))
                    if target_w >= pw:
                        target_w = pw - 1
                    if target_w < 4:
                        continue
                    scaled_t = cv2.resize(t, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(patch_gray, scaled_t, cv2.TM_CCOEFF_NORMED)
                    s = float(res.max())
                    if s > best_s:
                        best_s = s
                        best_digit = d
                        best_pos = np.unravel_index(res.argmax(), res.shape)
                        best_scaled_t = scaled_t

        return best_digit, best_s, best_pos, best_scaled_t

    def match_1digit_patch(self, patch_gray: np.ndarray, ch: int = None) -> Tuple[Optional[str], float]:
        """
        単一パッチを登録済み1桁数字（0〜9）テンプレートと高精度照合（NCC）する。
        """
        best_digit, best_s, _, _ = self.match_1digit_patch_detailed(patch_gray, ch)
        return best_digit, best_s

    def calculate_excess_ink(self, patch_gray: np.ndarray, best_pos: Optional[Tuple[int, int]], best_scaled_t: Optional[np.ndarray], kernel_size=(5, 5)) -> Tuple[float, int, int]:
        """
        テンプレート領域外の余剰インク比率（R_excess = A_excess / A_tmpl）を算出する。
        - テンプレートマスクに対して kernel_size の矩形カーネルで1回膨張（スキャンブレ・セリフの許容マージン）
        - 膨張マスク外に存在するインク画素数を A_excess、テンプレートインク画素数を A_tmpl とする
        """
        if best_pos is None or best_scaled_t is None:
            return 0.0, 0, 0

        by, bx = best_pos
        th, tw = best_scaled_t.shape
        ph, pw = patch_gray.shape

        patch_ink = (patch_gray < 200).astype(np.uint8)
        tmpl_ink = (best_scaled_t < 200).astype(np.uint8)

        a_tmpl = int(np.sum(tmpl_ink))
        if a_tmpl == 0:
            return 0.0, 0, 0

        tmpl_mask_in_patch = np.zeros((ph, pw), dtype=np.uint8)
        y2 = min(ph, by + th)
        x2 = min(pw, bx + tw)
        tmpl_sub_h = y2 - by
        tmpl_sub_w = x2 - bx

        tmpl_mask_in_patch[by:y2, bx:x2] = tmpl_ink[:tmpl_sub_h, :tmpl_sub_w]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        dilated_tmpl_mask = cv2.dilate(tmpl_mask_in_patch, kernel, iterations=1)

        excess_mask = (patch_ink == 1) & (dilated_tmpl_mask == 0)
        a_excess = int(np.sum(excess_mask))
        r_excess = float(a_excess) / float(a_tmpl)
        return r_excess, a_excess, a_tmpl

    def match_2digit_patch(self, patch_gray: np.ndarray) -> Tuple[Optional[int], float]:
        """
        単一パッチを登録済み2桁数字（00〜99）テンプレートと高精度照合（NCC）する。
        """
        if patch_gray is None or patch_gray.size == 0:
            return None, 0.0
        ph, pw = patch_gray.shape
        if ph < 10 or pw < 10:
            return None, 0.0

        if np.mean(patch_gray) < 100:
            patch_gray = 255 - patch_gray

        best_num = None
        best_s = -1.0
        for num, t_list in self.tcy_2digit_templates.items():
            for t in t_list:
                th, tw = t.shape
                for scale_h in [0.75, 0.85, 0.95, 1.05]:
                    target_h = int(ph * scale_h)
                    if target_h < 8 or target_h >= ph:
                        continue
                    target_w = max(6, int(tw * (target_h / th)))
                    if target_w >= pw:
                        target_w = pw - 1
                    if target_w < 6:
                        continue
                    scaled_t = cv2.resize(t, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(patch_gray, scaled_t, cv2.TM_CCOEFF_NORMED)
                    s = float(res.max())
                    if s > best_s:
                        best_s = s
                        best_num = num
        return best_num, best_s

    def find_tcy_in_vertical_line(self, line_img: np.ndarray) -> List[Dict[str, Any]]:
        if line_img is None or line_img.size == 0:
            return []

        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()

        lh, lw = gray.shape
        if lh < 30 or lw < 15:
            return []

        # 方針④: 大津の二値化 (Otsu) に統一して安定した輪郭抽出
        otsu_thresh, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

        components = []
        for i in range(1, num_labels):
            bx, by, bw, bh, area = stats[i]
            if area > 10 and bh > 6:
                components.append({"id": i, "x1": bx, "x2": bx + bw, "y1": by, "y2": by + bh, "w": bw, "h": bh, "area": area})

        detected = []
        used_comp_ids = set()

        # Step 1: 2桁数字ペアの検出（横並びコンポーネント照合）
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                c1 = components[i]
                c2 = components[j]

                y_overlap = max(0, min(c1["y2"], c2["y2"]) - max(c1["y1"], c2["y1"]))
                min_h = min(c1["h"], c2["h"])
                max_h = max(c1["h"], c2["h"])

                if min_h > 0 and (y_overlap / min_h) > 0.45 and (min_h / max_h) > 0.40:
                    left_c = c1 if c1["x1"] < c2["x1"] else c2
                    right_c = c2 if c1["x1"] < c2["x1"] else c1

                    h_gap = right_c["x1"] - left_c["x2"]
                    if -8 <= h_gap <= 24:
                        # 方針②拡張: ペアのいずれかが同一高さ帯の別コンポーネントと近接している場合（3つ以上のパーツからなる複合漢字）は除外
                        c_pair_ids = {c1["id"], c2["id"]}
                        has_third_part = False
                        for other in components:
                            if other["id"] in c_pair_ids:
                                continue
                            y_ov = min(max(c1["y2"], c2["y2"]), other["y2"]) - max(min(c1["y1"], c2["y1"]), other["y1"])
                            if y_ov > 0:
                                g1 = c1["x1"] - other["x2"] if c1["x1"] >= other["x2"] else other["x1"] - c1["x2"]
                                g2 = c2["x1"] - other["x2"] if c2["x1"] >= other["x2"] else other["x1"] - c2["x2"]
                                if min(g1, g2) < lw * self.HORIZONTAL_GAP_RATIO:
                                    has_third_part = True
                                    break
                        if has_third_part:
                            continue

                        y1 = max(0, min(c1["y1"], c2["y1"]) - 2)
                        y2 = min(lh, max(c1["y2"], c2["y2"]) + 2)
                        x1 = max(0, left_c["x1"] - 2)
                        x2 = min(lw, right_c["x2"] + 2)
                        combined_patch = gray[y1:y2, x1:x2]

                        num_patch, s_patch = self.match_2digit_patch(combined_patch)
                        if num_patch is not None and s_patch >= 0.75:
                            detected.append({
                                "num": num_patch,
                                "num_str": str(num_patch),
                                "score": s_patch,
                                "y1": y1,
                                "y2": y2,
                                "rel_y": (y1 + y2) / (2.0 * lh),
                                "method": "2digit_patch_match"
                            })
                            used_comp_ids.add(c1["id"])
                            used_comp_ids.add(c2["id"])
                        else:
                            # 左右個別の1桁照合 + 幾何平均信頼度（細身フォント・分断数字の横並びグルーピング）
                            patch_left = gray[max(0, left_c["y1"]-2):min(lh, left_c["y2"]+2), max(0, left_c["x1"]-2):min(lw, left_c["x2"]+2)]
                            patch_right = gray[max(0, right_c["y1"]-2):min(lh, right_c["y2"]+2), max(0, right_c["x1"]-2):min(lw, right_c["x2"]+2)]
                            d_left, s_left = self.match_1digit_patch(patch_left)
                            d_right, s_right = self.match_1digit_patch(patch_right)
                            if d_left is not None and d_right is not None and s_left >= 0.70 and s_right >= 0.70:
                                base_s = math.sqrt(s_left * s_right)
                                comb_n = int(d_left) * 10 + int(d_right)
                                # もし2桁丸ごとパッチ照合で候補があり有意（>=0.74）であれば個別照合の誤爆（12等）より優先
                                if num_patch is not None and s_patch >= 0.74:
                                    final_n = num_patch
                                    final_s = s_patch
                                else:
                                    final_n = comb_n
                                    final_s = base_s

                                # 漢字ストローク誤爆防止（11等は全体2桁テンプレート照合で0.88以上のみ許容）
                                if final_n == 11 and (num_patch != 11 or s_patch < 0.88):
                                    continue

                                if final_s >= 0.75:
                                    detected.append({
                                        "num": final_n,
                                        "num_str": str(final_n),
                                        "score": final_s,
                                        "y1": y1,
                                        "y2": y2,
                                        "rel_y": (y1 + y2) / (2.0 * lh),
                                        "method": "2digit_patch_match"
                                    })
                                    used_comp_ids.add(c1["id"])
                                    used_comp_ids.add(c2["id"])

        # Step 1.5: 単一コンポーネントの2桁数字照合（接触・連結した2桁数字）
        for c in components:
            if c["id"] in used_comp_ids:
                continue
            # 横幅が十分にあり、かつ縦横比が2桁数字に適している場合（接触連結している2桁数字）
            if 8 <= c["h"] <= 35 and 13 <= c["w"] and (c["w"] / float(max(1, c["h"]))) >= 0.60:
                y1 = max(0, c["y1"] - 2)
                y2 = min(lh, c["y2"] + 2)
                x1 = max(0, c["x1"] - 2)
                x2 = min(lw, c["x2"] + 2)
                patch = gray[y1:y2, x1:x2]
                num_patch, s_patch = self.match_2digit_patch(patch)
                if num_patch is not None and s_patch >= 0.78:
                    detected.append({
                        "num": num_patch,
                        "num_str": str(num_patch),
                        "score": s_patch,
                        "y1": y1,
                        "y2": y2,
                        "rel_y": (y1 + y2) / (2.0 * lh),
                        "method": "2digit_patch_match"
                    })
                    used_comp_ids.add(c["id"])

        # Step 2: 1桁数字の検出（未使用コンポーネントの単体照合）
        for c in components:
            if c["id"] in used_comp_ids:
                continue

            # 方針②: 水平近傍Gap判定（門構え等の複合漢字パーツの除外）
            min_neighbor_gap = None
            for other in components:
                if other["id"] == c["id"]:
                    continue
                y_overlap = min(c["y2"], other["y2"]) - max(c["y1"], other["y1"])
                if y_overlap > 0:
                    gap = c["x1"] - other["x2"] if c["x1"] >= other["x2"] else other["x1"] - c["x2"]
                    if min_neighbor_gap is None or gap < min_neighbor_gap:
                        min_neighbor_gap = gap

            if min_neighbor_gap is not None and min_neighbor_gap < lw * self.HORIZONTAL_GAP_RATIO:
                continue

            # 方針①: 幾何サイズ適合 & 余剰インク比率による減点・足切り
            if 8 <= c["h"] <= 65 and 4 <= c["w"] <= lw * 0.85:
                y1 = max(0, c["y1"] - 2)
                y2 = min(lh, c["y2"] + 2)
                x1 = max(0, c["x1"] - 2)
                x2 = min(lw, c["x2"] + 2)
                patch = gray[y1:y2, x1:x2]

                d, s, pos, scaled_t = self.match_1digit_patch_detailed(patch, ch=c["h"])
                if d is not None and s >= 0.70:
                    r_excess, a_excess, a_tmpl = self.calculate_excess_ink(patch, pos, scaled_t, kernel_size=(5, 5))

                    # 余剰インク足切り閾値 (0.350)
                    if r_excess > self.EXCESS_INK_RATIO_THRESH:
                        continue

                    # スキャンにじみ・セリフの不感帯(0.150)を考慮したペナルティ減点
                    excess_over_deadzone = max(0.0, r_excess - self.EXCESS_INK_DEADZONE)
                    penalty_factor = max(0.0, 1.0 - 2.0 * excess_over_deadzone)
                    final_score = s * penalty_factor

                    # 単体1桁スコア基準閾値 (0.885)
                    if final_score >= self.SINGLE_DIGIT_MIN_SCORE:
                        detected.append({
                            "num": int(d),
                            "num_str": str(d),
                            "score": final_score,
                            "raw_score": s,
                            "r_excess": r_excess,
                            "y1": y1,
                            "y2": y2,
                            "w": c["w"],
                            "h": c["h"],
                            "comp_id": c["id"],
                            "rel_y": (y1 + y2) / (2.0 * lh),
                            "method": "1digit_patch_match"
                        })
        # Step 2.5: 縦並び1桁数字ペアの幾何学的グルーピング（Layer 1: Y近接・X同軸性・幾何平均信頼度）
        one_digit_items = [it for it in detected if it.get("method") == "1digit_patch_match"]
        grouped_2digit = []
        used_1digit_ids = set()
        for i in range(len(one_digit_items)):
            if i in used_1digit_ids:
                continue
            for j in range(i + 1, len(one_digit_items)):
                if j in used_1digit_ids:
                    continue
                it1, it2 = one_digit_items[i], one_digit_items[j]
                top_it = it1 if it1["y1"] <= it2["y1"] else it2
                bot_it = it2 if it1["y1"] <= it2["y1"] else it1

                h1 = float(top_it["y2"] - top_it["y1"])
                h2 = float(bot_it["y2"] - bot_it["y1"])
                avg_h = (h1 + h2) / 2.0

                cy1 = (top_it["y1"] + top_it["y2"]) / 2.0
                cy2 = (bot_it["y1"] + bot_it["y2"]) / 2.0
                y_dist = cy2 - cy1
                gap_y = bot_it["y1"] - top_it["y2"]

                # Y近接判定（中心間距離が 0.70〜1.50倍）
                if 0.70 * avg_h <= y_dist <= 1.50 * avg_h and -0.30 * avg_h <= gap_y <= 0.50 * avg_h:
                    combined_num = top_it["num"] * 10 + bot_it["num"]
                    base_score = math.sqrt(top_it["score"] * bot_it["score"])
                    affinity = math.exp(- ((abs(y_dist - avg_h)) ** 2) / (2.0 * ((0.30 * avg_h) ** 2)))
                    combined_score = base_score * (0.80 + 0.20 * affinity)

                    grouped_2digit.append({
                        "num": combined_num,
                        "num_str": str(combined_num),
                        "score": combined_score,
                        "y1": top_it["y1"],
                        "y2": bot_it["y2"],
                        "rel_y": (top_it["y1"] + bot_it["y2"]) / (2.0 * lh),
                        "method": "2digit_patch_match"
                    })
                    used_1digit_ids.add(i)
                    used_1digit_ids.add(j)
                    break

        detected.extend(grouped_2digit)
        detected.sort(key=lambda c: -c["score"])
        filtered = []
        for cand in detected:
            overlap = False
            for f in filtered:
                if max(0, min(cand["y2"], f["y2"]) - max(cand["y1"], f["y1"])) > 8:
                    overlap = True
                    break
            if not overlap:
                filtered.append(cand)

        filtered.sort(key=lambda c: c["y1"])
        return filtered

    def refine_kanji_years_and_dates(self, line_img: np.ndarray, text: str) -> str:
        if not text or line_img is None or line_img.size == 0:
            return text
        
        if len(line_img.shape) == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img

        lh, lw = gray.shape
        if lh < 30 or lw < 10:
            return text

        # 1. 行頭の4桁西暦年号欠損・乱れの補助照合 (WinOCRアンサンブル)
        try:
            try:
                from app.ocr_pipeline.win_ocr import run_ocr_on_image
            except ImportError:
                try:
                    from ocr_engine.win_ocr import run_ocr_on_image
                except ImportError:
                    from win_ocr import run_ocr_on_image

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
                tmp_name = tf.name
            
            try:
                top_part = gray[0:min(lh, 450), :]
                padded = cv2.copyMakeBorder(top_part, 30, 30, 50, 50, cv2.BORDER_CONSTANT, value=255)
                cv2.imwrite(tmp_name, padded)
                wres = run_ocr_on_image(tmp_name, lang='ja')
                if wres and wres[0].get('text'):
                    wt = wres[0]['text'].replace(' ', '')
                    # 4桁年号における文字混同（「2。14年」「2〇14年」「2O14年」→「2014年」など）を正規化
                    wt = re.sub(r'([12])[。〇OoD]([0-9]{2})', r'\g<1>0\g<2>', wt)
                    wt = re.sub(r'([12][0-9])[。〇OoD]([0-9])', r'\g<1>0\g<2>', wt)
                    wt = re.sub(r'([12][0-9]{2})[。〇OoD]', r'\g<1>0', wt)
                    wt = re.sub(r'[!|lI]([89][0-9]{2})', r'1\g<1>', wt)

                    # 4桁年号 (例: 2014年2月27日, 2014年5月, 1989年)
                    m_4digit_year = re.search(r'([12][0-9]{3})\s*年', wt)
                    if m_4digit_year:
                        yr = m_4digit_year.group(1)
                        m_full_date = re.search(r'([12][0-9]{3})\s*年\s*([0-9]{1,2})\s*月\s*([0-9]{1,2})\s*日?', wt)
                        if m_full_date and re.match(r'^\s*[0-9一二三四五六七八九十]*年\s*[0-9一二三四五六七八九十]*月', text):
                            mo, dy = m_full_date.group(2), m_full_date.group(3)
                            suffix_day = f"{dy}日" if dy else ""
                            text = re.sub(r'^\s*[0-9一二三四五六七八九十]*年\s*[0-9一二三四五六七八九十]*月\s*[0-9一二三四五六七八九十]*日?', f"{yr}年{mo}月{suffix_day}", text)
                        else:
                            m_yr_mo = re.search(r'([12][0-9]{3})\s*年\s*([0-9]{1,2})\s*月', wt)
                            if m_yr_mo and re.match(r'^\s*[0-9一二三四五六七八九十]*年\s*[0-9一二三四五六七八九十]*月', text):
                                mo = m_yr_mo.group(2)
                                text = re.sub(r'^\s*[0-9一二三四五六七八九十]*年\s*[0-9一二三四五六七八九十]*月', f"{yr}年{mo}月", text)
                            elif re.match(r'^\s*[0-9一二三四五六七八九十]*年', text):
                                text = re.sub(r'^\s*[0-9一二三四五六七八九十]*年', f"{yr}年", text)
            finally:
                if os.path.exists(tmp_name):
                    try:
                        os.remove(tmp_name)
                    except Exception:
                        pass
        except Exception:
            pass

        # 3. "二〇年" or "二〇〇年" -> count '〇' circles in image
        if "二〇年" in text or "二〇〇年" in text or "一九年" in text:
            _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)
            circles = 0
            for i in range(1, num_labels):
                bx, by, bw, bh, area = stats[i]
                if area > 80 and bh >= 18 and bw >= 18:
                    aspect = bh / float(bw)
                    if 0.7 <= aspect <= 1.4:
                        circles += 1
            if circles >= 3:
                text = text.replace("二〇年", "二〇〇〇年").replace("二〇〇年", "二〇〇〇年")

        # 4. 鍵括弧構文破綻の WinOCR アンサンブル救済
        has_bracket_anomaly = bool(
            re.search(r'「[^」]*[、「][^」]*」', text) or
            (text.count('「') != text.count('」'))
        )
        if has_bracket_anomaly:
            try:
                try:
                    from app.ocr_pipeline.win_ocr import run_ocr_on_image
                except ImportError:
                    try:
                        from ocr_engine.win_ocr import run_ocr_on_image
                    except ImportError:
                        from win_ocr import run_ocr_on_image

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
                    tmp_name = tf.name
                try:
                    padded = cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
                    cv2.imwrite(tmp_name, padded)
                    wres = run_ocr_on_image(tmp_name, lang='ja')
                    if wres and wres[0].get('text'):
                        wt = wres[0]['text'].replace(' ', '')
                        m_win = re.search(r'「([^」]+)」', wt)
                        m_ndl = re.search(r'「([^」]+)」', text)
                        if m_win and m_ndl:
                            win_content = m_win.group(1)
                            ndl_content = m_ndl.group(1)
                            if any(c in ndl_content for c in ['、', '「']) and not any(c in win_content for c in ['、', '「']):
                                text = text[:m_ndl.start()] + f"「{win_content}」" + text[m_ndl.end():]
                finally:
                    if os.path.exists(tmp_name):
                        try:
                            os.remove(tmp_name)
                        except Exception:
                            pass
            except Exception:
                pass

        return text

    def _segment_line_adaptive(self, line_img: np.ndarray) -> List[Tuple[int, int]]:
        """
        【適応型インクプロファイル分割】
        行画像全体を大津の二値化により適応的に二値化し、
        行幅 lw に連動した動的ギャップで行内のインク塊（文字ブロック）区間を抽出する。
        """
        if line_img is None or line_img.size == 0:
            return []

        if line_img.ndim == 3:
            gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = line_img.copy()

        lh, lw = gray.shape
        if lh < 20 or lw < 10:
            return []

        # 大津の二値化（Otsu）によりスキャンムラ・紙の黄ばみを自動吸収
        if np.mean(gray) < 100:
            gray = 255 - gray
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 行幅 lw 連動の動的ギャップ（文字内部ストローク空隙での誤分裂を防止）
        dynamic_min_gap = max(3, int(lw * 0.15))
        noise_ink_thresh = max(1, int(lw * 0.05))

        # 水平方向インク射影ヒストグラム
        proj = np.sum(binary > 0, axis=1)
        is_ink = proj > noise_ink_thresh

        # 連続インク区間の抽出
        raw_blocks: List[Tuple[int, int]] = []
        in_block = False
        start_y = 0
        for y in range(len(is_ink)):
            if is_ink[y] and not in_block:
                start_y = y
                in_block = True
            elif not is_ink[y] and in_block:
                raw_blocks.append((start_y, y))
                in_block = False
        if in_block:
            raw_blocks.append((start_y, len(is_ink)))

        if not raw_blocks:
            return []

        # 動的ギャップによる近接ブロック結合
        merged_blocks: List[Tuple[int, int]] = []
        for b in raw_blocks:
            if merged_blocks and (b[0] - merged_blocks[-1][1]) < dynamic_min_gap:
                merged_blocks[-1] = (merged_blocks[-1][0], b[1])
            else:
                merged_blocks.append(b)

        # 極小ノイズブロックを除外
        valid_blocks = [
            b for b in merged_blocks
            if (b[1] - b[0]) >= max(4, int(lw * 0.10))
        ]
        return valid_blocks

    def _get_tcy_rank_in_blocks(self, tcy_item: Dict[str, Any], blocks: List[Tuple[int, int]]) -> int:
        """
        検出されたTCY候補の中心Y座標より上方にある文字ブロック数（手前の文字数 k）を返す。
        """
        if not blocks:
            return 0
        tcy_cy = (tcy_item["y1"] + tcy_item["y2"]) / 2.0
        rank = 0
        for b_start, b_end in blocks:
            b_cy = (b_start + b_end) / 2.0
            if b_cy < tcy_cy:
                rank += 1
            else:
                break
        return rank

    def _locate_and_embed_locked_digits(
        self,
        text: str,
        locked_items: List[Dict[str, Any]],
        line_blocks: List[Tuple[int, int]]
    ) -> str:
        """
        【画像主導・即時確定型 配置エンジン】
        画像照合で確定した数字アイテムを、画像上の物理位置（ブロック順位 k）を
        絶対的な根拠としてテキスト内に配置する。
        ※文脈判定や文字種による足切り・判定は完全撤廃。ノイズ文字や誤読数字の置換、
          または基準物理位置への直接割り込み挿入により100%出力する。
        """
        if not locked_items or not text:
            return text

        text_len = len(text)
        has_blocks = bool(line_blocks)

        # 配置プランのリスト: (開始インデックス, 終了インデックス, 置換・挿入文字列)
        placements = []

        # 上から下への物理配置を保つため、Y座標順にソート
        sorted_locked = sorted(locked_items, key=lambda it: it["y1"])

        for item in sorted_locked:
            num = item["num"]
            num_str = str(num)
            rel_y = item["rel_y"]

            # -------------------------------------------------------------
            # ステップ1: 手前ブロック数 k を直接ベース位置とする
            # （グローバル比率換算を撤廃し、遠隔の文字数誤差を遮断）
            # -------------------------------------------------------------
            if has_blocks:
                k = self._get_tcy_rank_in_blocks(item, line_blocks)
                base_idx = max(0, min(text_len, k))
            else:
                base_idx = int(round(rel_y * text_len))
                base_idx = max(0, min(text_len, base_idx))

            # -------------------------------------------------------------
            # ステップ2: 近傍スナップ（吸着）探索
            # base_idx の周辺 (±2文字) に、自然な合致先がないかを探索
            # -------------------------------------------------------------
            window_start = max(0, base_idx - 2)
            window_end = min(text_len, base_idx + 3)
            window_str = text[window_start:window_end]

            snapped = False

            # スナップ優先度①: 典型的な誤読・ノイズ記号（切, !, ?, |, 空白）の置換
            for noise_token in ['切', '!', '?', '|', ' ', '　']:
                local_pos = window_str.find(noise_token)
                if local_pos != -1:
                    target_pos = window_start + local_pos
                    placements.append((target_pos, target_pos + 1, num_str))
                    snapped = True
                    break

            # スナップ優先度②: 丸括弧スロット （ ） や ( ) への充填
            if not snapped:
                paren_match = re.search(r'([（\(])\s*([0-9０-９!|lI\s]{0,3})\s*([）\)])', window_str)
                if paren_match:
                    p_start = window_start + paren_match.start(2)
                    p_end = window_start + paren_match.end(2)
                    placements.append((p_start, p_end, num_str))
                    snapped = True

            # スナップ優先度③: 基準位置周辺にOCR誤読された1〜2桁数字がある場合はその位置を置換
            if not snapped:
                digit_match = re.search(r'[0-9０-９]{1,2}', window_str)
                if digit_match:
                    d_start = window_start + digit_match.start()
                    d_end = window_start + digit_match.end()
                    placements.append((d_start, d_end, num_str))
                    snapped = True

            # スナップ優先度④: 【無条件フォールバック（直接割り込み挿入）】
            # 画像照合スコアで確定した数字を、ブロック順位 k の基準位置に挿入
            # ※長音符「ー」の保護: 基準位置周辺にすでに「ー」が存在している場合は「1」の割り込み挿入を行わない
            if not snapped:
                is_chouon_conflict = (num_str == "1" and (
                    'ー' in window_str or
                    (base_idx < text_len and text[base_idx] == 'ー') or
                    (base_idx > 0 and text[base_idx - 1] == 'ー')
                ))
                if not is_chouon_conflict:
                    placements.append((base_idx, base_idx, num_str))

        if not placements:
            return text

        # -------------------------------------------------------------
        # ステップ3: テキストへの統合
        # インデックスのズレを防ぐため、後ろ（末尾側）から順に置換・挿入
        # -------------------------------------------------------------
        placements.sort(key=lambda x: -x[0])
        result_text = text
        for s_idx, e_idx, num_str in placements:
            result_text = result_text[:s_idx] + num_str + result_text[e_idx:]

        return result_text

    def refine_line(self, line_img: np.ndarray, text: str, next_line_text: Optional[str] = None) -> str:
        """
        行画像とOCR認識テキストを受け取り、縦中横の即時確定・配置補正を行う。
        """
        if not text:
            return text

        # 1. 画像解析による縦中横の検出
        tcy_items = self.find_tcy_in_vertical_line(line_img)
        if not tcy_items:
            return self.refine_kanji_years_and_dates(line_img, text)

        # 2. ★【画像スコアに基づく即時確定（Lock-in）】
        locked_digits = []
        for it in tcy_items:
            method = it.get("method", "")
            score = it.get("score", 0.0)
            num = it.get("num", 0)
            is_2d = (method == "2digit_patch_match")

            if is_2d and score >= 0.78:
                locked_digits.append(it)
            elif not is_2d and num in [1, 7] and score >= self.SINGLE_DIGIT_MIN_SCORE:
                locked_digits.append(it)
            elif not is_2d and num not in [1, 7] and score >= 0.85:
                locked_digits.append(it)

        if not locked_digits:
            return self.refine_kanji_years_and_dates(line_img, text)

        # 3. 行のインクプロファイル（文字ブロック）を適応型二値化で抽出
        line_blocks = self._segment_line_adaptive(line_img)

        # 4. ★【確定した数字を、物理位置（ブロック順位 k）を主軸にしてテキストへ確実に配置】
        refined = self._locate_and_embed_locked_digits(text, locked_digits, line_blocks)

        # 5. 年号範囲（1909~59年等）や日付ペアなどの最終テキスト正規化
        refined = self.refine_kanji_years_and_dates(line_img, refined)

        return refined

def refine_vertical_ocr_text(text: str, is_vertical: bool, line_img: np.ndarray, next_line_text: Optional[str] = None) -> str:
    if not is_vertical or line_img is None:
        return text
    refiner = TcyDigitRefiner.get_instance()
    return refiner.refine_line(line_img, text, next_line_text=next_line_text)

def refine_text_with_tcy(line_img: np.ndarray, text: str, next_line_text: Optional[str] = None) -> str:
    refiner = TcyDigitRefiner.get_instance()
    return refiner.refine_line(line_img, text, next_line_text=next_line_text)

