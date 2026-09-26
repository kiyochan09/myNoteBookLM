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

    def match_1digit_patch(self, patch_gray: np.ndarray) -> Tuple[Optional[str], float]:
        """
        単一パッチを登録済み1桁数字（0〜9）テンプレートと高精度照合（NCC）する。
        """
        if patch_gray is None or patch_gray.size == 0:
            return None, 0.0
        ph, pw = patch_gray.shape
        if ph < 8 or pw < 4:
            return None, 0.0

        if np.mean(patch_gray) < 100:
            patch_gray = 255 - patch_gray

        best_digit = None
        best_s = -1.0
        for d, t_list in self.single_templates.items():
            for t in t_list:
                th, tw = t.shape
                for scale_h in [0.70, 0.85, 0.95, 1.05, 1.15]:
                    target_h = int(ph * scale_h)
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
        return best_digit, best_s

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

        _, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
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

                                # 漢字ストローク誤爆防止（11等は厳格化）
                                if final_n != 11 or final_s >= 0.90:
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
            if 8 <= c["h"] <= 65 and 4 <= c["w"] <= lw * 0.85:
                y1 = max(0, c["y1"] - 2)
                y2 = min(lh, c["y2"] + 2)
                x1 = max(0, c["x1"] - 2)
                x2 = min(lw, c["x2"] + 2)
                patch = gray[y1:y2, x1:x2]
                d, s = self.match_1digit_patch(patch)
                if d is not None and s >= 0.78:
                    detected.append({
                        "num": int(d),
                        "num_str": str(d),
                        "score": s,
                        "y1": y1,
                        "y2": y2,
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

    def refine_line(self, line_img: np.ndarray, text: str, next_line_text: Optional[str] = None) -> str:
        if not text:
            return text

        refined = self.refine_kanji_years_and_dates(line_img, text)
        tcy_items = self.find_tcy_in_vertical_line(line_img)
        if not tcy_items:
            return refined

        # 優先度ソート:
        # 1. 2桁縦中横（2digit_patch_match）を最優先（漢字の直線ストローク誤爆に競り負けない）
        # 2. 1桁数字（非1/7）
        # 3. 1桁数字（1または7）
        def get_priority(it):
            method = it.get("method", "")
            num = it.get("num", 0)
            if method == "2digit_patch_match":
                return (0, -it["score"])
            elif num not in [1, 7]:
                return (1, -it["score"])
            else:
                return (2, -it["score"])

        # 漢字・仮名ストローク等の誤爆フィルタリング（特に縦棒ストローク「1」と払い「7」の厳格判定）
        clean_items = []
        for it in tcy_items:
            method = it.get("method", "")
            num = it.get("num", 0)
            score = it.get("score", 0.0)
            if method == "2digit_patch_match":
                clean_items.append(it)
            elif num == 1:
                if score >= 0.96:
                    clean_items.append(it)
            elif num == 7:
                if score >= 0.90:
                    clean_items.append(it)
            elif score >= 0.78:
                clean_items.append(it)

        # 2桁縦中横のバウンディングボックスと重なる1桁誤爆を除外
        final_items = []
        for it in clean_items:
            if it.get("method") == "1digit_patch_match":
                inside = any(abs(it["rel_y"] - d2["rel_y"]) < 0.025 for d2 in clean_items if d2.get("method") == "2digit_patch_match")
                if inside:
                    continue
            final_items.append(it)

        sorted_items = sorted(final_items, key=get_priority)
        used_item_indices = set()

        # Step 1: 西暦年号範囲の幾何照合（例: 1909~59年, 1914~18年）
        # NDLOCRが「1909~5年」と1文字落としたり、ノイズ記号になったりするのを高精度2桁アイテムで救済
        m_yr_range = re.search(r'([12][0-9]{3}\s*[~〜ー\-]\s*)([0-9]{1,2})\s*(年)', refined)
        if m_yr_range:
            cur_len = max(1, len(refined))
            s_idx, e_idx = m_yr_range.start(), m_yr_range.end()
            yr_center = (s_idx + e_idx) / (2.0 * cur_len)
            best_yr_idx = None
            best_yr_dist = 999.0
            for idx, it in enumerate(sorted_items):
                if idx in used_item_indices:
                    continue
                if it.get("method") == "2digit_patch_match" and it["score"] >= 0.85:
                    d = abs(it["rel_y"] - yr_center)
                    if d < best_yr_dist and d < 0.18:
                        best_yr_dist = d
                        best_yr_idx = idx
            if best_yr_idx is not None:
                it = sorted_items[best_yr_idx]
                used_item_indices.add(best_yr_idx)
                prefix_yr = m_yr_range.group(1)
                refined = refined[:s_idx] + f"{prefix_yr}{it['num']}年" + refined[e_idx:]

        # Step 3: 日付ペア「○月○日」の幾何学的照合（原本画像の上から下への物理的配置に基づき、月と日の順序逆転を完全防止）
        date_match = re.search(rf'([0-9０-９!一二三四五六七八九十I|l冗らろへー・\s]{{0,4}})月\s*([0-9０-９!一二三四五六七八九十I|l冗らろへー・\s]{{0,4}})日', refined)
        if date_match:
            cur_len = max(1, len(refined))
            date_center = (date_match.start() + date_match.end()) / (2.0 * cur_len)
            best_date_pair = None
            best_cost = 999.0
            for i in range(len(sorted_items)):
                if i in used_item_indices: continue
                for j in range(i + 1, len(sorted_items)):
                    if j in used_item_indices: continue
                    itA, itB = sorted_items[i], sorted_items[j]
                    if itA["rel_y"] > itB["rel_y"]:
                        itA, itB = itB, itA
                    y_diff = itB["rel_y"] - itA["rel_y"]
                    # 月と日の物理的離隔距離（0.030 <= y_diff <= 0.12）
                    if 0.030 <= y_diff <= 0.12:
                        pair_center = (itA["rel_y"] + itB["rel_y"]) / 2.0
                        cost = abs(pair_center - date_center)
                        if 1 <= itA["num"] <= 12 and 1 <= itB["num"] <= 31:
                            cost *= 0.5
                        if cost < best_cost and cost < 0.22:
                            best_cost = cost
                            best_date_pair = (i, j, itA, itB)
            if best_date_pair:
                i, j, itA, itB = best_date_pair
                used_item_indices.add(i)
                used_item_indices.add(j)
                refined = refined[:date_match.start()] + f"{itA['num']}月{itB['num']}日" + refined[date_match.end():]

        # 4桁年号および年号範囲（例: 2014年、1909~59年、(1909~59年)）の出現範囲を取得
        text_len = max(1, len(refined))
        protected_year_ranges = []
        for my in re.finditer(r'([（\(]?\s*[12][0-9]{3})(?:\s*[~〜ー\-]\s*[0-9]{1,4})?\s*年\s*[）\)]?', refined):
            y_start_rel = my.start() / float(text_len)
            y_end_rel = my.end() / float(text_len)
            protected_year_ranges.append((y_start_rel, y_end_rel, my.start(), my.end()))

        # パターン1: 見出し「第X章」→「第{num}章」
        for item in sorted_items:
            num = item["num"]
            rel_y = item["rel_y"]
            if "第" in refined and any(s in refined for s in ["章", "条", "回", "巻"]) and rel_y < 0.40:
                m_sec = re.search(r'第\s*([0-9０-９Ⅰ-Ⅻ!一二三四五六七八九十\s]*)\s*([章条回巻節号話部編])', refined)
                if m_sec:
                    refined = refined[:m_sec.start()] + f'第{num}{m_sec.group(2)}' + refined[m_sec.end():]
                    break

        # パターン2.5: 丸括弧内の数字照合（年齢・付番・注釈番号、例: タチアナ（41）、ビクテ(60)、（12）など）
        # 閉じ括弧がOCRで欠落している場合（例: ビクテ(()）もスロットとして認識し、適切に補正
        paren_replacements = []
        cur_text_len = max(1, len(refined))
        for m in re.finditer(r'([（\(]+)\s*([0-9０-９!|lI\s]{0,4})\s*([）\)]*)', refined):
            s_idx = m.start()
            e_idx = m.end()

            # 西暦年号・年号範囲（例: (1909~59年), (2014年)）の内部または直前の括弧はスキップ
            if any(p_s <= s_idx and e_idx <= p_e for _, _, p_s, p_e in protected_year_ranges):
                continue
            # 括弧内に4桁数字が含まれる、または直後に年号・波ダッシュが続く場合は年号表記のためスキップ
            sub_snippet = refined[s_idx:min(len(refined), e_idx + 4)]
            if re.search(r'[12][0-9]{3}|[~〜ー\-]|年', sub_snippet):
                continue

            open_p = m.group(1)[0]
            close_p = m.group(3)[-1] if m.group(3) else ("）" if open_p == "（" else ")")
            match_rel_y = (s_idx + e_idx) / (2.0 * float(cur_text_len))

            best_item_idx = None
            best_dist = 999.0

            # 丸括弧近傍に2桁縦中横アイテムが存在する場合、1桁数字（漢字ストローク等の誤爆）に優先
            has_nearby_2d = any(
                idx not in used_item_indices and it.get("method") == "2digit_patch_match" and abs(match_rel_y - it["rel_y"]) < 0.12
                for idx, it in enumerate(sorted_items)
            )

            for idx, it in enumerate(sorted_items):
                if idx in used_item_indices:
                    continue
                item_rel_y = it["rel_y"]
                is_2d = (it.get("method") == "2digit_patch_match")
                if has_nearby_2d and not is_2d:
                    continue
                dist = abs(match_rel_y - item_rel_y)
                effective_dist = dist * 0.5 if is_2d else dist
                if effective_dist < best_dist and dist < 0.20:
                    best_dist = effective_dist
                    best_item_idx = idx

            if best_item_idx is not None:
                it = sorted_items[best_item_idx]
                used_item_indices.add(best_item_idx)
                paren_replacements.append((s_idx, e_idx, f"{open_p}{it['num']}{close_p}", it['num']))

        # パターン3: 単位記号「%」「％」直前の縦中横数字補正
        # 縦書き文書でパーセンテージ（例: 57%、42%）は縦中横で配置され、NDLOCRが1桁落ち（ 7%）や
        # 漢字ストローク誤読（必%、但%、必42%）を起こすのを幾何学的に修復
        cur_text_len = max(1, len(refined))
        pct_replacements = []
        for m in re.finditer(r'([0-9０-９!I|l必但ハ八\s]{0,4})\s*([%％])', refined):
            s_idx = m.start()
            e_idx = m.end()
            pct_char = m.group(2)
            pct_rel_y = (s_idx + e_idx) / (2.0 * float(cur_text_len))

            best_item_idx = None
            best_dist = 999.0
            for idx, it in enumerate(sorted_items):
                if idx in used_item_indices:
                    continue
                # %の直上または近傍（縦書きでは上から下へ流れるため it["rel_y"] <= pct_rel_y + 0.08）
                diff = pct_rel_y - it["rel_y"]
                if -0.06 <= diff <= 0.20:
                    d = abs(diff)
                    if it.get("method") == "2digit_patch_match":
                        d *= 0.5
                    if d < best_dist and d < 0.25:
                        best_dist = d
                        best_item_idx = idx

            if best_item_idx is not None:
                it = sorted_items[best_item_idx]
                used_item_indices.add(best_item_idx)
                pct_replacements.append((s_idx, e_idx, f"{it['num']}{pct_char}"))

        pct_replacements.sort(key=lambda r: -r[0])
        for s_idx, e_idx, rep_str in pct_replacements:
            refined = refined[:s_idx] + rep_str + refined[e_idx:]

        return refined

def refine_vertical_ocr_text(text: str, is_vertical: bool, line_img: np.ndarray, next_line_text: Optional[str] = None) -> str:
    if not is_vertical or line_img is None:
        return text
    refiner = TcyDigitRefiner.get_instance()
    return refiner.refine_line(line_img, text, next_line_text=next_line_text)

def refine_text_with_tcy(line_img: np.ndarray, text: str, next_line_text: Optional[str] = None) -> str:
    refiner = TcyDigitRefiner.get_instance()
    return refiner.refine_line(line_img, text, next_line_text=next_line_text)

