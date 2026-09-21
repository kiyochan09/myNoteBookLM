import os
import sys
import re
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
            if area > 10 and bh > 6 and bw < lw * 0.85:
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
                            used_comp_ids.add(c1["id"])
                            used_comp_ids.add(c2["id"])

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

        # 1. 行頭のアラビア数字・年号・数量・世紀が欠損しているパターンの検出・高精度復元
        needs_leading_check = bool(
            re.match(r'^\s*[0-9一二三四五六七八九十]?\s*(?:年|月|日|世紀|代|人|回|%|％|割|万|千|度|点|部|本|枚|話|章|条|節|番|キロ|メートル)', text)
            or re.match(r'^\s*年(?:当時|代|末|初|半ば|頃|ごろ|前|後|に|は|の|と|で)', text)
        )

        if needs_leading_check:
            # (1) まず画像先頭パッチに対して登録画像テンプレート（1桁/2桁）で直接照合を試みる
            top_char_patch = gray[0:min(lh, 65), :]
            d1, s1 = self.match_1digit_patch(top_char_patch)
            d2, s2 = self.match_2digit_patch(top_char_patch)
            
            best_lead_digit = None
            if s2 >= 0.80 and (s2 >= s1 or s1 < 0.78):
                best_lead_digit = str(d2)
            elif s1 >= 0.78:
                best_lead_digit = str(d1)

            if best_lead_digit:
                m_suf = re.search(r'^\s*[0-9一二三四五六七八九十]*\s*(年|月|日|世紀|代|人|回|%|％|割|万|千|度|点|部|本|枚|話|章|条|節|番|キロ|メートル)', text)
                if m_suf:
                    suf = m_suf.group(1)
                    text = re.sub(rf'^\s*[0-9一二三四五六七八九十]*\s*{re.escape(suf)}', f"{best_lead_digit}{suf}", text)

            # (2) 次にWinOCRによる4桁年号・複数桁日付の補助照合
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

                        # (a) 4桁年号 + 年月日 (例: 2014年2月27日, 2014年5月, 1989年, 2016年当時, 2014年の時, 2014年に)
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
                                else:
                                    text = re.sub(r'^\s*[0-9一二三四五六七八九十]*年', f"{yr}年", text)
                        else:
                            # (b) 1989年などの「989年」脱落パターン
                            m_3digit_year = re.search(r'([89][0-9]{2})\s*年', wt)
                            if m_3digit_year:
                                yr = "1" + m_3digit_year.group(1)
                                text = re.sub(r'^\s*[0-9一二三四五六七八九十]*年', f"{yr}年", text)
                            else:
                                # (c) 単一数字・2桁数字 + 接尾辞 (例: 「5年を迎える」「13世紀」「60代」)
                                m_num_suffix = re.search(r'^([0-9]{1,3})\s*(年|月|日|世紀|代|人|回|%|％|割|万|千|度|点|部|本|枚|話|章|条|節|番|キロ|メートル)', wt)
                                if m_num_suffix:
                                    num = m_num_suffix.group(1)
                                    suf = m_num_suffix.group(2)
                                    text = re.sub(rf'^\s*[0-9一二三四五六七八九十]*{re.escape(suf)}', f"{num}{suf}", text)
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
        return text

    def refine_line(self, line_img: np.ndarray, text: str) -> str:
        if not text:
            return text

        refined = self.refine_kanji_years_and_dates(line_img, text)
        tcy_items = self.find_tcy_in_vertical_line(line_img)
        if not tcy_items:
            return refined

        MULTI_SUFFIXES = ["世紀", "ページ"]
        SINGLE_SUFFIXES = "代歳才人点個回件年月日時分秒%％万千度P部本枚匹頭隻曲皿羽冊階番勝敗円割倍"
        SUFFIX_REGEX = rf'(?:{"|".join(MULTI_SUFFIXES)}|[{SINGLE_SUFFIXES}])'

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

        sorted_items = sorted(tcy_items, key=get_priority)

        # 範囲表記の照合（例: 30〜40代, 10〜20人）
        tcy_pairs = [it for it in sorted_items if it.get("method") == "2digit_patch_match"]
        if len(tcy_pairs) >= 2:
            for k in range(len(tcy_pairs) - 1):
                item1 = tcy_pairs[k]
                item2 = tcy_pairs[k + 1]
                if abs(item2["rel_y"] - item1["rel_y"]) < 0.12:
                    num1, num2 = item1["num"], item2["num"]
                    range_match = re.search(rf'([0-9０-９\s~〜ー\-・切!]*[~〜ー\-][0-9０-９\s~〜ー\-・切!]*)({SUFFIX_REGEX})', refined)
                    if range_match:
                        s_idx = range_match.start()
                        e_idx = range_match.end()
                        suf = range_match.group(2)
                        refined = refined[:s_idx] + f"{num1}〜{num2}{suf}" + refined[e_idx:]
                        sorted_items = [it for it in sorted_items if it not in (item1, item2)]
                        break

        COMPOUND_WORDS_GUARD = {
            "自分", "気分", "充分", "十分", "半分", "身分", "多分", "随分", "五分", "不十分", "十二分", "一部分", "大部分", "過半数",
            "日常", "毎日", "先日", "昨日", "今日", "明日", "平日", "祝日", "休日", "月日", "期日", "連日", "同日", "初日", "終日", "在りし日",
            "人々", "一般人", "軍人", "恩人", "本人", "犯人", "証人", "住人", "邦人", "成人", "老人", "友人", "知人", "他人", "十数人",
            "前年", "昨年", "平年", "今年", "当年", "例年", "連年", "生年", "周年", "新年", "定年", "享年", "十数年",
            "年金", "年間", "年月", "年度", "年齢", "年頭", "年配", "年代",
            "強度", "制度", "高度", "深度", "限度", "速度", "過度", "態度", "適度", "温度", "湿度", "今度", "都度", "再度", "何度",
            "地点", "要点", "重点", "弱点", "利点", "美点", "沸点", "視点", "観点", "原点", "盲点", "論点", "欠点", "争点", "得点", "交点",
            "曲がる", "曲がり", "曲線", "曲折", "名曲", "序曲", "原曲", "楽曲",
            "時期", "時間", "時代", "時折", "時刻", "時々", "当時", "平時", "適時", "定時", "その時", "あの時", "一時",
            "分か", "分かり", "分かる", "分けて", "分かれ", "分散", "分析", "分別", "分離", "分量", "分担", "分野", "分界", "分解",
            "一番", "二番手", "三番手",
            "一部", "全部", "大部", "学部", "幹部", "本部", "支部", "部下", "部長", "部署", "部門", "部屋", "内部", "外部", "北部", "南部", "東部", "西部", "中部", "首脳部", "上部", "下部", "中央部", "周辺部", "細部", "局部", "一部始終",
            "一概", "一定", "一般", "万一", "一様", "一変", "一瞬", "一帯", "一体", "一環", "一息", "一幕", "一角", "一端", "一段", "一途", "一命", "一因", "一握", "一同", "一面", "一読"
        }

        # 4桁年号の出現位置を取得し、その範囲内の数字アイテムが後続の月などに誤マッチしないようマスクする
        text_len = max(1, len(refined))
        four_digit_years = []
        for my in re.finditer(r'([12][0-9]{3})\s*年', refined):
            y_start_rel = my.start() / float(text_len)
            y_end_rel = my.end() / float(text_len)
            four_digit_years.append((y_start_rel, y_end_rel))

        # パターン1: 見出し「第X章」→「第{num}章」
        for item in sorted_items:
            num = item["num"]
            rel_y = item["rel_y"]
            if "第" in refined and any(s in refined for s in ["章", "条", "回", "巻"]) and rel_y < 0.40:
                m_sec = re.search(r'第\s*([0-9０-９Ⅰ-Ⅻ!一二三四五六七八九十切\s]*)\s*([章条回巻節号話部編])', refined)
                if m_sec:
                    refined = refined[:m_sec.start()] + f'第{num}{m_sec.group(2)}' + refined[m_sec.end():]
                    break

        # パターン2: 助数詞・接尾辞との文脈照合（最寄り距離アルゴリズム）
        # 2桁縦中横がある場合は接頭文字0文字も許容（OCR完全脱落対応）
        has_2digit = any(it.get("method") == "2digit_patch_match" for it in sorted_items)
        quantifier = "{0,4}" if has_2digit else "{1,4}"
        suffix_matches = list(re.finditer(rf'([0-9０-９!一二三四五六七八九十切I|l冗らろへー・]{quantifier})\s*({SUFFIX_REGEX})', refined))

        replacements = []
        used_item_indices = set()

        for m in suffix_matches:
            s_idx = m.start()
            e_idx = m.end()
            prefix_tok = m.group(1).strip()
            suf = m.group(2)

            # 4桁年号は上書きしない
            if suf == '年' and len(prefix_tok) >= 4:
                continue

            # 熟語・複合語ガード（当該助数詞 suf を含む複合語が、まさにそのマッチ位置と重複している場合のみ除外）
            is_compound = False
            for cw in COMPOUND_WORDS_GUARD:
                if suf not in cw:
                    continue
                cw_start = refined.rfind(cw, max(0, s_idx - len(cw)), min(text_len, e_idx + len(cw)))
                if cw_start != -1 and (cw_start <= s_idx < cw_start + len(cw)):
                    is_compound = True
                    break
            if is_compound:
                continue

            match_rel_y = s_idx / float(text_len)

            # この助数詞に最も適合するアイテムを探索
            best_item_idx = None
            best_dist = 999.0

            for idx, it in enumerate(sorted_items):
                if idx in used_item_indices:
                    continue
                item_rel_y = it["rel_y"]
                is_2d = (it.get("method") == "2digit_patch_match")
                num = it["num"]

                # 4桁年号の範囲内にある数字アイテムは、年号の外の助数詞にはマッチさせない
                in_four_digit_year = any(y_s - 0.03 <= item_rel_y <= y_e + 0.01 for y_s, y_e in four_digit_years)
                if in_four_digit_year and suf != '年':
                    continue

                # 1桁数字（1または7）の場合、直前に明示的な数字・ノイズ記号がない場合は誤爆防止
                # また、すでにテキストに有効な数字（1以外の数字、例: 5月, 4月）が入っている場合は単体1や7での上書きを禁止
                if not is_2d and num in [1, 7]:
                    if prefix_tok.isdigit() and prefix_tok not in ["1", "7"]:
                        continue
                    if not prefix_tok:
                        continue

                # 1桁数字で接頭辞が0文字の場合はマッチさせない（2桁縦中横のみ0文字許容）
                if not is_2d and len(prefix_tok) == 0:
                    continue

                dist = abs(match_rel_y - item_rel_y)
                # 2桁縦中横なら優先度ボーナス（距離評価を半分にする）
                effective_dist = dist * 0.5 if is_2d else dist

                if effective_dist < best_dist and dist < 0.20:
                    best_dist = effective_dist
                    best_item_idx = idx

            if best_item_idx is not None:
                it = sorted_items[best_item_idx]
                used_item_indices.add(best_item_idx)
                replacements.append((s_idx, e_idx, f"{it['num']}{suf}"))

        # 後ろのインデックスから順に適用して文字列長のズレを防止
        replacements.sort(key=lambda r: -r[0])
        for s_idx, e_idx, rep_str in replacements:
            refined = refined[:s_idx] + rep_str + refined[e_idx:]

        # 行末の「午前」「午後」に続く数字の行またぎ補正
        for item in sorted_items:
            num = item["num"]
            rel_y = item["rel_y"]
            if rel_y > 0.88 and item["score"] >= 0.85 and re.search(r'(午前|午後)\s*$', refined):
                refined = re.sub(r'(午前|午後)\s*$', f'\\g<1>{num}', refined)
                break

        # パターン3: 明白なOCR誤読文字の置換（例: '切', '!', '|'）
        for item in sorted_items:
            is_2digit = (item.get("method") == "2digit_patch_match")
            if is_2digit:
                continue
            num = item["num"]
            rel_y = item["rel_y"]
            if len(refined) > 0:
                text_len = len(refined)
                target_char_idx = int(round(rel_y * text_len))
                target_char_idx = max(0, min(text_len - 1, target_char_idx))
                w_start = max(0, target_char_idx - 2)
                w_end = min(text_len, target_char_idx + 3)
                window_str = refined[w_start:w_end]
                for token in ['切', '!', '|']:
                    if token in window_str:
                        tok_pos = refined.find(token, w_start)
                        if tok_pos != -1:
                            refined = refined[:tok_pos] + str(num) + refined[tok_pos+1:]
                            break

        return refined

def refine_vertical_ocr_text(text: str, is_vertical: bool, line_img: np.ndarray) -> str:
    if not is_vertical or line_img is None:
        return text
    refiner = TcyDigitRefiner.get_instance()
    return refiner.refine_line(line_img, text)

def refine_text_with_tcy(line_img: np.ndarray, text: str) -> str:
    refiner = TcyDigitRefiner.get_instance()
    return refiner.refine_line(line_img, text)
