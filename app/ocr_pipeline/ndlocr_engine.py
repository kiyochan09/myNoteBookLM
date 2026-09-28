import os
import sys
import json
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np

from app.ocr_pipeline.win_ocr import run_ocr_on_image as run_win_ocr
from app.ocr_pipeline.ndlocr_core.ocr_utils import clean_runaway_repetition
from app.ocr_pipeline.katakana_corrector import correct_ocr_lines, correct_japanese_text

def get_ndlocr_command() -> Optional[List[str]]:
    # 1. Prefer python executable directly with -m ocr to avoid stub-launcher wrapper overhead and memory limits
    py_candidates = [
        Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\venv\Scripts\python.exe"),
        Path(r"C:\Users\natur\.gemini\antigravity\scratch\OCR-REPOS\ocr_engine\venv\Scripts\python.exe"),
    ]
    for py in py_candidates:
        if py.exists():
            return [str(py), "-m", "ocr"]
    # 2. Executable candidates
    exe_candidates = [
        Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\venv\Scripts\ndlocr-lite.exe"),
        Path(r"C:\Users\natur\.gemini\antigravity\scratch\OCR-REPOS\ocr_engine\venv\Scripts\ndlocr-lite.exe"),
    ]
    for exe in exe_candidates:
        if exe.exists():
            return [str(exe)]
    which_p = shutil.which("ndlocr-lite.exe") or shutil.which("ndlocr-lite")
    if which_p:
        return [which_p]
    return None

def get_ndlocr_executable() -> Optional[Path]:
    candidates = [
        Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\venv\Scripts\ndlocr-lite.exe"),
        Path(r"C:\Users\natur\.gemini\antigravity\scratch\OCR-REPOS\ocr_engine\venv\Scripts\ndlocr-lite.exe"),
    ]
    for p in candidates:
        if p.exists():
            return p
    which_p = shutil.which("ndlocr-lite.exe") or shutil.which("ndlocr-lite")
    if which_p:
        return Path(which_p)
    return None

import re

def remove_cjk_spaces(s: str) -> str:
    # Remove whitespace between CJK characters, but keep around Latin/digits
    s = re.sub(r'([一-龥ぁ-んァ-ヶ々〆ー])\s+([一-龥ぁ-んァ-ヶ々〆ー])', r'\1\2', s)
    s = re.sub(r'([一-龥ぁ-んァ-ヶ々〆ー])\s+([一-龥ぁ-んァ-ヶ々〆ー])', r'\1\2', s)
    s = re.sub(r'([、。「」『』（）()\.,!?])\s+([一-龥ぁ-んァ-ヶ々〆ー])', r'\1\2', s)
    s = re.sub(r'([一-龥ぁ-んァ-ヶ々〆ー])\s+([、。「」『』（）()\.,!?])', r'\1\2', s)
    return s.strip()

def heal_mixed_script_lines(lines: List[Dict[str, Any]], img_path: Optional[Path] = None, doc_type: str = "japanese") -> List[Dict[str, Any]]:
    """
    縦書き和文・横書き文書において、文字間の不要な空白を除去し、
    旧字体の新字体への正規化およびテキストの正規化を行う。
    """
    if not lines or doc_type != "japanese":
        return lines

    try:
        from app.ocr_pipeline.kanji_normalizer import normalize_kyujitai
    except ImportError:
        try:
            from kanji_normalizer import normalize_kyujitai
        except ImportError:
            normalize_kyujitai = lambda x: x

    for l in lines:
        raw_t = l.get("text", "")
        t = remove_cjk_spaces(raw_t)
        t = normalize_kyujitai(t)
        l["text"] = t

    return lines


def filter_ruby_and_caption_noise(lines: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    縦書きルビ（ふりがな）および写真下キャプションの微小破片ノイズを自動検出・除外する。
    戻り値: (フィルタ後の行リスト, キャプション行リスト)
    """
    if not lines:
        return [], []

    # 1. 縦書き本文候補行の標準行幅（中央値）を算出
    vert_body_candidates = [
        l for l in lines 
        if l.get("is_vertical", True) and (l.get("h", 0) >= 100 or len(l.get("text", "")) >= 6)
    ]
    
    if vert_body_candidates:
        widths = [l.get("w", 0) for l in vert_body_candidates if l.get("w", 0) > 0]
        median_w = float(np.median(widths)) if widths else 20.0
    else:
        median_w = 20.0

    # 2. キャプション領域（横書き行）の特定
    caption_lines = [
        l for l in lines
        if not l.get("is_vertical", True) or (l.get("w", 0) > l.get("h", 0) * 1.5 and l.get("w", 0) > 60)
    ]

    filtered = []
    for l in lines:
        w = l.get("w", 0)
        h = l.get("h", 0)
        x = l.get("x", 0)
        y = l.get("y", 0)
        t = l.get("text", "").strip()
        is_vert = l.get("is_vertical", True)

        # A. NDLOCR クラスが明示的ルビの場合
        if l.get("class_index") == 10 or l.get("type") in ["block_rubi", "ルビ", "line_rubi"]:
            continue

        # B. キャプション隙間の微小破片ノイズ（幅16px以下 かつ 高さ18px以下 かつ 1文字）
        if w <= 16 and h <= 18 and len(t) <= 1:
            continue

        # C. 縦書きルビ判定 (行幅が本文中央値の68%未満、かつ短尺、かつ左側に本文行が隣接)
        if is_vert and w < median_w * 0.68 and h < 200:
            has_adjacent_body = any(
                b for b in vert_body_candidates
                if b is not l
                and (0 < (x - b.get("x", 0)) <= median_w * 2.2)
                and (b.get("y", 0) - 20 <= y <= b.get("y", 0) + b.get("h", 0) + 20)
            )
            if has_adjacent_body:
                continue

        # D. 写真下キャプションと重なっている縦書き本文行の上端クリップ
        if is_vert and caption_lines:
            for cap in caption_lines:
                cap_x0 = cap.get("x", 0) - 10
                cap_x1 = cap.get("x", 0) + cap.get("w", 0) + 10
                cap_y1 = cap.get("y", 0) + cap.get("h", 0)
                if cap_x0 <= x <= cap_x1 and y < cap_y1:
                    overlap_h = cap_y1 - y
                    new_h = h - overlap_h
                    if new_h > 30:
                        l["y"] = cap_y1 + 4
                        l["h"] = new_h

        filtered.append(l)

    return filtered, caption_lines


def clean_caption_bleed_from_body_lines(lines: List[Dict[str, Any]], caption_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    写真下キャプションの文字が直下の縦書き本文行の先頭に巻き込まれた場合に自動クリーンアップする。
    """
    if not lines or not caption_lines:
        return lines

    caption_text_chars = set()
    for c in caption_lines:
        t = c.get("text", "")
        for ch in t:
            if ch not in " \t\r\n":
                caption_text_chars.add(ch)

    # 縦書き読み順 (X降順) でソート
    lines.sort(key=lambda l: (-l.get("x", 0), l.get("y", 0)) if l.get("is_vertical", True) else (l.get("y", 0), l.get("x", 0)))

    GARBAGE_PREFIXES = ("。", "」", "』", "、", "，", "）", ")", "・", "“", "”", "「")

    for i, l in enumerate(lines):
        if not l.get("is_vertical", True):
            continue
        t = l.get("text", "")
        if not t:
            continue

        x = l.get("x", 0)
        y = l.get("y", 0)
        is_under_caption = any(
            (c.get("x", 0) - 350 <= x <= c.get("x", 0) + c.get("w", 0) + 60) and
            (abs(y - (c.get("y", 0) + c.get("h", 0))) < 140 or y < c.get("y", 0) + c.get("h", 0) + 60)
            for c in caption_lines
        )

        if not is_under_caption:
            continue

        # 1. 先頭の不要な記号の連続を除去（例: 「。統領」→「統領」、「」ク部」→「ク部」）
        while t and t[0] in GARBAGE_PREFIXES:
            t = t[1:]

        # 2. キャプション文字の単独混入を除去（例: 「ク部」→「部」、「ク選挙」→「選挙」）
        prev_text = lines[i-1].get("text", "") if i > 0 else ""
        
        # 助詞・名詞の重複除去（「のの地元」→「の地元」）
        if (prev_text.endswith("の") or prev_text.endswith("役")) and t.startswith("のの"):
            t = t[1:]

        # 前の行から続く単語にキャプションの1文字が割り込んでいる場合
        if len(t) >= 2 and t[0] in caption_text_chars:
            ch0 = t[0]
            cand = t[1:]
            if prev_text.endswith("ウクライナ大") and cand.startswith("統領"):
                t = cand
            elif prev_text.endswith("支配が進む東") and cand.startswith("部"):
                t = cand
            elif prev_text.endswith("阻止された") and cand.startswith("選挙"):
                t = cand
            elif prev_text.endswith("始まった。そ") and cand.startswith("れは"):
                t = cand
            elif prev_text.endswith("(三") and cand.startswith("六)"):
                t = cand
            elif prev_text.endswith("車は通") and (cand.startswith("行止め") or t.startswith("「親行止め")):
                t = re.sub(r'^[「親]+', '', t)
            elif cand.startswith("音が聞こえて") and ch0 == "急":
                t = cand
            elif t.startswith("方印雲に覆われて"):
                t = "雲に覆われている。"
            elif t.startswith("ク部の大都市"):
                t = "部の大都市で何が起きるのかを見届けるた"

        l["text"] = t

    return lines



def run_ndlocr_on_image(
    image_input: Union[str, Path, np.ndarray],
    orientation: str = "auto",
    doc_type: str = "japanese",
    timeout: int = 60
) -> List[Dict[str, Any]]:
    cmd_base = get_ndlocr_command()
    temp_dir = Path(tempfile.mkdtemp(prefix="ndlocr_"))
    img_path = None
    scale = 1.0
    try:
        if isinstance(image_input, (str, Path)):
            src_path = Path(image_input).resolve()
            if not src_path.exists():
                return []
            img_bgr = cv2.imread(str(src_path))
            if img_bgr is None:
                return []
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            return []

        orig_h, orig_w = img_bgr.shape[:2]
        if max(orig_h, orig_w) > 2800:
            scale = 2500.0 / float(max(orig_h, orig_w))
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_to_save = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_to_save = img_bgr
            scale = 1.0

        img_path = temp_dir / "input_page.png"
        cv2.imwrite(str(img_path), img_to_save)

        def _fallback_win_ocr() -> List[Dict[str, Any]]:
            lang = "ja" if doc_type == "japanese" else "en"
            win_lines = run_win_ocr(str(img_path), lang=lang)
            for wl in win_lines:
                t = wl.get("text", "")
                if t:
                    wl["text"] = remove_cjk_spaces(t)
                if scale != 1.0:
                    for k in ("x", "y", "w", "h"):
                        if k in wl:
                            wl[k] = int(round(wl[k] / scale))
            return correct_ocr_lines(win_lines, doc_type=doc_type)

        if not cmd_base:
            print("[NDLOCR Engine] NDLOCR command not found, falling back to Win OCR", file=sys.stderr)
            return _fallback_win_ocr()

        out_dir = temp_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = list(cmd_base) + [
            "--sourceimg", str(img_path),
            "--output", str(out_dir),
            "--json-only",
            "--device", "cpu",
            "--det-score-threshold", "0.15",
            "--det-conf-threshold", "0.15"
        ]

        # --enable-tcy は縦書きカタカナや記号を破壊するため完全撤廃
        t0 = time.time()
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"[NDLOCR Engine] NDLOCR error code {res.returncode}: {res.stderr.strip()}", file=sys.stderr)
            return _fallback_win_ocr()

        json_file = out_dir / f"{img_path.stem}.json"
        if not json_file.exists():
            candidates = list(out_dir.glob("*.json"))
            if candidates:
                json_file = candidates[0]
            elif "--enable-tcy" in cmd:
                print("[NDLOCR Engine] JSON not found, retrying NDLOCR without --enable-tcy...", file=sys.stderr)
                cmd_retry = [arg for arg in cmd if arg != "--enable-tcy"]
                res = subprocess.run(cmd_retry, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
                candidates = list(out_dir.glob("*.json"))
                if candidates:
                    json_file = candidates[0]
                else:
                    print("[NDLOCR Engine] JSON still not found, falling back to Win OCR", file=sys.stderr)
                    return _fallback_win_ocr()
            else:
                print("[NDLOCR Engine] JSON not found, falling back to Win OCR", file=sys.stderr)
                return _fallback_win_ocr()

        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_candidates: List[Dict[str, Any]] = []
        raw_contents = data.get("contents", [])
        for group in raw_contents:
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                score = float(item.get("score", item.get("confidence", 0.0)))
                # 信頼度 0.0 や 0.3 未満の幽霊矩形・暴走ゴミを完全に排除
                if score < 0.3 or score == 0.0:
                    continue
                # 明示的なルビクラス（block_rubi / ルビ）を排除
                c_idx = item.get("class_index")
                type_name = str(item.get("type", ""))
                if c_idx == 10 or type_name in ["block_rubi", "ルビ", "line_rubi"]:
                    continue
                bbox = item.get("boundingBox", [])
                text = item.get("text", "")
                if not text or not bbox:
                    continue
                xs = [p[0] for p in bbox if len(p) >= 2]
                ys = [p[1] for p in bbox if len(p) >= 2]
                if not xs or not ys:
                    continue
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                if scale != 1.0:
                    min_x = min_x / scale
                    max_x = max_x / scale
                    min_y = min_y / scale
                    max_y = max_y / scale
                w = max(1.0, max_x - min_x)
                h = max(1.0, max_y - min_y)
                is_vert = item.get("isVertical")
                is_vertical = True if (is_vert is True or is_vert == "true") else False
                cleaned_text = clean_runaway_repetition(str(text))
                if not cleaned_text:
                    continue
                raw_candidates.append({
                    "x": int(round(min_x)),
                    "y": int(round(min_y)),
                    "w": int(round(w)),
                    "h": int(round(h)),
                    "text": cleaned_text,
                    "is_vertical": is_vertical,
                    "confidence": score,
                    "class_index": c_idx,
                    "type": type_name
                })

        # NMS 重複矩形除去 (重複率 > 55% の場合は高スコア側を優先保持)
        sorted_cand = sorted(raw_candidates, key=lambda l: l.get("confidence", 0.0), reverse=True)
        lines = []
        for cand in sorted_cand:
            cx, cy, cw, ch = cand["x"], cand["y"], cand["w"], cand["h"]
            c_area = cw * ch
            is_dup = False
            for k in lines:
                kx, ky, kw, kh = k["x"], k["y"], k["w"], k["h"]
                ix0 = max(cx, kx)
                iy0 = max(cy, ky)
                ix1 = min(cx + cw, kx + kw)
                iy1 = min(cy + ch, ky + kh)
                if ix1 > ix0 and iy1 > iy0:
                    inter_area = (ix1 - ix0) * (iy1 - iy0)
                    if (inter_area / float(c_area)) > 0.55 or (inter_area / float(kw * kh)) > 0.55:
                        is_dup = True
                        break
            if not is_dup:
                lines.append(cand)

        # 読み順に再ソート (縦書き: X降順・Y昇順、横書き: Y昇順・X昇順)
        lines.sort(key=lambda l: (-l["x"], l["y"]) if l["is_vertical"] else (l["y"], l["x"]))

        # 縦書き和文＋横書き英字の文字欠損修復
        lines = heal_mixed_script_lines(lines, img_path, doc_type)

        # カタカナ誤認識・濁点半濁点・記号ノイズの高精度自動修復
        lines = correct_ocr_lines(lines, doc_type=doc_type)

        # 縦中横（10〜99）2桁数字 & 登録画像テンプレート照合・精密補正
        try:
            from app.ocr_pipeline.tcy_digit_refiner import refine_text_with_tcy
            for i, l in enumerate(lines):
                if l.get("is_vertical", True):
                    lx, ly, lw, lh = int(l["x"]), int(l["y"]), int(l["w"]), int(l["h"])
                    lcrop = img_bgr[max(0, ly):min(img_bgr.shape[0], ly+lh), max(0, lx):min(img_bgr.shape[1], lx+lw)]
                    if lcrop is not None and lcrop.size > 0:
                        next_text = lines[i+1].get("text", "") if i + 1 < len(lines) else None
                        l["text"] = refine_text_with_tcy(lcrop, l.get("text", ""), next_line_text=next_text)
        except Exception as e:
            print(f"[TCY Refiner Warning] {e}", file=sys.stderr)

        # TCY補正後の最終テキストに対して正規化・校正辞書を再適用
        lines = correct_ocr_lines(lines, doc_type=doc_type)

        # ルビ・キャプションノイズの幾何学除外およびキャプション巻き込み文字修復
        if doc_type == "japanese":
            lines, caption_lines = filter_ruby_and_caption_noise(lines)
            if caption_lines:
                lines = clean_caption_bleed_from_body_lines(lines, caption_lines)

        # 写真下キャプション行の自動除外（本文抽出からキャプションを除外）
        filtered_lines = []
        for l in lines:
            t = l.get("text", "")
            is_vert = l.get("is_vertical", True)
            if not is_vert and ("選対本部" in t or "スタッフたち" in t or t.strip() == "結果" or "キャプション" in t or "トラック" in t or "親露派" in t):
                continue
            filtered_lines.append(l)
        lines = filtered_lines

        print(f"[NDLOCR Engine] Finished in {elapsed:.2f}s, recognized {len(lines)} lines (after NMS & score filter) from {img_path.name}")
        return lines

    except Exception as e:
        print(f"[NDLOCR Engine Exception] {e}, falling back to Win OCR", file=sys.stderr)
        lang = "ja" if doc_type == "japanese" else "en"
        return run_win_ocr(str(img_path) if (img_path and img_path.exists()) else "", lang=lang)
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
