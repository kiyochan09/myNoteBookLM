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
                    "confidence": score
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

        # 写真下キャプション行の自動除外（ユーザー要望: 本文抽出からキャプションを除外）
        filtered_lines = []
        for l in lines:
            t = l.get("text", "")
            is_vert = l.get("is_vertical", True)
            if not is_vert and ("選対本部" in t or "スタッフたち" in t or t.strip() == "結果" or "キャプション" in t):
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
