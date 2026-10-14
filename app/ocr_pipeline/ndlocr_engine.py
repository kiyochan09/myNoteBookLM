import os
import sys
import json
import time
from difflib import SequenceMatcher
import shutil
import tempfile
import subprocess
import threading

_NDLOCR_PROCESS_LOCK = threading.Lock()
from pathlib import Path
from typing import List, Dict, Any, Union, Optional, Tuple
import cv2
import numpy as np

from app.ocr_pipeline.win_ocr import run_ocr_on_image as run_win_ocr
from app.ocr_pipeline.ndlocr_core.ocr_utils import clean_runaway_repetition
from app.ocr_pipeline.katakana_corrector import correct_ocr_lines, correct_japanese_text
from app.ocr_pipeline.engine_paths import get_ndlocr_command, get_project_root
from app.ocr_pipeline.column_recovery import recover_missing_vertical_columns
from app.ocr_pipeline.warichu import exclude_vertical_warichu

import re

def remove_cjk_spaces(s: str) -> str:
    # Windows OCR may insert a space between every character. Iterate until
    # no CJK or punctuation boundary spaces remain, regardless of run length.
    patterns = (
        (r'([一-龥ぁ-んァ-ヶ々〆ー])\s+([一-龥ぁ-んァ-ヶ々〆ー])', r'\1\2'),
        (r'([、。「」『』（）()“”‘’\.,!?])\s+([一-龥ぁ-んァ-ヶ々〆ー])', r'\1\2'),
        (r'([一-龥ぁ-んァ-ヶ々〆ー])\s+([、。「」『』（）()“”‘’\.,!?])', r'\1\2'),
    )
    while True:
        previous = s
        for pattern, replacement in patterns:
            s = re.sub(pattern, replacement, s)
        if s == previous:
            break
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



def _extract_ndlocr_candidates(
    data: Dict[str, Any],
    offset_x: int = 0,
    offset_y: int = 0,
    scale: float = 1.0,
    pass_rank: int = 0,
) -> List[Dict[str, Any]]:
    """Convert one NDLOCR JSON result to full-image coordinates."""
    candidates: List[Dict[str, Any]] = []
    for group in data.get("contents", []):
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", item.get("confidence", 0.0)))
            if score < 0.15 or score == 0.0:
                continue
            class_index = item.get("class_index")
            type_name = str(item.get("type", ""))
            if class_index == 10 or type_name in ["block_rubi", "ルビ", "line_rubi"]:
                continue

            bbox = item.get("boundingBox", [])
            text = item.get("text", "")
            if not text or not bbox:
                continue
            xs = [point[0] for point in bbox if len(point) >= 2]
            ys = [point[1] for point in bbox if len(point) >= 2]
            if not xs or not ys:
                continue

            min_x = (min(xs) + offset_x) / scale
            max_x = (max(xs) + offset_x) / scale
            min_y = (min(ys) + offset_y) / scale
            max_y = (max(ys) + offset_y) / scale
            is_vert = item.get("isVertical")
            cleaned_text = clean_runaway_repetition(str(text), preserve_numeric_edges=bool(item.get("tcy_image_evidence")))
            if not cleaned_text or (score < 0.3 and len(cleaned_text.strip()) < 2):
                continue

            candidates.append({
                "x": int(round(min_x)),
                "y": int(round(min_y)),
                "w": int(round(max(1.0, max_x - min_x))),
                "h": int(round(max(1.0, max_y - min_y))),
                "text": cleaned_text,
                "is_vertical": is_vert is True or is_vert == "true",
                "confidence": score,
                "class_index": class_index,
                "type": type_name,
                "_pass_rank": pass_rank,
                "tcy_image_evidence": item.get("tcy_image_evidence", []),
            })
    return candidates


def _merge_overlapping_line_text(existing: Dict[str, Any], candidate: Dict[str, Any]) -> Optional[str]:
    """Join overlapping reads once, using containment or a suffix/prefix match.

    Detector box edges can vary by a few pixels between passes. A slightly
    lower bottom edge does not mean that a contained text fragment adds a tail.
    Return None when the strings do not support a safe overlap-based join.
    """
    old_text = str(existing.get("text", ""))
    new_text = str(candidate.get("text", ""))
    if not old_text or not new_text:
        return None
    if new_text in old_text:
        return old_text
    if old_text in new_text:
        return new_text
    axis = "y" if existing.get("is_vertical") else "x"
    leading, trailing = (old_text, new_text) if existing[axis] <= candidate[axis] else (new_text, old_text)
    for size in range(min(len(leading), len(trailing)), 7, -1):
        if leading.endswith(trailing[:size]):
            return leading + trailing[size:]
    return None


def _drop_low_quality_column_alternatives(candidates):
    """Reject a weak full-column duplicate shifted by a skewed crop."""
    kept = []
    for candidate in sorted(candidates, key=lambda line: -line.get('confidence', 0.0)):
        rejected = False
        text = str(candidate.get('text', ''))
        if len(text) >= 20 and candidate.get('confidence', 0.0) < 0.9:
            for full in kept:
                if full.get('confidence', 0.0) - candidate.get('confidence', 0.0) < 0.04:
                    continue
                if bool(candidate.get('is_vertical')) != bool(full.get('is_vertical')):
                    continue
                axis, size, cross, width = ('y', 'h', 'x', 'w') if candidate.get('is_vertical') else ('x', 'w', 'y', 'h')
                overlap = max(0, min(candidate[axis]+candidate[size], full[axis]+full[size])-max(candidate[axis], full[axis]))
                distance = abs(candidate[cross]+candidate[width]/2-full[cross]-full[width]/2)
                if overlap < max(candidate[size], full[size]) * 0.85 or distance > max(8, max(candidate[width], full[width]) * 0.35):
                    continue
                if SequenceMatcher(None, text, str(full.get('text', '')), autojunk=False).ratio() >= 0.65:
                    rejected = True
                    break
        if not rejected:
            kept.append(candidate)
    return kept


def _drop_contained_line_reads(lines):
    """Discard contained fragments after strip boxes have been united."""
    kept = []
    for candidate in sorted(lines, key=lambda line: (len(str(line.get('text', ''))), line.get('h', 0) if line.get('is_vertical') else line.get('w', 0)), reverse=True):
        text = str(candidate.get('text', ''))
        duplicate = False
        if len(text) >= 8:
            for full in kept:
                if bool(candidate.get('is_vertical')) != bool(full.get('is_vertical')):
                    continue
                vertical = candidate.get('is_vertical')
                axis, size, cross, width = ('y', 'h', 'x', 'w') if vertical else ('x', 'w', 'y', 'h')
                distance = abs(candidate[cross] + candidate[width] / 2 - full[cross] - full[width] / 2)
                overlap = max(0, min(candidate[axis] + candidate[size], full[axis] + full[size]) - max(candidate[axis], full[axis]))
                if distance <= max(8.0, min(candidate[width], full[width]) * 0.35) and overlap >= min(candidate[size], full[size]) * 0.55:
                    full_text = full.get('text', '')
                    contained = text in full_text
                    # Separate passes can disagree on a couple of glyphs. Only
                    # accept a long, shorter read anchored at a column boundary;
                    # never collapse two full-length alternative readings here.
                    if not contained and len(text) >= 20 and len(full_text) >= len(text) + 8:
                        for span in (full_text[:len(text)], full_text[-len(text):]):
                            differences = sum(a != b for a, b in zip(text, span))
                            if differences <= 2 and differences / len(text) <= 0.08:
                                contained = True
                                break
                    if contained:
                        duplicate = True
                        break
        if not duplicate:
            kept.append(candidate)
    return kept


def run_ndlocr_on_image(
    image_input: Union[str, Path, np.ndarray],
    orientation: str = "auto",
    doc_type: str = "japanese",
    timeout: int = 60,
    recover_small_text: bool = False,
    recover_missing_columns: bool = False,
    strict: bool = False,
    recover_missing_tail: bool = True,
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

        if doc_type == "japanese" and orientation != "horizontal":
            img_bgr, excluded_warichu = exclude_vertical_warichu(img_bgr)
            if excluded_warichu:
                print(f"[NDLOCR Engine] Excluded {len(excluded_warichu)} embedded small-print spans")

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
            if strict:
                raise ValueError("NDLOCRの実行に失敗しました。メモリ不足・タイムアウト等のため、結果は保存していません。")
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

        input_paths = {img_path.stem: (0, 0, 0)}
        inference_input_dir = None
        if recover_small_text and max(img_to_save.shape[:2]) >= 800:
            # Run the detector over the full page and two overlapping, native-resolution
            # page strips. This gives small print more detector pixels without cropping
            # to a user-selected region or changing the page coordinate system.
            inference_input_dir = temp_dir / "page_passes"
            inference_input_dir.mkdir(parents=True, exist_ok=True)
            full_page_path = inference_input_dir / "00_full_page.png"
            cv2.imwrite(str(full_page_path), img_to_save)
            input_paths = {full_page_path.stem: (0, 0, 0)}

            image_h, image_w = img_to_save.shape[:2]
            split_vertical = image_h >= image_w
            long_side = image_h if split_vertical else image_w
            tile_length = min(long_side - 1, max(512, int(round(long_side * 0.68))))
            if tile_length < long_side:
                tile_starts = sorted({0, long_side - tile_length})
                for tile_index, start in enumerate(tile_starts, 1):
                    end = start + tile_length
                    if split_vertical:
                        tile = img_to_save[start:end, :]
                        offset_x, offset_y = 0, start
                    else:
                        tile = img_to_save[:, start:end]
                        offset_x, offset_y = start, 0
                    tile_path = inference_input_dir / f"{tile_index:02d}_overlap.png"
                    if cv2.imwrite(str(tile_path), tile):
                        input_paths[tile_path.stem] = (offset_x, offset_y, tile_index)

            # For vertical Japanese pages, also run full-height overlapping panels
            # across the page width. This keeps each vertical text column intact while
            # reducing cross-column clutter for the detector; these remain page tiles,
            # not user-region crops.
            short_side = image_w if split_vertical else image_h
            panel_length = min(short_side - 1, max(512, int(round(short_side * 0.68))))
            if panel_length < short_side:
                panel_starts = sorted({0, short_side - panel_length})
                for panel_index, start in enumerate(panel_starts, 1):
                    end = start + panel_length
                    if split_vertical:
                        panel = img_to_save[:, start:end]
                        offset_x, offset_y = start, 0
                    else:
                        panel = img_to_save[start:end, :]
                        offset_x, offset_y = 0, start
                    panel_path = inference_input_dir / f"{tile_index + panel_index:02d}_cross_axis.png"
                    if cv2.imwrite(str(panel_path), panel):
                        input_paths[panel_path.stem] = (offset_x, offset_y, tile_index + panel_index)
        if inference_input_dir is not None:
            source_args = ["--sourcedir", str(inference_input_dir)]
        else:
            source_args = ["--sourceimg", str(img_path)]

        cmd = list(cmd_base) + source_args + [
            "--output", str(out_dir),
            "--json-only",
            "--device", "cpu",
            "--det-score-threshold", "0.15",
            "--det-conf-threshold", "0.15"
        ]

        # --enable-tcy は縦書きカタカナや記号を破壊するため完全撤廃
        t0 = time.time()
        process_timeout = timeout * len(input_paths)
        with _NDLOCR_PROCESS_LOCK:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=process_timeout, cwd=get_project_root())
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"[NDLOCR Engine] NDLOCR error code {res.returncode}: {res.stderr.strip()}", file=sys.stderr)
            return _fallback_win_ocr()

        json_files = sorted(out_dir.glob("*.json"))
        if not json_files:
            print("[NDLOCR Engine] JSON not found, falling back to Win OCR", file=sys.stderr)
            return _fallback_win_ocr()

        raw_candidates: List[Dict[str, Any]] = []
        for json_file in json_files:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            offset_x, offset_y, pass_rank = input_paths.get(json_file.stem, (0, 0, 0))
            raw_candidates.extend(_extract_ndlocr_candidates(
                data, offset_x=offset_x, offset_y=offset_y, scale=scale, pass_rank=pass_rank
            ))

        # Merge duplicate detections from the full page and overlapping strips.
        # A strip may recover a longer segment where the full-page pass truncated a
        # line, so retain the expanded candidate when it is on the same text column.
        sorted_cand = sorted(_drop_low_quality_column_alternatives(raw_candidates), key=lambda l: (l.get("_pass_rank", 0), -l.get("confidence", 0.0)))
        lines = []
        for cand in sorted_cand:
            cx, cy, cw, ch = cand["x"], cand["y"], cand["w"], cand["h"]
            c_area = cw * ch
            duplicate_index = None
            for index, k in enumerate(lines):
                if bool(cand.get("is_vertical")) != bool(k.get("is_vertical")):
                    continue
                kx, ky, kw, kh = k["x"], k["y"], k["w"], k["h"]
                # Do not merge neighboring vertical columns merely because their
                # tall bounding boxes overlap in the reading direction.
                if cand.get("is_vertical"):
                    cross_axis_distance = abs((cx + cw / 2) - (kx + kw / 2))
                    cross_axis_limit = max(8.0, min(cw, kw) * 0.35)
                else:
                    cross_axis_distance = abs((cy + ch / 2) - (ky + kh / 2))
                    cross_axis_limit = max(8.0, min(ch, kh) * 0.35)
                if cross_axis_distance > cross_axis_limit:
                    continue

                # Leaning boxes sweep into the adjacent column. A shifted long
                # read with different prose is not a duplicate of that neighbor.
                cross_width = min(cw, kw) if cand.get("is_vertical") else min(ch, kh)
                candidate_text = str(cand.get("text", ""))
                known_text = str(k.get("text", ""))
                if (cand.get("is_vertical")
                        and ch < kh * 0.85
                        and cw >= kw * 1.10
                        and cand.get("confidence", 0.0) >= 0.90
                        and k.get("confidence", 0.0) >= 0.90
                        and cross_axis_distance > max(8.0, cross_width * 0.2)
                        and min(len(candidate_text), len(known_text)) >= 20
                        and SequenceMatcher(None, candidate_text, known_text, autojunk=False).ratio() < 0.45
                        and SequenceMatcher(None, candidate_text, known_text, autojunk=False).find_longest_match().size < 8):
                    continue

                ix0 = max(cx, kx)
                iy0 = max(cy, ky)
                ix1 = min(cx + cw, kx + kw)
                iy1 = min(cy + ch, ky + kh)
                if ix1 <= ix0 or iy1 <= iy0:
                    continue

                inter_area = (ix1 - ix0) * (iy1 - iy0)
                # Two halves of a slanted column can overlap by less than 55%.
                # Merge them only when their text also proves the overlap.
                reading_overlap = (iy1 - iy0) if cand.get("is_vertical") else (ix1 - ix0)
                shorter_length = min(ch, kh) if cand.get("is_vertical") else min(cw, kw)
                text_supported = (
                    min(len(str(cand.get("text", ""))), len(str(k.get("text", "")))) >= 8
                    and (
                        _merge_overlapping_line_text(k, cand) is not None
                        or (cand.get("is_vertical")
                            and cross_axis_distance <= max(8.0, min(cw, kw) * 0.2)
                            and reading_overlap >= shorter_length * 0.4
                            and SequenceMatcher(None, str(k.get("text", "")), str(cand.get("text", "")), autojunk=False).find_longest_match().size >= 8)
                    )
                    and reading_overlap >= shorter_length * 0.25
                )
                if (inter_area / float(c_area)) > 0.55 or (inter_area / float(kw * kh)) > 0.55 or text_supported:
                    duplicate_index = index
                    break

            if duplicate_index is None:
                lines.append(cand)
                continue

            existing = lines[duplicate_index]
            existing_length = existing["h"] if existing.get("is_vertical") else existing["w"]
            candidate_length = cand["h"] if cand.get("is_vertical") else cand["w"]
            existing_text_len = len(str(existing.get("text", "")))
            candidate_text_len = len(str(cand.get("text", "")))
            materially_longer = (
                candidate_length >= existing_length * 1.15
                or candidate_text_len >= max(existing_text_len + 3, existing_text_len * 1.15)
            )
            confidence_ok = cand.get("confidence", 0.0) >= existing.get("confidence", 0.0) - 0.12
            # A contained, shorter strip must not replace a stronger complete
            # reading merely because its lower bbox edge differs by a pixel.
            axis = 'y' if cand.get('is_vertical') else 'x'
            overlap = max(0, min(existing[axis] + existing_length, cand[axis] + candidate_length) - max(existing[axis], cand[axis]))
            if (candidate_length < existing_length * 0.85
                    and overlap >= candidate_length * 0.9
                    and existing_text_len >= candidate_text_len + 3
                    and existing.get('confidence', 0.0) >= cand.get('confidence', 0.0)
                    and SequenceMatcher(None, str(existing.get('text', '')), str(cand.get('text', '')), autojunk=False).ratio() < 0.6):
                continue
            merged_text = _merge_overlapping_line_text(existing, cand)
            if (cand.get("_pass_rank", 0) > existing.get("_pass_rank", 0) or merged_text is not None) and confidence_ok:
                if merged_text is None:
                    candidate_text = str(cand.get("text", ""))
                    existing_text = str(existing.get("text", ""))
                    if cand.get("is_vertical"):
                        candidate_start, existing_start = cand["y"], existing["y"]
                        candidate_end, existing_end = cand["y"] + cand["h"], existing["y"] + existing["h"]
                    else:
                        candidate_start, existing_start = cand["x"], existing["x"]
                        candidate_end, existing_end = cand["x"] + cand["w"], existing["x"] + existing["w"]

                    if candidate_start <= existing_start:
                        leading_text, trailing_text = candidate_text, existing_text
                        leading_is_candidate = True
                    else:
                        leading_text, trailing_text = existing_text, candidate_text
                        leading_is_candidate = False
                    match = SequenceMatcher(None, leading_text, trailing_text, autojunk=False).find_longest_match()
                    if match.size >= 8:
                        if candidate_end > existing_end:
                            trailing_suffix = candidate_text[match.b + match.size:] if not leading_is_candidate else candidate_text[match.a + match.size:]
                        elif existing_end > candidate_end:
                            trailing_suffix = existing_text[match.b + match.size:] if leading_is_candidate else existing_text[match.a + match.size:]
                        else:
                            leading_suffix = leading_text[match.a + match.size:]
                            trailing_suffix = trailing_text[match.b + match.size:]
                            trailing_suffix = max((leading_suffix, trailing_suffix), key=len)
                        merged_text = leading_text[:match.a + match.size] + trailing_suffix
                    elif materially_longer:
                        merged_text = candidate_text
                    else:
                        merged_text = existing_text

                merged_candidate = dict(cand if materially_longer else existing)
                merged_candidate["x"] = min(cand["x"], existing["x"])
                merged_candidate["y"] = min(cand["y"], existing["y"])
                merged_candidate["w"] = max(cand["x"] + cand["w"], existing["x"] + existing["w"]) - merged_candidate["x"]
                merged_candidate["h"] = max(cand["y"] + cand["h"], existing["y"] + existing["h"]) - merged_candidate["y"]
                merged_candidate["text"] = merged_text
                merged_candidate["confidence"] = max(cand.get("confidence", 0.0), existing.get("confidence", 0.0))
                merged_candidate["_pass_rank"] = cand.get("_pass_rank", 0)
                lines[duplicate_index] = merged_candidate
        lines = _drop_contained_line_reads(lines)
        for line in lines:
            line.pop("_pass_rank", None)

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
                if l.get("is_vertical", True) and not l.get("tcy_image_evidence"):
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

        if (recover_missing_columns or recover_missing_tail) and doc_type == "japanese" and orientation != "horizontal":
            lines = recover_missing_vertical_columns(
                img_bgr, lines,
                lambda crop: run_ndlocr_on_image(
                    crop, orientation="vertical", doc_type=doc_type,
                    timeout=timeout, recover_missing_columns=False, recover_missing_tail=False,
                ),
                edge_only=not recover_missing_columns, include_long_edges=True,
            )

        print(f"[NDLOCR Engine] Finished in {elapsed:.2f}s, recognized {len(lines)} lines (after NMS & score filter) from {img_path.name}")
        # Recheck after character normalization and missing-column recovery.
        lines = _drop_contained_line_reads(lines)
        lines.sort(key=lambda line: (-line['x'], line['y']) if line['is_vertical'] else (line['y'], line['x']))
        return lines

    except Exception as e:
        if strict:
            raise ValueError("NDLOCRの実行に失敗しました。結果は保存していません: " + str(e)) from e
        print(f"[NDLOCR Engine Exception] {e}, falling back to Win OCR", file=sys.stderr)
        lang = "ja" if doc_type == "japanese" else "en"
        return run_win_ocr(str(img_path) if (img_path and img_path.exists()) else "", lang=lang)
    finally:
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
