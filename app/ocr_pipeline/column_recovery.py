"""Locate short vertical prose columns that the page OCR detector omitted."""
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np


def _estimate_vertical_lean(image):
    """Estimate column lean from neighboring ordinary-size glyph centers."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    glyphs = np.array([
        (x + w / 2, y + h / 2, w, h)
        for x, y, w, h, area in stats[1:]
        if area >= 12 and 8 <= w <= 48 and 8 <= h <= 48
    ], dtype=float)
    if len(glyphs) < 30:
        return 0.0
    glyph_w, glyph_h = np.median(glyphs[:, 2:], axis=0)
    glyphs = glyphs[(glyphs[:, 2] >= glyph_w * 0.65) & (glyphs[:, 3] >= glyph_h * 0.75)]
    slopes = []
    for glyph in glyphs:
        dx = glyphs[:, 0] - glyph[0]
        dy = glyphs[:, 1] - glyph[1]
        possible = np.flatnonzero(
            (dy >= glyph_h * 0.7) & (dy <= glyph_h * 1.8) & (np.abs(dx) <= glyph_w * 0.6)
        )
        if len(possible):
            neighbor = possible[np.argmin(np.abs(dx[possible]))]
            slopes.append(dx[neighbor] / dy[neighbor])
    return float(np.clip(np.median(slopes), -0.12, 0.12)) if len(slopes) >= 20 else 0.0


def _same_vertical_column(x0, x1, y0, y1, line, lean=0.0):
    """Compare short crops with tall leaning boxes without treating their
    wide swept area as evidence that an adjacent short column was read.
    Similar-height boxes can still use substantial horizontal overlap.
    """
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    cx = (x0 + x1) / 2
    line_cx = line['x'] + line['w'] / 2
    if line['h'] > height * 1.8:
        # Compare the tall column at the crop's height, rather than at its
        # midpoint: its axis shifts horizontally on a leaning scan.
        line_cx += lean * ((y0 + y1) / 2 - (line['y'] + line['h'] / 2))
    centered = abs(cx - line_cx) <= max(8.0, min(width, line['w']) * 0.55)
    x_overlap = max(0, min(x1, line['x'] + line['w']) - max(x0, line['x']))
    y_overlap = max(0, min(y1, line['y'] + line['h']) - max(y0, line['y']))
    similar_height = line['h'] <= height * 1.8
    return (
        (centered or (similar_height and x_overlap >= width * 0.65))
        and y_overlap >= min(height, line['h']) * 0.5
    )


def find_missing_vertical_columns(
    image: np.ndarray,
    lines: List[Dict[str, Any]],
    edge_only: bool = False,
    include_long_edges: bool = False,
) -> List[Tuple[int, int, int, int]]:
    """Return tight crop bounds for unrecognized, normal-size short columns.

    Long recognized prose columns establish the text area. Pixel components
    must have the ordinary body font size; ruby and single marginal numerals
    are not candidates. Return at most three crops to bound retry latency.
    """
    ih, iw = image.shape[:2]
    vertical = [l for l in lines if l.get('is_vertical') and l.get('text', '').strip()]
    long_lines = [l for l in vertical if l.get('h', 0) >= ih * 0.55]
    if len(long_lines) < 4 or ih < 300:
        return []
    centers = sorted(l['x'] + l['w'] / 2 for l in long_lines)
    gaps = [b - a for a, b in zip(centers, centers[1:]) if 16 <= b - a <= 100]
    edge_margin = int(round(np.median(gaps))) if gaps else 32
    rx = max(0, int(min(l['x'] for l in long_lines)) - edge_margin)
    ry = max(0, int(min(l['y'] for l in long_lines)))
    x_end = min(iw, int(max(l['x'] + l['w'] for l in long_lines)) + edge_margin)
    y_end = min(ih, int(max(l['y'] + l['h'] for l in long_lines)))
    if x_end <= rx or y_end <= ry:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    binary = cv2.threshold(gray[ry:y_end, rx:x_end], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if binary.size == 0:
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    max_glyph = max(48, int(min(ih, iw) * 0.08))
    glyphs = [
        (int(x + rx), int(y + ry), int(w), int(h))
        for x, y, w, h, area in stats[1:]
        if area >= 12 and 8 <= w <= max_glyph and 8 <= h <= max_glyph
    ]
    if len(glyphs) < 30:
        return []
    glyph_w = float(np.median([g[2] for g in glyphs]))
    glyph_h = float(np.median([g[3] for g in glyphs]))
    lean = _estimate_vertical_lean(image)
    counts = np.zeros(x_end - rx, dtype=np.int32)
    # Scanned columns can lean across their entire height. Project only the
    # first few glyphs so a neighboring long column's lower end does not bridge
    # the otherwise clear gap around a short paragraph-ending column.
    for x, y, w, _ in glyphs:
        if y <= ry + glyph_h * 6:
            counts[x - rx:x - rx + w] += 1
    active = counts > 0
    candidates = []
    x = 0
    while x < len(active):
        if not active[x]:
            x += 1
            continue
        start = x
        while x < len(active) and active[x]:
            x += 1
        members = [g for g in glyphs if rx + start <= g[0] + g[2] / 2 < rx + x]
        if not members:
            continue
        if include_long_edges and rx + x <= min(l['x'] for l in long_lines) - glyph_w * 0.15:
            # Narrow full-height kana can be absent from the ordinary-width
            # glyph set. Keep them in the outer unread lane so they bridge
            # the column; otherwise a real long tail looks like a short
            # fragment and is rejected as having continuation below it.
            members.extend(
                (int(px + rx), int(py + ry), int(pw), int(ph))
                for px, py, pw, ph, area in stats[1:]
                if area >= 12 and max(3, glyph_w * 0.25) <= pw < 8
                and glyph_h * 0.75 <= ph <= max_glyph
                and rx + start <= px + rx + pw / 2 < rx + x
            )
        members.sort(key=lambda g: g[1])
        contiguous = [members[0]]
        bottom = members[0][1] + members[0][3]
        for glyph in members[1:]:
            if glyph[1] - bottom > glyph_h * 2:
                break
            contiguous.append(glyph)
            bottom = max(bottom, glyph[1] + glyph[3])
        members = contiguous
        if not members:
            continue
        body_members = [g for g in members if g[3] >= glyph_h * 0.65]
        if not body_members:
            continue
        x0 = min(g[0] for g in body_members)
        y0 = min(g[1] for g in body_members)
        x1 = max(g[0] + g[2] for g in body_members)
        y1 = max(g[1] + g[3] for g in body_members)
        tail_edge = x1 <= min(l['x'] for l in long_lines) - glyph_w * 0.15
        if edge_only and not tail_edge:
            continue
        single_glyph = len(body_members) == 1
        if single_glyph:
            # Join disconnected strokes within one body-character footprint.
            # Kana can have a tall left stroke and small right-hand components.
            pieces = [
                (int(x + rx), int(y + ry), int(w), int(h))
                for x, y, w, h, area in stats[1:]
                if area >= 3 and y0 - 2 <= y + ry and y + ry + h <= y1 + 2
                and x0 - glyph_w * 0.2 <= x + rx and x + rx + w <= x0 + glyph_w * 1.4
            ]
            if pieces:
                x0 = min(x0, min(g[0] for g in pieces))
                x1 = max(x1, max(g[0] + g[2] for g in pieces))
            single_width = x1 - x0
            # A single ordinary body glyph is eligible only with a separate
            # small period just below it, in the same vertical text lane.
            periods = [
                (int(x + rx), int(y + ry), int(w), int(h))
                for x, y, w, h, area in stats[1:]
                if area >= 3 and 2 <= w <= glyph_w * 0.55 and 2 <= h <= glyph_h * 0.55
                and 0 <= y + ry - y1 <= glyph_h * 1.1
                and x0 - glyph_w * 0.4 <= x + rx <= x0 + glyph_w * 1.1
            ]
            if not periods:
                continue
            period = min(periods, key=lambda g: g[1])
            x0 = min(x0, period[0])
            x1 = max(x1, period[0] + period[2])
            y1 = period[1] + period[3]
        if not (
            x1 - x0 <= glyph_w * 2.2
            and glyph_h * (1.1 if single_glyph else 1.7) <= y1 - y0 <= (y_end - ry) * (1.05 if include_long_edges and tail_edge else 0.55)
            and y0 <= ry + glyph_h * 4
            and (single_width if single_glyph else np.median([g[2] for g in body_members])) >= glyph_w * 0.65
            and np.median([g[3] for g in body_members]) >= glyph_h * 0.75
        ):
            continue
        # A leaning long column may leave the fixed X lane and look short.
        # Reject it when ordinary-size glyphs continue immediately below its
        # last glyph, just outside that lane. Small final punctuation is kept
        # inside the padded retry crop rather than treated as continuation.
        last = max(members, key=lambda g: g[1] + g[3])
        last_center = last[0] + last[2] / 2
        if any(
            0 <= g[1] - y1 <= glyph_h * 3
            and abs(g[0] + g[2] / 2 - last_center) <= glyph_w * 0.9
            and g[2] >= glyph_w * 0.65 and g[3] >= glyph_h * 0.75
            for g in glyphs
        ):
            continue
        cx = (x0 + x1) / 2
        covered = any(
            _same_vertical_column(x0, x1, y0, y1, line, lean)
            for line in vertical
        )
        if covered:
            continue
        pad_x = max(8, int(round(glyph_w * 0.50)))
        pad_y = max(8, int(round(glyph_h)))
        candidates.append((max(0, x0 - pad_x), max(0, y0 - pad_y), min(iw, x1 + pad_x), min(ih, y1 + pad_y)))
    return sorted(candidates, reverse=True)[:3]


def recover_missing_vertical_columns(image, lines, recognize, edge_only=False, include_long_edges=False):
    """Re-read only missing columns and retain the existing recognized lines."""
    recovered = list(lines)
    crops = find_missing_vertical_columns(image, lines, edge_only=edge_only, include_long_edges=include_long_edges)
    lean = _estimate_vertical_lean(image) if crops else 0.0
    for x0, y0, x1, y1 in crops:
        crop = image[y0:y1, x0:x1]
        retries = recognize(crop)
        agreement_texts = None
        for retry in retries:
            text = str(retry.get('text', '')).strip()
            if (
                not retry.get('is_vertical') or len(text) < 2
                or retry.get('class_index') == 10
            ):
                continue
            if retry.get('confidence', 0) < 0.35:
                # Two or three glyphs can receive low detector confidence even
                # when read correctly. Require an independent scale variant
                # to agree exactly instead of relaxing the score for all text.
                if len(text) > 4 or retry.get('confidence', 0) < 0.15:
                    continue
                if agreement_texts is None:
                    enlarged = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    agreement_texts = {
                        str(line.get('text', '')).strip()
                        for line in recognize(enlarged)
                        if line.get('is_vertical') and line.get('class_index') != 10
                        and line.get('confidence', 0) >= 0.15
                    }
                if text not in agreement_texts:
                    continue
            candidate = {**retry, 'x': int(retry['x'] + x0), 'y': int(retry['y'] + y0)}
            # A crop can also expose part of a neighboring column. Accept only
            # detections centered within the candidate's inner text lane.
            cx = candidate['x'] + candidate['w'] / 2
            if not x0 + 6 <= cx <= x1 - 6:
                continue
            if any(
                line.get('is_vertical')
                and _same_vertical_column(
                    candidate['x'], candidate['x'] + candidate['w'],
                    candidate['y'], candidate['y'] + candidate['h'], line, lean
                )
                for line in recovered
            ):
                continue
            recovered.append(candidate)
    recovered.sort(key=lambda l: (-l['x'], l['y']) if l.get('is_vertical') else (l['y'], l['x']))
    return recovered
