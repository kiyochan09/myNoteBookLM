"""Split isolated vertical headings using page pixels, without recognizing text."""

from typing import Any, Dict, List

import cv2
import numpy as np


def split_vertical_heading_regions(
    binary: np.ndarray,
    regions: List[Dict[str, Any]],
    orientation: str = "auto",
) -> List[Dict[str, Any]]:
    """Separate short, spaced vertical title columns from coarse body boxes.

    Require long body columns nearby and a larger-than-normal gutter on both
    sides (one side for an outer title). Paragraph endings share normal column
    spacing and remain body. Geometry alone cannot reliably classify a title
    that has the same length and spacing as prose; keep those conservative.
    """
    if orientation == "horizontal" or binary.size == 0:
        return regions

    image_h, image_w = binary.shape[:2]
    result = []
    used_ids = {r.get("id") for r in regions}
    heading_index = 0
    for reg in regions:
        if reg.get("type") != "body":
            result.append(reg)
            continue
        rx = max(0, min(image_w, int(reg.get("x", 0))))
        ry = max(0, min(image_h, int(reg.get("y", 0))))
        x_end = max(rx, min(image_w, int(reg.get("x", 0) + reg.get("w", 0))))
        y_end = max(ry, min(image_h, int(reg.get("y", 0) + reg.get("h", 0))))
        roi = binary[ry:y_end, rx:x_end]
        if roi.size == 0:
            result.append(reg)
            continue

        _, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        max_glyph_dim = max(48, int(min(image_w, image_h) * 0.08))
        glyphs = [
            (int(x + rx), int(y + ry), int(w), int(h))
            for x, y, w, h, area in stats[1:]
            if area >= 12 and w >= 8 and h >= 8
            and w <= max_glyph_dim and h <= max_glyph_dim
        ]
        if len(glyphs) < 30:
            result.append(reg)
            continue
        glyph_w = float(np.median([g[2] for g in glyphs]))
        glyph_h = float(np.median([g[3] for g in glyphs]))
        counts = np.zeros(x_end - rx, dtype=np.int32)
        for x, _, w, _ in glyphs:
            counts[x - rx:x - rx + w] += 1
        active = counts > 0
        tracks = []
        x = 0
        while x < len(active):
            if not active[x]:
                x += 1
                continue
            start = x
            while x < len(active) and active[x]:
                x += 1
            members = [g for g in glyphs if rx + start <= g[0] + g[2] / 2 < rx + x]
            if len(members) < 3:
                continue
            x0 = min(g[0] for g in members)
            y0 = min(g[1] for g in members)
            x1 = max(g[0] + g[2] for g in members)
            y1 = max(g[1] + g[3] for g in members)
            tracks.append((x0, y0, x1, y1))

        body_top = min((t[1] for t in tracks), default=ry)
        body_bottom = max((t[3] for t in tracks), default=y_end)
        body_height = body_bottom - body_top
        # Several long columns establish that this is vertical prose, even in
        # auto mode. Horizontal rows do not produce full-height narrow tracks.
        long_tracks = [t for t in tracks if t[3] - t[1] >= body_height * 0.60]
        if body_height < glyph_h * 12 or len(long_tracks) < 2:
            result.append(reg)
            continue
        if orientation == "auto" and not any(
            t[3] - t[1] > (t[2] - t[0]) * 3 for t in long_tracks
        ):
            result.append(reg)
            continue

        candidates = []
        for i, track in enumerate(tracks):
            x0, y0, x1, y1 = track
            if not (
                x1 - x0 <= glyph_w * 2.2
                and glyph_h * 2.5 <= y1 - y0 <= body_height * 0.55
                and body_top <= y0 <= body_top + glyph_h * 4
            ):
                continue
            left_gap = x0 - tracks[i - 1][2] if i else None
            right_gap = tracks[i + 1][0] - x1 if i + 1 < len(tracks) else None
            gaps = [gap for gap in (left_gap, right_gap) if gap is not None]
            minimum_gap = glyph_w * (1.25 if len(gaps) == 2 else 1.60)
            if not gaps or min(gaps) < minimum_gap:
                continue
            # Do not label a tiny isolated numeral or a short marginal mark.
            if y1 - y0 < glyph_h * 3:
                continue
            left_cut = int((tracks[i - 1][2] + x0) / 2) if i else rx
            right_cut = int((x1 + tracks[i + 1][0]) / 2) if i + 1 < len(tracks) else x_end
            candidates.append((track, left_cut, right_cut))

        if not candidates:
            result.append(reg)
            continue

        parts = []
        boundaries = [(rx, candidates[0][1])]
        boundaries.extend((a[2], b[1]) for a, b in zip(candidates, candidates[1:]))
        boundaries.append((candidates[-1][2], x_end))
        for part_index, (start, end) in enumerate(boundaries, 1):
            members = [g for g in glyphs if start <= g[0] + g[2] / 2 < end]
            if not members:
                continue
            # Preserve the full original height, including small punctuation
            # omitted by the glyph measurement filter.
            parts.append({
                **reg, "id": f"{reg.get('id', 'detect-body')}-part-{part_index}",
                "name": f"{reg.get('name', '本文候補')} ({part_index})",
                "x": start, "y": ry, "w": end - start, "h": y_end - ry,
                "text": "",
            })
        for (x0, y0, x1, y1), left_cut, right_cut in candidates:
            heading_index += 1
            heading_id = f"detect-heading-{heading_index}"
            while heading_id in used_ids:
                heading_index += 1
                heading_id = f"detect-heading-{heading_index}"
            used_ids.add(heading_id)
            hx = max(left_cut, x0 - 7)
            hy = max(ry, y0 - 8)
            hx1 = min(right_cut, x1 + 7)
            hy1 = min(y_end, y1 + 8)
            parts.append({
                "id": heading_id, "name": f"見出し候補 {heading_index}",
                "type": "heading", "orientation": "vertical",
                "x": hx, "y": hy, "w": hx1 - hx, "h": hy1 - hy,
                "text": "",
            })
        # Each incoming region already has page/band reading order. Only order
        # its subdivisions right-to-left; retain table and other page ordering.
        parts.sort(key=lambda r: -(r["x"] + r["w"] / 2))
        result.extend(parts)

    for order, reg in enumerate(result, 1):
        reg["reading_order"] = order
    return result
