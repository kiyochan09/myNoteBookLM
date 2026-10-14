"""Separate vertical headings from prose using spacing and local print style."""
from typing import Any, Dict, List
import cv2
import numpy as np


def _vertical_tracks(binary, region):
    ih, iw = binary.shape[:2]
    rx, ry = max(0, int(region.get("x", 0))), max(0, int(region.get("y", 0)))
    end_x = min(iw, int(region.get("x", 0) + region.get("w", 0)))
    end_y = min(ih, int(region.get("y", 0) + region.get("h", 0)))
    roi = binary[ry:end_y, rx:end_x]
    if not roi.size:
        return []
    _, _, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
    limit = max(48, int(min(iw, ih) * .08))
    glyphs = [(int(x+rx), int(y+ry), int(w), int(h)) for x, y, w, h, area in stats[1:]
              if area >= 12 and min(w, h) >= 8 and max(w, h) <= limit]
    counts = np.zeros(end_x-rx, np.int32)
    for x, _, w, _ in glyphs:
        counts[x-rx:x-rx+w] += 1
    xs = np.flatnonzero(counts)
    groups = np.split(xs, np.flatnonzero(np.diff(xs)>1)+1) if len(xs) else []
    tracks = []
    for group in groups:
        members = [g for g in glyphs if rx+group[0] <= g[0]+g[2]/2 < rx+group[-1]+1]
        if len(members) < 3:
            continue
        x0, y0 = min(g[0] for g in members), min(g[1] for g in members)
        x1, y1 = max(g[0]+g[2] for g in members), max(g[1]+g[3] for g in members)
        patch = (binary[y0:y1, x0:x1]>0).astype(np.uint8)
        distance = cv2.distanceTransform(patch, cv2.DIST_L2, 3)
        tracks.append({"box": (x0, y0, x1, y1), "glyph_w": float(np.median([g[2] for g in members])),
                       "glyph_h": float(np.median([g[3] for g in members])),
                       "stroke": float(np.mean(distance[distance>0])), "count": len(members)})
    return tracks


def split_vertical_heading_regions(binary: np.ndarray, regions: List[Dict[str, Any]], orientation: str = "auto") -> List[Dict[str, Any]]:
    if orientation == "horizontal" or binary.size == 0:
        return regions
    ih, iw = binary.shape[:2]
    measured = [_vertical_tracks(binary, r) if r.get("type")=="body" else [] for r in regions]
    all_tracks = [t for group in measured for t in group]
    if not all_tracks:
        return regions
    max_height = max(t["box"][3]-t["box"][1] for t in all_tracks)
    references = [t for t in all_tracks if t["box"][3]-t["box"][1] >= max_height*.60
                  and (orientation == "vertical" or t["box"][3]-t["box"][1] > (t["box"][2]-t["box"][0])*3)]
    if len(references) < 2:
        return regions
    result, used_ids = [], {r.get("id") for r in regions}
    heading_index = 0
    for region, tracks in zip(regions, measured):
        if not tracks:
            result.append(region)
            continue
        rx, ry, rw, rh = [int(region.get(k, 0)) for k in ("x", "y", "w", "h")]
        candidates = []
        for track in tracks:
            x0, y0, x1, y1 = track["box"]
            nearby = [t for t in references if abs((t["box"][0]+t["box"][2]-x0-x1)/2) <= max(200, iw*.25)
                      and t["box"][1] < y1 and t["box"][3] > y0]
            if len(nearby) < 2:
                nearby = sorted((t for t in references if t["box"][1] < y1 and t["box"][3] > y0),
                    key=lambda t: abs(t["box"][0]+t["box"][2]-x0-x1))[:4]
                if len(nearby) < 2:
                    continue
            glyph_w = float(np.median([t["glyph_w"] for t in nearby]))
            glyph_h = float(np.median([t["glyph_h"] for t in nearby]))
            body_top = float(np.median([t["box"][1] for t in nearby]))
            body_height = float(np.median([t["box"][3]-t["box"][1] for t in nearby]))
            stroke = float(np.median([t["stroke"] for t in nearby]))
            width = float(np.median([t["box"][2]-t["box"][0] for t in nearby]))
            if not (x1-x0 <= glyph_w*2.2 and glyph_h*3 <= y1-y0 <= body_height*.55
                    and body_top-glyph_h*.5 <= y0 <= body_top+glyph_h*4):
                continue
            # Neighboring regions supply context too: coarse layout detection
            # may have split a large gutter before this heading refinement.
            neighbors = [t["box"] for t in all_tracks if t is not track and t["box"][1] < y1 and t["box"][3] > y0]
            left = max((b[2] for b in neighbors if b[2] <= x0), default=None)
            right = min((b[0] for b in neighbors if b[0] >= x1), default=None)
            gaps = [gap for gap in (x0-left if left is not None else None, right-x1 if right is not None else None) if gap is not None]
            if not gaps:
                continue
            indent = y0-body_top
            size_ratio = track["glyph_w"]/max(1, glyph_w)
            height_ratio = track["glyph_h"]/max(1, glyph_h)
            width_ratio = (x1-x0)/max(1, width)
            stroke_ratio = track["stroke"]/max(.01, stroke)
            distinctive = (
                (size_ratio >= 1.12 and height_ratio >= 1.07 and stroke_ratio >= 1.015)
                or (stroke_ratio >= 1.07 and width_ratio >= 1.03)
                or (size_ratio >= 1.12 and width_ratio >= 1.08 and stroke_ratio >= 1.025)
                or (width_ratio >= 1.08 and height_ratio >= 1.03 and stroke_ratio >= 1.045)
                or (stroke_ratio >= 1.04 and max(gaps) >= glyph_w*2 and x1-x0 >= glyph_w*.95)
                or (size_ratio >= 1.12 and height_ratio >= 1.03 and stroke_ratio >= 1.04)
                or (height_ratio >= 1.07 and stroke_ratio >= 1.035 and x1-x0 >= glyph_w*1.05)
            )
            styled = distinctive and indent >= glyph_h*.65 and min(gaps) >= glyph_w*.40
            isolated = min(gaps) >= glyph_w*(1.25 if len(gaps)==2 else 1.60)
            # Keep isolated short columns eligible even when their print style
            # matches prose: favor heading recall over rare ending false positives.
            if not (styled or isolated):
                continue
            left_cut = max(rx, int((left+x0)/2)) if left is not None else rx
            right_cut = min(rx+rw, int((x1+right)/2)) if right is not None else rx+rw
            # Include thin punctuation/long vowel marks rejected by the glyph
            # size measurement, stopping at a genuine blank paragraph gap.
            ink = binary[max(0, ry):min(ih, ry+rh), max(left_cut, x0-3):min(right_cut, x1+3)]
            ink_rows = np.flatnonzero(np.any(ink>0, axis=1))
            runs = np.split(ink_rows, np.flatnonzero(np.diff(ink_rows)>glyph_h*1.8)+1) if len(ink_rows) else []
            matching = [run for run in runs if ry+run[0] <= y0 and ry+run[-1]+1 >= y1]
            if matching:
                y0, y1 = ry+int(matching[0][0]), ry+int(matching[0][-1])+1
            candidates.append(((x0,y0,x1,y1),left_cut,right_cut))
        if not candidates:
            result.append(region)
            continue
        candidates.sort(key=lambda c:c[1])
        parts = []
        intervals = [(rx,candidates[0][1])] + [(a[2],b[1]) for a,b in zip(candidates,candidates[1:])] + [(candidates[-1][2],rx+rw)]
        for index, (start,end) in enumerate(intervals,1):
            if end <= start or not np.any(binary[max(0,ry):min(ih,ry+rh), max(0,start):min(iw,end)]):
                continue
            _, _, support, _ = cv2.connectedComponentsWithStats(binary[max(0,ry):min(ih,ry+rh), max(0,start):min(iw,end)], connectivity=8)
            if not any(a>=12 and w>=8 and h>=8 for _,_,w,h,a in support[1:]):
                continue
            identifier = f"{region.get('id','detect-body')}-part-{index}"
            while identifier in used_ids:
                identifier += "-part"
            used_ids.add(identifier)
            parts.append({**region,"id":identifier,"name":f"{region.get('name','本文候補')} ({index})",
                          "x":start,"y":ry,"w":end-start,"h":rh,"text":""})
        for (x0,y0,x1,y1),left_cut,right_cut in candidates:
            heading_index += 1
            identifier = f"detect-heading-{heading_index}"
            while identifier in used_ids:
                heading_index += 1
                identifier = f"detect-heading-{heading_index}"
            used_ids.add(identifier)
            hx,hy = max(left_cut,x0-7),max(ry,y0-8)
            hx1,hy1 = min(right_cut,x1+7),min(ry+rh,y1+8)
            parts.append({"id":identifier,"name":f"見出し候補 {heading_index}","type":"heading","orientation":"vertical",
                          "heading_level":2,"x":hx,"y":hy,"w":hx1-hx,"h":hy1-hy,"text":""})
        parts.sort(key=lambda r:-(r["x"]+r["w"]/2))
        result.extend(parts)
    for order, region in enumerate(result,1):
        region["reading_order"] = order
    return result
