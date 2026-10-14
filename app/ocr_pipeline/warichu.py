"""Exclude bracketed, paired small-print streams in a vertical body column.

Require enclosing brackets AND multiple streams of much smaller characters.
All thresholds use the printed column width. Ordinary-size bracketed notes,
one-column notes, ruby and short tate-chu-yoko remain recognition targets.
"""
import cv2
import numpy as np


def _runs(bits):
    edges = np.diff(np.r_[False, bits, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def exclude_vertical_warichu(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if min(gray.shape) < 100:
        return image, []
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    occupied = (ink != 0)
    projection = occupied.sum(axis=0)
    bands = list(_runs(projection >= max(8, gray.shape[0] * .025)))
    # A straight page of vertical body text provides its own scale and grid.
    bands = [(a, b) for a, b in bands if 8 <= b-a <= gray.shape[1] * .05]
    if len(bands) < 5:
        return image, []
    width = float(np.median([b-a for a, b in bands]))
    bands = [(a,b) for a,b in bands if .75*width <= b-a <= 1.4*width]
    if len(bands) < 5:
        return image, []
    regions = []
    for left, right in bands:
        # Recover faint outer pixels without entering the next body column.
        while left > 0 and projection[left-1] > 0:
            left -= 1
        while right < gray.shape[1] and projection[right] > 0:
            right += 1
        bw = right-left
        if not .75*width <= bw <= 1.65*width:
            continue
        _, labels, stats, _ = cv2.connectedComponentsWithStats(ink[:,left:right], 8)
        comps = stats[1:]
        # Require ordinary full-size characters in this same column.
        large = [r for r in comps if r[2] >= .80*width and r[3] >= .80*width]
        body_edges = [r for r in comps if r[2] >= .80*width or r[3] >= .75*width]
        # In vertical print, an enclosing bracket is a wide, shallow cap.
        # Require matching caps immediately above AND below the small streams;
        # brackets alone never qualify ordinary one-column notes for removal.
        caps = [r for r in comps if r[2] >= .85*width and r[3] <= .50*width
                and r[2] >= r[3]*2.5]
        if len(large) < 5:
            continue
        candidates = []
        for seam in range(left+int(bw*.35), left+int(bw*.65)+1):
            # A tiny scan bridge or a small bracket may cross the gap between
            # the two streams. Ignore only crossings whose entire component
            # is small; full-size body strokes still stop the candidate span.
            seam_labels = labels[:,seam-left]
            clear_seam = ~occupied[:,seam].copy()
            for a,b in _runs(occupied[:,seam]):
                if b-a > max(1,round(width*.15)):
                    continue
                crossing = stats[np.unique(seam_labels[a:b])]
                if all(r[3] <= width*.70 and (r[2] < width*.8 or r[2] <= r[3]*2.5) for r in crossing):
                    clear_seam[a:b] = True
            for top, bottom in _runs(clear_seam):
                if bottom-top < width*4:
                    continue
                area = occupied[top:bottom, left:right]
                split = seam-left
                if min(area[:,:split].sum(), area[:,split+1:].sum()) < (bottom-top)*width*.05:
                    continue
                small_left, small_right = [], []
                invalid = False
                for r in comps:
                    x,y,w,h,pixels = map(int,r)
                    if y < top or y+h > bottom or pixels < 5:
                        continue
                    if w >= width*.7 and h > width*.70:
                        invalid = True
                        break
                    if not (width*.20 <= h <= width*.70 and w <= width*.70):
                        continue
                    if x+w <= split:
                        small_left.append((y,y+h))
                    elif x > split:
                        small_right.append((y,y+h))
                if invalid or min(len(small_left),len(small_right)) < 6:
                    continue
                starts = [min(a for a,b in side) for side in (small_left,small_right)]
                ends = [max(b for a,b in side) for side in (small_left,small_right)]
                if abs(starts[0]-starts[1]) > width or abs(ends[0]-ends[1]) > width:
                    continue
                y1,y2 = min(starts),max(ends)
                above = [r for r in caps if 0 <= y1-(r[1]+r[3]) <= width*.8]
                below = [r for r in caps if 0 <= r[1]-y2 <= width*.8]
                if not any(abs((a[0]+a[2]/2)-(b[0]+b[2]/2)) <= width*.2
                           and min(a[2],b[2]) >= max(a[2],b[2])*.8
                           for a in above for b in below):
                    continue
                # Inline notes have full-size body characters on both sides.
                # Do not suppress independent small-print columns or endnotes.
                if not any(0 <= y1-(r[1]+r[3]) <= width*3 for r in body_edges):
                    continue
                if not any(0 <= r[1]-y2 <= width*3 for r in body_edges):
                    continue
                if y2-y1 >= width*4:
                    candidates.append((y1,y2))
        for top,bottom in sorted(candidates):
            if regions and regions[-1][0] == left and top <= regions[-1][3]:
                x1,y1,x2,y2 = regions[-1]
                regions[-1] = (x1,min(y1,top),x2,max(y2,bottom))
            else:
                regions.append(tuple(map(int,(left,top,right,bottom))))
    if not regions:
        return image, []
    result = image.copy()
    for x1,y1,x2,y2 in regions:
        result[y1:y2,x1:x2] = 255
    return result, regions


