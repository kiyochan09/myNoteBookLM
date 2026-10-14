import sys
sys.setrecursionlimit(5000)
import os

# Windows環境におけるUTF-8標準入出力の強制設定（CP932/Shift-JIS文字化け・UnicodeEncodeError防止）
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import re
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
from pathlib import Path
from deim import DEIM
from parseq import PARSEQ

from yaml import safe_load
from concurrent.futures import ThreadPoolExecutor
import time
import shutil
import json
import glob
from reading_order.xy_cut.eval import eval_xml
from ndl_parser import convert_to_xml_string3

class RecogLine:
    def __init__(self,npimg:np.ndarray,idx:int,pred_char_cnt:int,pred_str:str="",crop_context=None):
        self.npimg = npimg
        self.idx   = idx
        self.pred_char_cnt = pred_char_cnt
        self.pred_str = pred_str
        self.crop_context = crop_context
    def shifted_x_crop(self, offset:int):
        if self.crop_context is None:
            return None
        source, y1, y2, x1, x2 = self.crop_context
        shifted_x1 = x1 + offset
        shifted_x2 = x2 + offset
        if shifted_x1 < 0 or shifted_x2 > source.shape[1]:
            return None
        return source[y1:y2, shifted_x1:shifted_x2, :]
    def __lt__(self, other):
        return self.idx < other.idx

def find_word_boundary_split(baseimg, is_horiz=True):
    h, w = baseimg.shape[:2]
    if is_horiz:
        mid = w // 2
        gray = cv2.cvtColor(baseimg, cv2.COLOR_BGR2GRAY) if len(baseimg.shape) == 3 else baseimg
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = np.sum(binary, axis=0)
        
        search_start = max(50, int(mid - 0.20 * w))
        search_end = min(w - 50, int(mid + 0.20 * w))
        
        threshold_ink = max(50.0, np.max(ink) * 0.08)
        low_ink = (ink <= threshold_ink).astype(np.uint8)
        
        runs = []
        in_run = False
        r_start = 0
        for x in range(search_start, search_end):
            if low_ink[x] and not in_run:
                in_run = True
                r_start = x
            elif not low_ink[x] and in_run:
                in_run = False
                runs.append((r_start, x, x - r_start))
        if in_run:
            runs.append((r_start, search_end, search_end - r_start))
            
        if runs:
            best_run = max(runs, key=lambda r: (min(r[2], 25) * 10 - abs((r[0]+r[1])/2.0 - mid)*0.1))
            split_pos = (best_run[0] + best_run[1]) // 2
        else:
            split_pos = search_start + int(np.argmin(ink[search_start:search_end]))
            
        return split_pos
    else:
        mid = h // 2
        gray = cv2.cvtColor(baseimg, cv2.COLOR_BGR2GRAY) if len(baseimg.shape) == 3 else baseimg
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = np.sum(binary, axis=1)
        
        search_start = max(50, int(mid - 0.20 * h))
        search_end = min(h - 50, int(mid + 0.20 * h))
        
        threshold_ink = max(50.0, np.max(ink) * 0.08)
        low_ink = (ink <= threshold_ink).astype(np.uint8)
        
        runs = []
        in_run = False
        r_start = 0
        for y in range(search_start, search_end):
            if low_ink[y] and not in_run:
                in_run = True
                r_start = y
            elif not low_ink[y] and in_run:
                in_run = False
                runs.append((r_start, y, y - r_start))
        if in_run:
            runs.append((r_start, search_end, search_end - r_start))
            
        if runs:
            best_run = max(runs, key=lambda r: (min(r[2], 25) * 10 - abs((r[0]+r[1])/2.0 - mid)*0.1))
            split_pos = (best_run[0] + best_run[1]) // 2
        else:
            split_pos = search_start + int(np.argmin(ink[search_start:search_end]))
            
        return split_pos

def is_mostly_latin(s: str) -> bool:
    if not s:
        return False
    latin_cnt = sum(1 for c in s if ord(c) < 256 and c.isalnum())
    total_cnt = sum(1 for c in s if c.isalnum())
    return (latin_cnt / total_cnt >= 0.70) if total_cnt > 0 else False

def _preserve_numeric_repeat(match):
    """Numeric values can legitimately repeat; collapse only nonnumeric runs."""
    return match.group(0) if any(char.isnumeric() for char in match.group(1)) else match.group(1)


def clean_runaway_repetition(text: str) -> str:
    if not text:
        return text
    # 数字・漢数字を保持し、数値以外の同一文字の反復を縮約
    text = re.sub(r'([^\s\d])\1{2,}', _preserve_numeric_repeat, text)
    # 数値の連続を保持し、記号の暴走出力を除去
    for repeated in re.finditer(r'([\d\.\-_=])\1{4,}', text):
        if not repeated.group(1).isnumeric():
            text = text[:repeated.start()]
            break
    text = text.strip()
    # 繰り返し単語・フレーズの暴走（the the the the, そう言うようもないと言うように 等）を除去
    text = re.sub(r'(\b\w+\b\s+)\1{2,}.*$', lambda match: match.group(0) if any(char.isnumeric() for char in match.group(1)) else '', text).strip()
    text = re.sub(r'((?:[^\s]{2,8}))\1{2,}', _preserve_numeric_repeat, text)
    # 年号直後の括弧誤認識 (例: 2005al. -> 2005a]., 1997l. -> 1997].)
    text = re.sub(r'(\b\d{4}[a-z]?)l\.', r'\1].', text)
    text = re.sub(r'(\b\d{4}[a-z]?)1\.', r'\1].', text)
    return text

def is_symbol_runaway(text: str) -> bool:
    """Flag dense bracket/ichi/hyphen loops while leaving ordinary parenthetical prose alone."""
    compact = re.sub(r"\s+", "", text or "")
    if len(compact) < 14:
        return False
    for match in re.finditer(r"(?:[()（）][一ー\-]?){5,}", compact):
        if sum(ch in "一ー-" for ch in match.group(0)) >= 2:
            return True
    return False


def retry_symbol_runaway(lineobj, recognizer, text: str) -> str:
    """Retry suspicious vertical-line output with a one-pixel horizontal crop shift."""
    if not text or not is_symbol_runaway(text) or lineobj.npimg.shape[0] <= lineobj.npimg.shape[1]:
        return text
    for offset in (1, -1):
        candidate_crop = lineobj.shifted_x_crop(offset)
        if candidate_crop is None or candidate_crop.size == 0:
            continue
        try:
            candidate = clean_runaway_repetition(recognizer.read(candidate_crop))
        except Exception:
            continue
        if candidate and not is_symbol_runaway(candidate):
            return candidate
    return text

def retry_adjacent_duplicate(lineobj, neighbor, text: str) -> str:
    """Retry a long vertical line that exactly duplicates its adjacent column."""
    if not text or not neighbor.pred_str or lineobj.crop_context is None or neighbor.crop_context is None:
        return text
    compact = re.sub(r"\s+", "", text)
    neighbor_compact = re.sub(r"\s+", "", neighbor.pred_str)
    if len(compact) < 24 or compact != neighbor_compact:
        return text
    source, y1, y2, x1, x2 = lineobj.crop_context
    other_source, oy1, oy2, ox1, ox2 = neighbor.crop_context
    if source is not other_source or min(y2-y1, oy2-oy1) < 350:
        return text
    x_gap = abs((x1+x2)/2 - (ox1+ox2)/2)
    if x_gap < min(x2-x1, ox2-ox1)*0.7 or x_gap > max(x2-x1, ox2-ox1)*4:
        return text
    overlap = max(0, min(y2,oy2)-max(y1,oy1))
    if overlap / max(1,min(y2-y1,oy2-oy1)) < 0.65:
        return text
    recognizer = getattr(lineobj,"recognizer",None)
    if recognizer is None:
        return text
    best = None
    best_score = -1.0
    for offset in (1,-1):
        candidate_crop = lineobj.shifted_x_crop(offset)
        if candidate_crop is None or candidate_crop.size == 0:
            continue
        try:
            candidate = clean_runaway_repetition(recognizer.read(candidate_crop))
        except Exception:
            continue
        candidate_compact = re.sub(r"\s+", "", candidate or "")
        if len(candidate_compact) < max(20,int(len(compact)*0.65)) or candidate_compact == neighbor_compact or is_symbol_runaway(candidate):
            continue
        japanese_count = sum(1 for ch in candidate_compact if re.search(r"[ぁ-んァ-ヶ一-龯々]",ch))
        if japanese_count / max(1,len(candidate_compact)) < 0.60:
            continue
        score = japanese_count / len(candidate_compact) + japanese_count * 0.001
        if score > best_score:
            best, best_score = candidate, score
    return best if best is not None else text
def trim_line_box(img, x1, y1, x2, y2, is_horiz):
    h, w = img.shape[:2]
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return x1, y1, x2, y2
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if is_horiz:
        ink = np.sum(binary, axis=0)
        ink_cols = np.where(ink > 0)[0]
        if len(ink_cols) == 0:
            return x1, y1, x2, y2
        line_h = y2 - y1
        first_ink = ink_cols[0]
        last_ink = ink_cols[-1]
        new_x1 = x1
        new_x2 = x2
        if first_ink > max(35, int(line_h * 1.2)):
            new_x1 = x1 + first_ink - 4
        if (crop.shape[1] - 1 - last_ink) > max(35, int(line_h * 1.2)):
            new_x2 = x1 + last_ink + 4
        return max(0, new_x1), y1, min(w, new_x2), y2
    else:
        ink = np.sum(binary, axis=1)
        ink_rows = np.where(ink > 0)[0]
        if len(ink_rows) == 0:
            return x1, y1, x2, y2
        line_w = x2 - x1
        first_ink = ink_rows[0]
        last_ink = ink_rows[-1]
        new_y1 = y1
        new_y2 = y2
        if first_ink > max(35, int(line_w * 1.2)):
            new_y1 = y1 + first_ink - 4
        if (crop.shape[0] - 1 - last_ink) > max(35, int(line_w * 1.2)):
            new_y2 = y1 + last_ink + 4
        return x1, max(0, new_y1), x2, min(h, new_y2)

def vertical_tail_limit(root, detections):
    """Select the last three long vertical columns in right-to-left reading order."""
    boxes = [(float(line.get('X')), float(line.get('Y')), float(line.get('WIDTH')), float(line.get('HEIGHT')))
             for line in root.findall('.//LINE')]
    if not boxes:
        boxes = [(d['box'][0], d['box'][1], d['box'][2]-d['box'][0], d['box'][3]-d['box'][1])
                 for d in detections]
    columns = sorted((x + w / 2, w) for x, y, w, h in boxes if h >= 350 and h > w * 6 and w >= 8)
    if not columns:
        return -1
    center, width = columns[min(2, len(columns)-1)]
    return center + width / 2


def vertical_recognition_padding(img, x, y, width, height, tail_limit=float("inf")):
    """Give long vertical reads a stable blank margin without adding adjacent ink."""
    if x + width / 2 > tail_limit or height < 350 or height <= width * 6 or width < 8:
        return 2
    x, y, width, height = map(int, (x, y, width, height))
    ih, iw = img.shape[:2]
    top, bottom = max(0, y - 8), min(ih, y + height + 8)
    for left, right in ((x - 4, x - 2), (x + width + 2, x + width + 4)):
        left, right = max(0, min(iw, left)), max(0, min(iw, right))
        if right <= left:
            continue
        strip = img[top:bottom, left:right]
        if strip.size == 0:
            continue
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
        if np.count_nonzero(gray < 180) > max(4, height * 0.02):
            return 2
    return 4


def tail_consensus_key(text):
    """Compare width and dash typography without changing the accepted text."""
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-"}))
    return re.sub(r'\s+', '', text)


def retry_tail_consensus(lineobj, recognizer, text):
    """Accept a tail reading only when full, shifted and split crops agree."""
    shifted = lineobj.shifted_x_crop(1)
    if shifted is None:
        return text
    candidate = clean_runaway_repetition(recognizer.read(lineobj.npimg))
    candidate_key = tail_consensus_key(candidate)
    if len(candidate_key) < 20 or candidate_key == tail_consensus_key(text):
        return text
    confirmation = clean_runaway_repetition(recognizer.read(shifted))
    if candidate_key == tail_consensus_key(confirmation):
        split = find_word_boundary_split(lineobj.npimg, is_horiz=False)
        upper = clean_runaway_repetition(recognizer.read(lineobj.npimg[:split]))
        lower = clean_runaway_repetition(recognizer.read(lineobj.npimg[split:]))
        if tail_consensus_key(upper + lower) == candidate_key:
            return candidate
    return text


_tcy_image_matcher = None


def tcy_text_positions(text):
    value, positions = [], []
    for index, char in enumerate(text):
        key = tail_consensus_key(char).replace('〈', '<').replace('〉', '>')
        value.extend(key)
        positions.extend([index] * len(key))
    return ''.join(value), positions


def tcy_key(text):
    return tcy_text_positions(text)[0]


def read_vertical_piece(recognizer, image):
    if image is None or not image.size:
        return ''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(binary)
    if points is None:
        return ''
    # Retain the original crop for long pieces. For a short piece, explicitly
    # preserve vertical reading direction rather than guessing from white margins.
    if image.shape[0] > image.shape[1]:
        return recognizer.read(image)
    x, y, width, height = cv2.boundingRect(points)
    rotated = cv2.rotate(image[y:y + height, x:x + width], cv2.ROTATE_90_COUNTERCLOCKWISE)
    extra = max(0, rotated.shape[0] - rotated.shape[1])
    rotated = cv2.copyMakeBorder(rotated, 1, 1, 1 + extra, 1,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))
    result = recognizer.read(rotated)
    # A single short stroke can decode as two dashes when enlarged. Verify the
    # one-component image before collapsing that recognition duplicate.
    if (re.fullmatch(r'-{2,}', tail_consensus_key(result)) and width <= 3
            and height >= max(6, width * 3)
            and cv2.connectedComponents(binary[y:y + height, x:x + width])[0] == 2):
        return result[:1]
    return result


def place_tcy_pixels(text, head50, head100, tail50, tail100, item, image_height, image_width):
    """Locate a pixel-confirmed number using nearby matching text, without meaning rules."""
    body, positions = tcy_text_positions(text)
    head, other_head = tcy_key(head50), tcy_key(head100)
    tail, other_tail = tcy_key(tail50), tcy_key(tail100)
    if tail != other_tail:
        return None
    choices = []
    for cut in range(min(8, len(head)) + 1):
        prefix = head[:len(head) - cut] if cut else head
        added = head[len(prefix):]
        if added and not other_head.endswith(added):
            continue
        for size in range(min(12, len(prefix)), 3, -1):
            anchor = prefix[-size:]
            starts = [m.start() for m in re.finditer(re.escape(anchor), body)]
            if len(starts) != 1:
                continue
            left = starts[0] + size
            if abs(left - item['y1'] / image_height * len(body)) <= 12:
                choices.append((size, -cut, left, added))
                break
    if choices:
        _, _, left, added = max(choices)
    elif len(head) <= 3 and head == other_head and item['y1'] <= image_width * 2:
        left, added = 0, head
    else:
        return None
    targets = []
    for cut in range(min(8, max(0, len(tail) - 4)) + 1):
        anchor = tail[cut:cut + 10]
        if len(anchor) < 4:
            continue
        starts = [m.start() for m in re.finditer(re.escape(anchor), body)]
        if len(starts) == 1:
            right = starts[0]
            if 0 <= right - left <= len(added) + len(item['digits']) + cut + 8:
                targets.append((len(anchor), -cut, right, tail[:cut]))
    if targets:
        _, _, right, trailing = max(targets)
    elif len(tail) <= 3 and image_height - item['y2'] <= image_width * 2:
        right, trailing = len(body), tail
        if right - left > len(added) + len(item['digits']) + len(trailing) + 8:
            return None
    else:
        return None
    if (left == 0 and not choices and re.search(r'[一-龥ぁ-んァ-ヶ]', body[:right])
            and not (head and body.startswith(head))):
        return None
    start = positions[left - 1] + 1 if left else 0
    end = positions[right] if right < len(positions) else len(text)
    replacement = added + item['digits'] + trailing
    if body[left:right] == replacement:
        return text
    return text[:start] + replacement + text[end:]


def recover_tcy_images(lines, recognizer50, recognizer100):
    global _tcy_image_matcher
    from tcy_image_matcher import ImageDigitMatcher
    if _tcy_image_matcher is None:
        _tcy_image_matcher = ImageDigitMatcher(Path.cwd() / 'data' / 'tcy_templates')
    for line in lines:
        if line.crop_context is None or line.npimg is None or not line.pred_str:
            continue
        image = line.npimg
        for item in _tcy_image_matcher.candidates(image):
            a, b, x, width = (item[k] for k in ('y1', 'y2', 'x', 'width'))
            patch = image[a:b, x:x + width]
            confirmations = []
            for pad in (0, 1):
                extra = max(0, patch.shape[0] - patch.shape[1])
                horizontal = cv2.copyMakeBorder(patch, pad, pad, pad + (extra + 1) // 2,
                                               pad + extra // 2, cv2.BORDER_CONSTANT,
                                               value=(255, 255, 255))
                confirmations.extend([recognizer50.read(horizontal), recognizer100.read(horizontal)])
            if any(value != item['digits'] for value in confirmations):
                continue
            # The threshold-defined digit box can leave faint edge pixels
            # in the adjacent crop. Split the empty inter-glyph gap at its
            # midpoint so re-reading context cannot see a fragment of the digit.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            rows = (ink > 0).any(axis=1)
            preceding = np.flatnonzero(rows[:a])
            following = np.flatnonzero(rows[b:])
            head_end = (int(preceding[-1]) + 1 + a) // 2 if preceding.size else a
            tail_start = b + int(following[0]) // 2 if following.size else b
            head50 = read_vertical_piece(recognizer50, image[:head_end])
            head100 = read_vertical_piece(recognizer100, image[:head_end])
            tail50 = read_vertical_piece(recognizer50, image[tail_start:])
            tail100 = read_vertical_piece(recognizer100, image[tail_start:])
            corrected = place_tcy_pixels(line.pred_str, head50, head100, tail50, tail100,
                                         item, image.shape[0], image.shape[1])
            if corrected is None:
                continue
            line.pred_str = corrected
            if not hasattr(line, 'tcy_image_evidence'):
                line.tcy_image_evidence = []
            line.tcy_image_evidence.append(item)
def process_cascade(alllineobj:RecogLine,recognizer30,recognizer50,recognizer100,is_cascade=True):
    targetdflist30,targetdflist50,targetdflist100,targetdflist200=[],[],[],[]
    for lineobj in alllineobj:
        img_h, img_w = lineobj.npimg.shape[:2] if lineobj.npimg is not None else (0, 0)
        is_horiz = (img_w >= img_h)
        dim_main = img_w if is_horiz else img_h
        
        # 縦書き・横書きともに長行（350px以上 / 15文字以上）は直接最上位PARSEQ-100へ送り文字脱落を防止
        if dim_main >= 350 or dim_main >= (img_h if is_horiz else img_w) * 6 or not is_cascade:
            targetdflist100.append(lineobj)
        elif lineobj.pred_char_cnt == 3 and is_cascade and dim_main < 250:
            targetdflist30.append(lineobj)
        elif lineobj.pred_char_cnt == 2 and is_cascade and dim_main < 350:
            targetdflist50.append(lineobj)
        else:
            targetdflist100.append(lineobj)
            
    tail_candidates = [line for line in alllineobj
                       if line.crop_context is not None and line.npimg.shape[0] >= 350
                       and line.npimg.shape[0] > line.npimg.shape[1] * 6]
    tail_candidates.sort(key=lambda line: (line.crop_context[3] + line.crop_context[4]) / 2)
    tail_ids = {id(line) for line in tail_candidates[:3]}
    targetdflistall=[]
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="thread") as executor:
        resultlines30,resultlines50,resultlines100,resultlines200=[],[],[],[]
        if len(targetdflist30)>0:
            resultlines30 = executor.map(recognizer30.read, [t.npimg for t in targetdflist30])
            resultlines30 = list(resultlines30)
            for i in range(len(targetdflist30)):
                lineobj=targetdflist30[i]
                pred_str=retry_symbol_runaway(lineobj, recognizer30, clean_runaway_repetition(resultlines30[i]))
                if len(pred_str)>=20 or is_mostly_latin(pred_str):
                    targetdflist50.append(lineobj)
                else:
                    lineobj.pred_str=pred_str
                    lineobj.recognizer=recognizer30
                    targetdflistall.append(lineobj)
                    
        if len(targetdflist50)>0:
            resultlines50 = executor.map(recognizer50.read, [t.npimg for t in targetdflist50])
            resultlines50 = list(resultlines50)
            for i in range(len(targetdflist50)):
                lineobj=targetdflist50[i]
                pred_str=retry_symbol_runaway(lineobj, recognizer50, clean_runaway_repetition(resultlines50[i]))
                if len(pred_str)>=38 or is_mostly_latin(pred_str):
                    targetdflist100.append(lineobj)
                else:
                    lineobj.pred_str=pred_str
                    lineobj.recognizer=recognizer50
                    targetdflistall.append(lineobj)
                    
        if len(targetdflist100)>0:
            resultlines100 = executor.map(recognizer100.read, [t.npimg for t in targetdflist100])
            resultlines100 = list(resultlines100)
            for i in range(len(targetdflist100)):
                lineobj=targetdflist100[i]
                pred_str=retry_symbol_runaway(lineobj, recognizer100, clean_runaway_repetition(resultlines100[i]))
                if id(lineobj) in tail_ids:
                    pred_str=retry_tail_consensus(lineobj, recognizer50, pred_str)
                lineobj.pred_str=pred_str
                lineobj.recognizer=recognizer100
                targetdflistall.append(lineobj)
                    
        targetdflistall=sorted(targetdflistall)
        for i in range(1,len(targetdflistall)):
            current=targetdflistall[i]
            previous=targetdflistall[i-1]
            if current.npimg.shape[0] > current.npimg.shape[1] and previous.npimg.shape[0] > previous.npimg.shape[1]:
                current.pred_str=retry_adjacent_duplicate(current,previous,current.pred_str)
        recover_tcy_images(targetdflistall, recognizer50, recognizer100)
        resultlinesall=[t.pred_str for t in targetdflistall]
    return resultlinesall


def find_resource_file(file_path_str, folder_name="model"):
    if not file_path_str:
        return file_path_str
    p = Path(file_path_str)
    if p.is_file():
        return str(p)
    fname = p.name
    cur_dir = Path(__file__).resolve().parent
    candidates = [cur_dir / folder_name / fname]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    return file_path_str

def get_detector(args):
    weights_path = find_resource_file(args.det_weights, "model")
    classes_path = find_resource_file(args.det_classes, "config")
    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {classes_path}"
    detector = DEIM(model_path=weights_path,
                      class_mapping_path=classes_path,
                      score_threshold=args.det_score_threshold,
                      conf_threshold=args.det_conf_threshold,
                      iou_threshold=args.det_iou_threshold,
                      device=args.device)
    return detector

def get_recognizer(args,weights_path=None):
    if weights_path is None:
        weights_path = args.rec_weights
    weights_path = find_resource_file(weights_path, "model")
    classes_path = find_resource_file(args.rec_classes, "config")

    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {classes_path}"

    charobj=None
    with open(classes_path,encoding="utf-8") as f:
        charobj=safe_load(f)
    charlist=list(charobj["model"]["charset_train"])
    
    recognizer = PARSEQ(model_path=weights_path,charlist=charlist,device=args.device)
    if getattr(args, 'enable_tcy', False):
        from tcy_wrapper import TateChuYokoWrapper
        tcy_kwargs = {k: v for k, v in vars(args).items() if k.startswith('tcy_') and k != 'enable_tcy' and v is not None}
        recognizer = TateChuYokoWrapper(recognizer, **tcy_kwargs)
    return recognizer

class DeferredRecognizer:
    def __init__(self, args, weights_path):
        self.args = args
        self.weights_path = weights_path
        self._recognizer = None
        from threading import Lock
        self._lock = Lock()

    def read(self, image):
        if self._recognizer is None:
            with self._lock:
                if self._recognizer is None:
                    self._recognizer = get_recognizer(self.args, self.weights_path)
        return self._recognizer.read(image)


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area <= 0:
        return 0.0
    
    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    
    ios = inter_area / min(area1, area2)
    iou = inter_area / union_area
    return max(iou, ios * 0.85)

LINE_CLASS_INDICES = {1, 2, 3, 4, 5, 16}

def merge_detections_nms(detections, iou_threshold=0.5):
    sorted_dets = sorted(detections, key=lambda x: (x.get('confidence', 0.5), (x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1])), reverse=True)
    kept = []
    
    for det in sorted_dets:
        box = list(det['box'])
        c_idx = det['class_index']
        w = max(1, box[2] - box[0])
        h = max(1, box[3] - box[1])
        area = w * h
        is_horizontal = (w >= h)
        is_line = (c_idx in LINE_CLASS_INDICES)
        
        merged_into_existing = False
        duplicate = False
        total_overlap = 0.0
        
        for k in kept:
            k_box = k['box']
            k_c_idx = k['class_index']
            k_w = max(1, k_box[2] - k_box[0])
            k_h = max(1, k_box[3] - k_box[1])
            k_is_line = (k_c_idx in LINE_CLASS_INDICES)
            
            if (c_idx == k_c_idx or (is_line and k_is_line)):
                inter_x1 = max(box[0], k_box[0])
                inter_y1 = max(box[1], k_box[1])
                inter_x2 = min(box[2], k_box[2])
                inter_y2 = min(box[3], k_box[3])
                inter_w = max(0, inter_x2 - inter_x1)
                inter_h = max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                
                if is_line and k_is_line:
                    # 横書き行の同一行マージ（左右タイルの分割線で分断された行を完全結合）
                    if is_horizontal and (k_w >= k_h):
                        cy1 = (box[1] + box[3]) / 2.0
                        k_cy = (k_box[1] + k_box[3]) / 2.0
                        cy_diff = abs(cy1 - k_cy)
                        min_h = min(h, k_h)
                        vert_overlap = inter_h / min_h if min_h > 0 else 0
                        h_ratio = h / k_h if k_h > 0 else 1.0

                        if cy_diff <= max(8.0, min_h * 0.35) and vert_overlap >= 0.50 and (0.50 <= h_ratio <= 1.80):
                            horiz_overlap = inter_w / min(w, k_w) if min(w, k_w) > 0 else 0
                            horiz_gap = max(0, max(box[0], k_box[0]) - min(box[2], k_box[2]))
                            is_contained = horiz_overlap > 0.60
                            det_conf = det.get('confidence', 0.5)
                            
                            if (horiz_gap < 80 or (horiz_overlap > 0 and not is_contained)):
                                new_x1 = min(k_box[0], box[0])
                                new_x2 = max(k_box[2], box[2])
                                expands_left = box[0] < k_box[0] - 25
                                expands_right = box[2] > k_box[2] + 25
                                
                                if expands_left or expands_right:
                                    if det_conf < 0.35:
                                        merged_into_existing = True
                                        break
                                    if w > k_w * 1.3:
                                        k_box[1] = box[1]
                                        k_box[3] = box[3]
                                    elif k_w > w * 1.3:
                                        pass
                                    else:
                                        k_box[1] = int(round((k_box[1] + box[1]) / 2.0))
                                        k_box[3] = int(round((k_box[3] + box[3]) / 2.0))
                                    k_box[0] = new_x1
                                    k_box[2] = new_x2
                                    k['confidence'] = max(k.get('confidence', 0.5), det.get('confidence', 0.5))
                                merged_into_existing = True
                                break
                    
                    # 縦書き行の同一列マージ（上下タイルの分割線で分断された列を完全結合）
                    elif not is_horizontal and (k_h > k_w):
                        cx1 = (box[0] + box[2]) / 2.0
                        k_cx = (k_box[0] + k_box[2]) / 2.0
                        cx_diff = abs(cx1 - k_cx)
                        min_w = min(w, k_w)
                        horiz_overlap = inter_w / min_w if min_w > 0 else 0
                        w_ratio = w / k_w if k_w > 0 else 1.0
                        is_contained = horiz_overlap > 0.60
                        det_conf = det.get('confidence', 0.5)

                        if cx_diff <= max(8.0, min_w * 0.35) and horiz_overlap >= 0.50 and (0.50 <= w_ratio <= 1.80):
                            vert_overlap = inter_h / min(h, k_h) if min(h, k_h) > 0 else 0
                            vert_gap = max(0, max(box[1], k_box[1]) - min(box[3], k_box[3]))
                            
                            if (vert_gap < 80 or (vert_overlap > 0 and not is_contained)):
                                new_y1 = min(k_box[1], box[1])
                                new_y2 = max(k_box[3], box[3])
                                expands_top = box[1] < k_box[1] - 25
                                expands_bottom = box[3] > k_box[3] + 25
                                
                                if expands_top or expands_bottom:
                                    if det_conf < 0.35:
                                        merged_into_existing = True
                                        break
                                    if h > k_h * 1.3:
                                        k_box[0] = box[0]
                                        k_box[2] = box[2]
                                    elif k_h > h * 1.3:
                                        pass
                                    else:
                                        k_box[0] = int(round((k_box[0] + box[0]) / 2.0))
                                        k_box[2] = int(round((k_box[2] + box[2]) / 2.0))
                                    k_box[1] = new_y1
                                    k_box[3] = new_y2
                                    k['confidence'] = max(k.get('confidence', 0.5), det.get('confidence', 0.5))
                                merged_into_existing = True
                                break

                # 通常のNMS重複判定
                if inter_area > 0:
                    k_area = k_w * k_h
                    union_area = area + k_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    containment = inter_area / area
                    if iou > iou_threshold or (containment > 0.80 and k.get('confidence', 0.5) >= det.get('confidence', 0.5)):
                        duplicate = True
                        break
                    total_overlap += inter_area
                    
        if not duplicate and not merged_into_existing:
            if total_overlap / area > 0.85:
                duplicate = True
            if not duplicate:
                det['box'] = box
                kept.append(det)
            
    return kept

def detect_tiled(detector, img):
    h, w = img.shape[:2]
    all_detections = list(detector.detect(img))
    
    if w > 1600 or h > 1600:
        tiles = []
        if w > 1600 and h > 1600:
            overlap_x = int(w * 0.15)
            overlap_y = int(h * 0.15)
            mid_x = w // 2
            mid_y = h // 2
            tiles.append((0, 0, mid_x + overlap_x, mid_y + overlap_y))
            tiles.append((mid_x - overlap_x, 0, w, mid_y + overlap_y))
            tiles.append((0, mid_y - overlap_y, mid_x + overlap_x, h))
            tiles.append((mid_x - overlap_x, mid_y - overlap_y, w, h))
        elif w > 1600:
            overlap_x = int(w * 0.15)
            mid_x = w // 2
            tiles.append((0, 0, mid_x + overlap_x, h))
            tiles.append((mid_x - overlap_x, 0, w, h))
        else:
            overlap_y = int(h * 0.15)
            mid_y = h // 2
            tiles.append((0, 0, w, mid_y + overlap_y))
            tiles.append((0, mid_y - overlap_y, w, h))
            
        for tx1, ty1, tx2, ty2 in tiles:
            tile_img = img[ty1:ty2, tx1:tx2]
            tile_dets = detector.detect(tile_img)
            for d in tile_dets:
                bx1, by1, bx2, by2 = d['box']
                d_offset = dict(d)
                d_offset['box'] = [bx1 + tx1, by1 + ty1, bx2 + tx1, by2 + ty1]
                all_detections.append(d_offset)
                
    blocks = [d for d in all_detections if d['class_index'] not in LINE_CLASS_INDICES and d.get('confidence', 1.0) >= 0.20]
    # 巨大な2次元領域（幅>120かつ高さ>120）は行ではなくブロック/ノイズのため行候補から除外
    lines = [
        d for d in all_detections 
        if d['class_index'] in LINE_CLASS_INDICES 
        and d.get('confidence', 1.0) >= 0.15
        and not ((d['box'][2] - d['box'][0] > 120) and (d['box'][3] - d['box'][1] > 120))
    ]
    
    merged_blocks = merge_detections_nms(blocks, iou_threshold=0.4)
    merged_lines = merge_detections_nms(lines, iou_threshold=0.5)
    
    # Remove block fragments that are single-line and covered by or overlapping lines
    valid_blocks = []
    for b in merged_blocks:
        if b['class_index'] == 0:
            bx1, by1, bx2, by2 = b['box']
            bw = bx2 - bx1
            bh = by2 - by1
            
            # Check if this block is just a single-line fragment inside an existing text line
            is_partial_fragment = False
            for l in merged_lines:
                lx1, ly1, lx2, ly2 = l['box']
                lw = lx2 - lx1
                lh = ly2 - ly1
                
                if abs((by1+by2)/2.0 - (ly1+ly2)/2.0) <= max(15, lh * 0.5):
                    if bh <= max(65, lh * 1.4):
                        if (lx1 <= bx1 + 35 and lx2 >= bx2 - 35) or (lw > bw * 1.2 and max(lx1, bx1) < min(lx2, bx2)):
                            is_partial_fragment = True
                            break
            if not is_partial_fragment:
                valid_blocks.append(b)
        else:
            valid_blocks.append(b)
            
    # Ensure every line has an enclosing block
    for line in merged_lines:
        lx1, ly1, lx2, ly2 = line['box']
        has_block = False
        for b in valid_blocks:
            if b['class_index'] == 0:
                bx1, by1, bx2, by2 = b['box']
                if bx1 <= lx1 + 50 and by1 <= ly1 + 50 and bx2 >= lx2 - 50 and by2 >= ly2 - 50:
                    has_block = True
                    break
        if not has_block:
            valid_blocks.append({
                'box': [max(0, lx1 - 10), max(0, ly1 - 10), min(w, lx2 + 10), min(h, ly2 + 10)],
                'confidence': line['confidence'],
                'class_index': 0,
                'pred_char_count': line.get('pred_char_count', 100.0)
            })
            
    return valid_blocks + merged_lines

def inference_on_detector(args,inputname:str,npimage:np.ndarray,outputpath:str,issaveimg:bool=True):
    print("[INFO] Intialize Model")
    detector = get_detector(args)
    print("[INFO] Inference Image")
    detections = detect_tiled(detector, npimage)
    classeslist=list(detector.classes.values())
    if issaveimg:
        drawimage = npimage.copy()
        pil_image =detector.draw_detections(drawimage, detections=detections)
        os.makedirs(outputpath,exist_ok=True)
        output_filepath = os.path.join(outputpath,f"viz_{Path(inputname).name}")
        if output_filepath.split(".")[-1]=="jp2":
            output_filepath=output_filepath[:-4]+".jpg"
        print(f"[INFO] Saving result on {output_filepath}")
        pil_image.save(output_filepath)
    return detections,classeslist

def process_detector(detector,inputname:str,npimage:np.ndarray,outputpath:str,issaveimg:bool=True):
    detections = detect_tiled(detector, npimage)
    classeslist=list(detector.classes.values())
    if issaveimg:
        drawimage = npimage.copy()
        pil_image =detector.draw_detections(drawimage, detections=detections)
        os.makedirs(outputpath,exist_ok=True)
        output_filepath = os.path.join(outputpath,f"viz_{Path(inputname).name}")
        if output_filepath.split(".")[-1]=="jp2":
            output_filepath=output_filepath[:-4]+".jpg"
        print(f"[INFO] Saving result on {output_filepath}")
        pil_image.save(output_filepath)
    return detections,classeslist

def _run_ocr_on_image_array(
    detector,
    recognizer30,
    recognizer50,
    recognizer100,
    inputname: str,
    img: np.ndarray,
    outputpath: str,
    save_viz: bool = False,
):
    img_h, img_w = img.shape[:2]
    detections, classeslist = process_detector(
        detector=detector,
        inputname=inputname,
        npimage=img,
        outputpath=outputpath,
        issaveimg=save_viz,
    )
    resultobj = [dict(), dict()]
    resultobj[0][0] = list()
    for i in range(17):
        resultobj[1][i] = []
    for det in detections:
        xmin, ymin, xmax, ymax = det["box"]
        conf = det["confidence"]
        if det["class_index"] == 0:
            resultobj[0][0].append([xmin, ymin, xmax, ymax])
        resultobj[1][det["class_index"]].append([xmin, ymin, xmax, ymax, conf, det["pred_char_count"]])

    xmlstr = convert_to_xml_string3(img_w, img_h, inputname, classeslist, resultobj)
    xmlstr = "<OCRDATASET>" + xmlstr + "</OCRDATASET>"
    root = ET.fromstring(xmlstr)
    eval_xml(root, logger=None)

    alllineobj = []
    tatelinecnt = 0
    alllinecnt = 0

    tail_limit = vertical_tail_limit(root, detections)
    for idx, lineobj in enumerate(root.findall(".//LINE")):
        xmin = int(lineobj.get("X"))
        ymin = int(lineobj.get("Y"))
        line_w = int(lineobj.get("WIDTH"))
        line_h = int(lineobj.get("HEIGHT"))
        is_vert = line_h > line_w
        tx1, ty1, tx2, ty2 = trim_line_box(img, xmin, ymin, xmin + line_w, ymin + line_h, not is_vert)
        xmin, ymin, line_w, line_h = tx1, ty1, max(1, tx2 - tx1), max(1, ty2 - ty1)
        lineobj.set("X", str(xmin))
        lineobj.set("Y", str(ymin))
        lineobj.set("WIDTH", str(line_w))
        lineobj.set("HEIGHT", str(line_h))
        try:
            pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
        except Exception:
            pred_char_cnt = 100.0
        if line_h > line_w:
            tatelinecnt += 1
            pad_x = vertical_recognition_padding(img, xmin, ymin, line_w, line_h, tail_limit)
            pad_y = 8
        else:
            pad_x = 8
            pad_y = 2
        alllinecnt += 1
        p_ymin = max(0, ymin - pad_y)
        p_ymax = min(img_h, ymin + line_h + pad_y)
        p_xmin = max(0, xmin - pad_x)
        p_xmax = min(img_w, xmin + line_w + pad_x)
        lineimg = img[p_ymin:p_ymax, p_xmin:p_xmax, :]
        alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt, crop_context=(img, p_ymin, p_ymax, p_xmin, p_xmax) if is_vert else None))

    if len(alllineobj) == 0 and len(detections) > 0:
        page = root.find("PAGE")
        for idx, det in enumerate(detections):
            xmin, ymin, xmax, ymax = det["box"]
            line_w = int(xmax - xmin)
            line_h = int(ymax - ymin)
            if line_w <= 0 or line_h <= 0:
                continue
            is_vert = line_h > line_w
            tx1, ty1, tx2, ty2 = trim_line_box(img, int(xmin), int(ymin), int(xmax), int(ymax), not is_vert)
            xmin, ymin, line_w, line_h = tx1, ty1, max(1, tx2 - tx1), max(1, ty2 - ty1)
            line_elem = ET.SubElement(page, "LINE")
            c_idx = int(det["class_index"])
            type_name = "本文"
            line_elem.set("TYPE", type_name)
            line_elem.set("X", str(int(xmin)))
            line_elem.set("Y", str(int(ymin)))
            line_elem.set("WIDTH", str(line_w))
            line_elem.set("HEIGHT", str(line_h))
            line_elem.set("CONF", f"{det['confidence']:0.3f}")
            pred_char_cnt = det.get("pred_char_count", 100.0)
            line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
            if line_h > line_w:
                tatelinecnt += 1
                pad_x = vertical_recognition_padding(img, xmin, ymin, line_w, line_h, tail_limit)
                pad_y = 8
            else:
                pad_x = 8
                pad_y = 2
            alllinecnt += 1
            p_ymin = max(0, int(ymin) - pad_y)
            p_ymax = min(img_h, int(ymax) + pad_y)
            p_xmin = max(0, int(xmin) - pad_x)
            p_xmax = min(img_w, int(xmax) + pad_x)
            lineimg = img[p_ymin:p_ymax, p_xmin:p_xmax, :]
            alllineobj.append(RecogLine(lineimg, idx, pred_char_cnt, crop_context=(img, p_ymin, p_ymax, p_xmin, p_xmax) if is_vert else None))

    resultlinesall = process_cascade(
        alllineobj,
        recognizer30,
        recognizer50,
        recognizer100,
        is_cascade=True,
    )

    evidence_by_index = {line.idx: getattr(line, "tcy_image_evidence", []) for line in alllineobj}
    resjsonarray = []
    text_layer_lines = []
    for idx, lineobj in enumerate(root.findall(".//LINE")):
        text = resultlinesall[idx] if idx < len(resultlinesall) else ""
        lineobj.set("STRING", text)
        xmin = int(lineobj.get("X"))
        ymin = int(lineobj.get("Y"))
        line_w = int(lineobj.get("WIDTH"))
        line_h = int(lineobj.get("HEIGHT"))
        is_vertical = line_h > line_w
        try:
            conf = float(lineobj.get("CONF"))
        except Exception:
            conf = 0.0

        type_str = lineobj.get("TYPE", "")
        c_idx = classeslist.index(type_str) if type_str in classeslist else 1

        if conf == 0.0 and (not text.strip() or len(text.strip()) <= 3):
            continue

        resjsonarray.append({
            "boundingBox": [
                [xmin, ymin],
                [xmin, ymin + line_h],
                [xmin + line_w, ymin],
                [xmin + line_w, ymin + line_h],
            ],
            "id": len(resjsonarray),
            "isVertical": "true" if is_vertical else "false",
            "text": text,
            "isTextline": "true",
            "confidence": conf,
            "class_index": c_idx,
            "tcy_image_evidence": evidence_by_index.get(idx, []),
        })
        text_layer_lines.append({
            "x": xmin,
            "y": ymin,
            "width": line_w,
            "height": line_h,
            "text": text,
            "is_vertical": is_vertical,
        })

    page_xml = ET.tostring(root.find("PAGE"), encoding="unicode")
    page_text = "\n".join(resultlinesall)
    return {
        "page_xml": page_xml,
        "text": page_text,
        "json_lines": resjsonarray,
        "text_layer_lines": text_layer_lines,
        "img_width": img_w,
        "img_height": img_h,
        "img_name": inputname,
        "line_count": alllinecnt,
        "vertical_line_count": tatelinecnt,
    }

def _text_layer_font_size(width: float, height: float, text: str, is_vertical: bool):
    text_len = max(len(text), 1)
    if is_vertical:
        size = min(width * 0.95, (height / text_len) * 1.8)
    else:
        size = min(height * 0.95, (width / text_len) * 1.8)
    return max(1.0, min(size, 72.0))

def _draw_text_layer_line(
    canvas_obj,
    line: dict,
    img_width: int,
    img_height: int,
    page_width: float,
    page_height: float,
    visible: bool,
):
    text = line["text"]
    if not text:
        return
    scale_x = page_width / max(img_width, 1)
    scale_y = page_height / max(img_height, 1)
    x = line["x"] * scale_x
    y_top = page_height - line["y"] * scale_y
    width = line["width"] * scale_x
    height = line["height"] * scale_y
    if width <= 0 or height <= 0:
        return
    is_vertical = line["is_vertical"]
    fontsize = _text_layer_font_size(width, height, text, is_vertical)
    if is_vertical:
        canvas_obj.setFont("HeiseiMin-W3", fontsize)
        draw_x = x + width * 0.5
        draw_y = y_top
    else:
        canvas_obj.setFont("HeiseiKakuGo-W5", fontsize)
        draw_x = x
        draw_y = y_top - height + max((height - fontsize) * 0.5, 0)
    if visible:
        canvas_obj.setFillColorRGB(0, 0, 1)
    canvas_obj.drawString(draw_x, draw_y, text)

def embed_text_layer_pdf(input_pdf: str, output_pdf: str, page_results: list, visible_text: bool = False):
    try:
        from io import BytesIO
        from pypdf import PdfReader, PdfWriter
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    except ImportError as exc:
        raise RuntimeError(
            "PDF text-layer output requires pypdf and reportlab. Install dependencies from requirements.txt."
        ) from exc

    output_path = Path(output_pdf)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.resolve() == Path(input_pdf).resolve():
        raise ValueError("Output PDF must be different from the input PDF.")

    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5", isVertical=False))
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3", isVertical=True))

    reader = PdfReader(input_pdf)
    if len(reader.pages) != len(page_results):
        raise ValueError(f"PDF page count mismatch: {len(reader.pages)} pages, {len(page_results)} OCR results")

    writer = PdfWriter()
    if reader.metadata:
        writer.add_metadata({key: str(value) for key, value in reader.metadata.items() if value is not None})
    for page, page_result in zip(reader.pages, page_results):
        page_width = float(page.mediabox.width)
        page_height = float(page.mediabox.height)
        overlay_buffer = BytesIO()
        overlay_canvas = canvas.Canvas(overlay_buffer, pagesize=(page_width, page_height))
        if visible_text:
            overlay_canvas.setFillColorRGB(0, 0, 1)
        else:
            overlay_canvas.setFillAlpha(0)
        for line in page_result["text_layer_lines"]:
            _draw_text_layer_line(
                canvas_obj=overlay_canvas,
                line=line,
                img_width=page_result["img_width"],
                img_height=page_result["img_height"],
                page_width=page_width,
                page_height=page_height,
                visible=visible_text,
            )
        overlay_canvas.save()
        overlay_buffer.seek(0)
        overlay_reader = PdfReader(overlay_buffer)
        page.merge_page(overlay_reader.pages[0])
        writer.add_page(page)

    with open(output_path, "wb") as wf:
        writer.write(wf)

def process_pdf_documents(args, pdf_paths: list[str]):
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires pypdfium2. Install dependencies from requirements.txt."
        ) from exc
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PDF text-layer output requires pypdf. Install dependencies from requirements.txt."
        ) from exc

    if len(pdf_paths) > 1 and getattr(args, "pdf_output", None):
        print("--pdf-output can only be used with a single PDF input.")
        return

    if not os.path.exists(args.output):
        print("Output Directory is not found.")
        return

    print("[INFO] Intialize Model")
    detector = get_detector(args)
    recognizer100 = get_recognizer(args=args)
    recognizer30 = DeferredRecognizer(args, args.rec_weights30)
    recognizer50 = DeferredRecognizer(args, args.rec_weights50)

    render_scale = max(float(getattr(args, "pdf_render_dpi", 300)), 1.0) / 72.0

    for pdf_path in pdf_paths:
        start = time.time()
        pdf_path_obj = Path(pdf_path)
        output_stem = pdf_path_obj.stem
        pdf_doc = pypdfium2.PdfDocument(str(pdf_path_obj))
        page_results = []
        all_json_contents = []
        page_infos = []
        all_text_pages = []
        all_page_xml = []

        for page_index in range(len(pdf_doc)):
            page_name = f"{output_stem}_{page_index + 1:05}.png"
            print(f"[INFO] OCR PDF page {page_index + 1}/{len(pdf_doc)}: {pdf_path_obj.name}")
            rendered_pages = pdf_doc.render(
                pypdfium2.PdfBitmap.to_pil,
                page_indices=[page_index],
                scale=render_scale,
            )
            pil_image = next(iter(rendered_pages)).convert("RGB")
            img = np.array(pil_image)
            page_result = _run_ocr_on_image_array(
                detector=detector,
                recognizer30=recognizer30,
                recognizer50=recognizer50,
                recognizer100=recognizer100,
                inputname=page_name,
                img=img,
                outputpath=args.output,
                save_viz=args.viz,
            )
            page_results.append(page_result)
            all_json_contents.append(page_result["json_lines"])
            page_infos.append({
                "page_index": page_index,
                "img_width": page_result["img_width"],
                "img_height": page_result["img_height"],
                "img_name": page_result["img_name"],
            })
            all_text_pages.append(page_result["text"])
            all_page_xml.append(page_result["page_xml"])

        pdf_doc.close()

        if not getattr(args, "json_only", False):
            with open(os.path.join(args.output, output_stem + ".xml"), "w", encoding="utf-8") as wf:
                wf.write("<OCRDATASET>\n")
                wf.write("\n".join(all_page_xml))
                wf.write("\n</OCRDATASET>")
            with open(os.path.join(args.output, output_stem + ".txt"), "w", encoding="utf-8") as wtf:
                wtf.write("\n\n".join(all_text_pages))

        with open(os.path.join(args.output, output_stem + ".json"), "w", encoding="utf-8") as wf:
            alljsonobj = {
                "contents": all_json_contents,
                "pdfinfo": {
                    "pdf_path": str(pdf_path_obj),
                    "pdf_name": pdf_path_obj.name,
                    "page_count": len(page_results),
                    "render_dpi": float(getattr(args, "pdf_render_dpi", 300)),
                },
                "pages": page_infos,
            }
            wf.write(json.dumps(alljsonobj, ensure_ascii=False, indent=2))

        output_pdf = getattr(args, "pdf_output", None)
        if not output_pdf:
            output_pdf = os.path.join(args.output, output_stem + "_text.pdf")
        print(f"[INFO] Writing text-layer PDF: {output_pdf}")
        embed_text_layer_pdf(
            input_pdf=str(pdf_path_obj),
            output_pdf=output_pdf,
            page_results=page_results,
            visible_text=getattr(args, "pdf_visible_text", False),
        )
        print("Total PDF calculation time:", time.time() - start)

def process(args):
    rawinputpathlist=[]
    inputpathlist=[]
    pdfpathlist=[]
    if args.sourcedir is not None:
        for inputpath in glob.glob(os.path.join(args.sourcedir,"*")):
            rawinputpathlist.append(inputpath)
    if args.sourceimg is not None:
        rawinputpathlist.append(args.sourceimg)
    if args.sourcepdf is not None:
        pdfpathlist.append(args.sourcepdf)
    for inputpath in rawinputpathlist:
        ext=inputpath.split(".")[-1]
        if ext.lower() in ["jpg","png","tiff","jp2","tif","jpeg","bmp","webp"]:
            inputpathlist.append(inputpath)
        elif ext.lower() in ["pdf",]:
            pdfpathlist.append(inputpath)

    if len(pdfpathlist) > 0:
        process_pdf_documents(args, pdfpathlist)
        if len(inputpathlist) == 0:
            return
    if len(inputpathlist)==0:
        print("Images are not found.")
        return
    if not os.path.exists(args.output):
        print("Output Directory is not found.")
        return
    
    detector=get_detector(args)
    recognizer100=get_recognizer(args=args)
    recognizer30=DeferredRecognizer(args, args.rec_weights30)
    recognizer50=DeferredRecognizer(args, args.rec_weights50)
    tatelinecnt=0
    alllinecnt=0
    
    for inputpath in inputpathlist:
        ext=inputpath.split(".")[-1]
        pil_image = Image.open(inputpath).convert('RGB')
        img = np.array(pil_image)
        start = time.time()
        allxmlstr="<OCRDATASET>\n"
        alltextlist=[]
        resjsonarray=[]
        imgname=os.path.basename(inputpath)
        img_h,img_w=img.shape[:2]
        detections,classeslist=process_detector(detector,inputname=imgname,npimage=img,outputpath=args.output,issaveimg=args.viz)
        e1=time.time()
        resultobj=[dict(),dict()]
        resultobj[0][0]=list()
        for i in range(17):
            resultobj[1][i]=[]
        for det in detections:
            xmin,ymin,xmax,ymax=det["box"]
            conf=det["confidence"]
            char_count=det["pred_char_count"]
            if det["class_index"]==0:
                resultobj[0][0].append([xmin,ymin,xmax,ymax])
            resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf,char_count])
        xmlstr=convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj)
        xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"
        # print(xmlstr)
        root = ET.fromstring(xmlstr)
        eval_xml(root, logger=None)
        alllineobj = []
        alltextlist = []

        tail_limit = vertical_tail_limit(root, detections)
        for idx, lineobj in enumerate(root.findall(".//LINE")):
            xmin = int(lineobj.get("X"))
            ymin = int(lineobj.get("Y"))
            line_w = int(lineobj.get("WIDTH"))
            line_h = int(lineobj.get("HEIGHT"))
            is_vert = line_h > line_w
            tx1, ty1, tx2, ty2 = trim_line_box(img, xmin, ymin, xmin + line_w, ymin + line_h, not is_vert)
            xmin, ymin, line_w, line_h = tx1, ty1, max(1, tx2 - tx1), max(1, ty2 - ty1)
            lineobj.set("X", str(xmin))
            lineobj.set("Y", str(ymin))
            lineobj.set("WIDTH", str(line_w))
            lineobj.set("HEIGHT", str(line_h))
            try:
                pred_char_cnt = float(lineobj.get("PRED_CHAR_CNT"))
            except:
                pred_char_cnt = 100.0
            
            if line_h > line_w:
                tatelinecnt += 1
                pad_x = vertical_recognition_padding(img, xmin, ymin, line_w, line_h, tail_limit)
                pad_y = 8
            else:
                pad_x = 8
                pad_y = 2
            alllinecnt += 1
            # 部分画像の切り出し
            p_ymin = max(0, ymin - pad_y)
            p_ymax = min(img_h, ymin + line_h + pad_y)
            p_xmin = max(0, xmin - pad_x)
            p_xmax = min(img_w, xmin + line_w + pad_x)
            lineimg = img[p_ymin:p_ymax, p_xmin:p_xmax, :]
            linerecogobj = RecogLine(lineimg, idx, pred_char_cnt, crop_context=(img, p_ymin, p_ymax, p_xmin, p_xmax) if is_vert else None)
            alllineobj.append(linerecogobj)

        if len(alllineobj) == 0 and len(detections) > 0:
            # LINE 要素がないが検出がある場合は検出領域を LINE として扱う
            page = root.find("PAGE")
            for idx, det in enumerate(detections):
                xmin, ymin, xmax, ymax = det["box"]
                line_w = int(xmax - xmin)
                line_h = int(ymax - ymin)
                if line_w > 0 and line_h > 0:
                    is_vert = line_h > line_w
                    tx1, ty1, tx2, ty2 = trim_line_box(img, int(xmin), int(ymin), int(xmax), int(ymax), not is_vert)
                    xmin, ymin, line_w, line_h = tx1, ty1, max(1, tx2 - tx1), max(1, ty2 - ty1)
                    line_elem = ET.SubElement(page, "LINE")
                    c_idx = int(det["class_index"])
                    type_name = "本文"
                    line_elem.set("TYPE", type_name)
                    line_elem.set("X", str(int(xmin)))
                    line_elem.set("Y", str(int(ymin)))
                    line_elem.set("WIDTH", str(line_w))
                    line_elem.set("HEIGHT", str(line_h))
                    line_elem.set("CONF", f"{det['confidence']:0.3f}")
                    pred_char_cnt = det.get("pred_char_count", 100.0)
                    line_elem.set("PRED_CHAR_CNT", f"{pred_char_cnt:0.3f}")
                    if line_h > line_w:
                        tatelinecnt += 1
                        pad_x = vertical_recognition_padding(img, xmin, ymin, line_w, line_h, tail_limit)
                        pad_y = 8
                    else:
                        pad_x = 8
                        pad_y = 2
                    alllinecnt += 1
                    p_ymin = max(0, int(ymin) - pad_y)
                    p_ymax = min(img_h, int(ymax) + pad_y)
                    p_xmin = max(0, int(xmin) - pad_x)
                    p_xmax = min(img_w, int(xmax) + pad_x)
                    lineimg = img[p_ymin:p_ymax, p_xmin:p_xmax, :]
                    linerecogobj = RecogLine(lineimg, idx, pred_char_cnt, crop_context=(img, p_ymin, p_ymax, p_xmin, p_xmax) if is_vert else None)
                    alllineobj.append(linerecogobj)

        # 認識プロセス
        resultlinesall = process_cascade(
            alllineobj, recognizer30, recognizer50, recognizer100, is_cascade=True
        )
        alltextlist.append("\n".join(resultlinesall))
        
        evidence_by_index = {line.idx: getattr(line, "tcy_image_evidence", []) for line in alllineobj}
        for idx,lineobj in enumerate(root.findall(".//LINE")):
            lineobj.set("STRING",resultlinesall[idx])
            xmin=int(lineobj.get("X"))
            ymin=int(lineobj.get("Y"))
            line_w=int(lineobj.get("WIDTH"))
            line_h=int(lineobj.get("HEIGHT"))
            try:
                conf=float(lineobj.get("CONF"))
            except:
                conf=0.0
            
            # XML TYPE -> c_idx
            type_str = lineobj.get("TYPE", "")
            c_idx = classeslist.index(type_str) if type_str in classeslist else 1

            if conf == 0.0 and (not resultlinesall[idx].strip() or len(resultlinesall[idx].strip()) <= 3):
                continue

            jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                "id": len(resjsonarray),"isVertical": "true" if line_h > line_w else "false","text": resultlinesall[idx],"isTextline": "true","confidence": conf, "class_index": c_idx}
            jsonobj["tcy_image_evidence"] = evidence_by_index.get(idx, [])
            resjsonarray.append(jsonobj)

        allxmlstr+=(ET.tostring(root.find("PAGE"), encoding='unicode')+"\n")
        allxmlstr+="</OCRDATASET>"
        if alllinecnt>0 and tatelinecnt/alllinecnt>0.5:
            alltextlist=alltextlist[::-1]
        output_stem = os.path.splitext(os.path.basename(inputpath))[0]
        
        if not getattr(args, "json_only", False):
            with open(os.path.join(args.output,output_stem+".xml"),"w",encoding="utf-8") as wf:
                wf.write(allxmlstr)
                
        with open(os.path.join(args.output,output_stem+".json"),"w",encoding="utf-8") as wf:
            alljsonobj={
                "contents":[resjsonarray],
                "imginfo": {
                    "img_width": img_w,
                    "img_height": img_h,
                    "img_path":inputpath,
                    "img_name":os.path.basename(inputpath)
                }
            }
            alljsonstr=json.dumps(alljsonobj,ensure_ascii=False,indent=2)
            wf.write(alljsonstr)
            
        if not getattr(args, "json_only", False):
            with open(os.path.join(args.output,output_stem+".txt"),"w",encoding="utf-8") as wtf:
                wtf.write("\n".join(alltextlist))
        print("Total calculation time (Detection + Recognition):",time.time()-start)

def main():
    import argparse
    from pathlib import Path
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Arguments for NDLkotenOCR-Lite")

    parser.add_argument("--sourcedir", type=str, required=False, help="Path to image directory")
    parser.add_argument("--sourceimg", type=str, required=False, help="Path to image directory")
    parser.add_argument("--sourcepdf", type=str, required=False, help="Path to source PDF")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    parser.add_argument("--viz", type=bool, required=False, help="Save visualized image",default=False)
    parser.add_argument("--pdf-output", type=str, required=False, help="Path to output text-layer PDF")
    parser.add_argument("--pdf-render-dpi", "--pdf-dpi", dest="pdf_render_dpi", type=float, required=False, default=300.0, help="DPI used to render PDF pages for OCR")
    parser.add_argument("--pdf-visible-text", action="store_true", help="Draw PDF text layer visibly in blue for debugging")
    parser.add_argument("--det-weights", type=str, required=False, help="Path to deim onnx file", default=str(base_dir / "model" / "deim-s-1024x1024.onnx"))
    parser.add_argument("--det-classes", type=str, required=False, help="Path to list of class in yaml file", default=str(base_dir / "config" / "ndl.yaml"))
    parser.add_argument("--det-score-threshold", type=float, required=False, default=0.2)
    parser.add_argument("--det-conf-threshold", type=float, required=False, default=0.25)
    parser.add_argument("--det-iou-threshold", type=float, required=False, default=0.2)
    parser.add_argument("--simple-mode", type=bool, required=False, help="Read line with one model(Setting this option to True will slow down processing, but it simplifies the architecture and may slightly improve accuracy.)",default=False)
    parser.add_argument("--rec-weights30", type=str, required=False, help="Path to parseq-tiny onnx file", default=str(base_dir / "model" / "parseq-ndl-24x256-30-tiny-189epoch-tegaki3-r8data-202604.onnx"))
    parser.add_argument("--rec-weights50", type=str, required=False, help="Path to parseq-tiny onnx file", default=str(base_dir / "model" / "parseq-ndl-24x384-50-tiny-300epoch-tegaki3-r8data-202604.onnx"))
    parser.add_argument("--rec-weights", type=str, required=False, help="Path to parseq-tiny onnx file", default=str(base_dir / "model" / "parseq-ndl-24x768-100-tiny-153epoch-tegaki3-r8data-202604.onnx"))
    parser.add_argument("--rec-classes", type=str, required=False, help="Path to list of class in yaml file", default=str(base_dir / "config" / "NDLmoji.yaml"))
    parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cuda)", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--enable-tcy", action="store_true", dest="enable_tcy", default=False, help="Enable tate-chuu-yoko (縦中横) detection for vertical text (e.g. newspaper OCR)")
    parser.add_argument("--json-only", action="store_true", help="Disable .xml and .txt output and only output JSON")
    args, remaining = parser.parse_known_args()
    if args.enable_tcy and remaining:
        from tcy_wrapper import add_tcy_arguments
        tcy_parser = add_tcy_arguments(parser)
        tcy_args = tcy_parser.parse_args(remaining)
        for k, v in vars(tcy_args).items():
            if v is not None:
                setattr(args, k, v)
    args = parser.parse_args()
    process(args)

if __name__=="__main__":
    main()
