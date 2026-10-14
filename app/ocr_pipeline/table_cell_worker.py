"""Local, image-based recognition of horizontal table cells (OCR runtime)."""
from pathlib import Path
import json, math, re, sys, types
import cv2
import numpy as np
import onnxruntime as ort

NUMERIC = re.compile(r"[-+]?\d+(?:[.,]\d+)*%?")


def recognize(image, specs, model_path, doc_type="japanese"):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    chars = [""] + session.get_modelmeta().custom_metadata_map["character"].splitlines() + [" "]
    input_name = session.get_inputs()[0].name
    japanese = {}

    def latin_read(crop):
        h, w = crop.shape[:2]
        width = max(8, math.ceil(48 * w / h))
        arr = np.zeros((1, 3, 48, max(320, width)), np.float32)
        arr[0, :, :, :width] = (cv2.resize(crop, (width, 48)).astype(np.float32) / 127.5 - 1).transpose(2, 0, 1)
        pred = session.run(None, {input_name: arr})[0][0]
        previous = -1
        text, scores = "", []
        for index, token in enumerate(pred.argmax(1)):
            if token and token != previous:
                text += chars[token]
                scores.append(float(pred[index, token]))
            previous = token
        return text.strip(), float(np.mean(scores)) if scores else 0

    def read_piece(piece, allow_numeric):
        piece = cv2.copyMakeBorder(piece, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        value, score = latin_read(piece)
        if allow_numeric and NUMERIC.fullmatch(value) and score >= .75:
            # Small decimal points/minus signs can disappear during resampling.
            # Retain a punctuation-bearing reading only when another rendering
            # of the same pixels agrees on all digits; never infer it from context.
            readings = [(value, score)]
            gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
            for threshold in (100, 150, 200, 230):
                binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
                candidate, confidence = latin_read(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR))
                if NUMERIC.fullmatch(candidate) and confidence >= .78:
                    readings.append((candidate, confidence))
            same_digits = [(v, s) for v, s in readings if re.sub(r"\D", "", v) == re.sub(r"\D", "", value)]
            value, score = max(same_digits, key=lambda pair: (len(pair[0]), pair[1]))
            return value, score
        if doc_type in ("western", "english"):
            return value, score
        model_length = 30 if piece.shape[1] / piece.shape[0] < 15 else 100
        if model_length not in japanese:
            import ocr
            root = Path(ocr.__file__).parent
            model = next((root / "model").glob(f"*parseq*-{model_length}-*.onnx"))
            args = types.SimpleNamespace(rec_classes=str(root / "config/NDLmoji.yaml"), device="cpu")
            japanese[model_length] = ocr.get_recognizer(args, str(model))
        if piece.shape[1] / piece.shape[0] > 30:
            # Long Japanese notes lose punctuation when squeezed into one input.
            # Split only at image whitespace and concatenate the independent crops.
            columns = (cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY) < 160).sum(axis=0)
            cuts, left = [0], 0
            target, radius = piece.shape[0] * 14, piece.shape[0]
            while piece.shape[1] - left > target * 1.4:
                center = left + target
                options = range(max(left + 1, center-radius), min(piece.shape[1], center+radius))
                cut = min(options, key=lambda pos: (columns[pos] > 0, abs(pos-center)))
                if columns[cut] > 0:
                    break
                cuts.append(cut)
                left = cut
            cuts.append(piece.shape[1])
            if len(cuts) > 2:
                return "".join(read_piece(piece[:, a:b], False)[0] for a, b in zip(cuts, cuts[1:])), None
        return japanese[model_length].read(piece).strip(), None

    cleaned = image.copy()
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    binary = (gray < 160).astype(np.uint8) * 255
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, max(60, image.shape[1] // 20)), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((max(60, image.shape[0] // 4), 1), np.uint8))
    cleaned[cv2.dilate(horizontal | vertical, np.ones((3, 3), np.uint8)) > 0] = 255
    results = []
    for spec in specs:
        x1, y1, x2, y2 = spec["box"]
        band = cleaned[max(0, y1 + 3):min(cleaned.shape[0], y2 - 3)]
        if band.size == 0:
            results.append({"text": "", "confidence": None})
            continue
        ink = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) < 160
        # Manual rules are logical borders. Move a border only to nearby white
        # space so a glyph crossing that border is not cut in half.
        def snap(x):
            if x <= 0 or x >= band.shape[1]:
                return max(0, min(band.shape[1], x))
            radius = min(12, max(2, (x2 - x1) // 6))
            candidates = np.arange(max(0, x - radius), min(band.shape[1], x + radius + 1))
            counts = ink[:, candidates].sum(0)
            return int(min(candidates, key=lambda p: (counts[p - candidates[0]] > 0, abs(p - x))))
        left, right = snap(x1), snap(x2)
        crop = band[:, left:right]
        if crop.size == 0:
            results.append({"text": "", "confidence": None})
            continue
        occupied = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 160
        ys = np.flatnonzero(occupied.any(axis=1))
        groups = np.split(ys, np.flatnonzero(np.diff(ys) > 2) + 1) if len(ys) else []
        values, scores = [], []
        for group in groups:
            if len(group) < 3:
                continue
            top, bottom = int(group[0]), int(group[-1]) + 1
            xs = np.flatnonzero(occupied[top:bottom].any(axis=0))
            if not len(xs):
                continue
            value, score = read_piece(crop[top:bottom, int(xs[0]):int(xs[-1]) + 1], spec["numeric"])
            values.append(value)
            if score is not None:
                scores.append(score)
        results.append({"text": "\n".join(values), "confidence": min(scores) if scores else None})
    return results


if __name__ == "__main__":
    request_path, response_path = map(Path, sys.argv[1:3])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    image = cv2.imdecode(np.fromfile(request["image"], np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Table image could not be read")
    result = recognize(image, request["cells"], request["model"], request.get("doc_type", "japanese"))
    response_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
