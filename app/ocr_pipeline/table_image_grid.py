"""Recognize real cell images without distributing a line's text by character count."""
from pathlib import Path
import json, subprocess, tempfile
import cv2
import numpy as np


def render_table_image(page, region, image_override=None, page_size=None):
    import fitz
    x, y, w, h = [int(region.get(k, 0)) for k in ("x", "y", "w", "h")]
    if w <= 0 or h <= 0:
        raise ValueError("表領域のサイズを確認してください。")
    if image_override is not None:
        crop = image_override[y:y+h, x:x+w]
        if crop.size == 0:
            raise ValueError("表領域が画像の外にあります。")
        return cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    pw, ph = page_size or (round(page.rect.width * 2), round(page.rect.height * 2))
    sx, sy = page.rect.width / pw, page.rect.height / ph
    clip = fitz.Rect(x*sx, y*sy, (x+w)*sx, (y+h)*sy)
    if not page.rect.contains(clip):
        raise ValueError("表領域がPDFページの外にあります。")
    pix = page.get_pixmap(matrix=fitz.Matrix(3/sx, 3/sy), clip=clip, alpha=False, colorspace=fitz.csRGB)
    return cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)


def extract_image_grid(image, x, y, w, h, rules, doc_type="japanese", worker_path=None, model_path=None, command=None):
    from app.ocr_pipeline.engine_paths import get_ndlocr_command, get_project_root
    root = Path(__file__).resolve().parent
    worker = Path(worker_path or root / "table_cell_worker.py")
    model = Path(model_path or get_project_root() / "models/table/en_PP-OCRv4_rec_mobile.onnx")
    command = command or get_ndlocr_command()
    if not command or not model.is_file():
        raise RuntimeError("表のセル認識に必要なOCRモデルが見つかりません。結果は変更していません。")
    def clustered(values):
        result = []
        for value in sorted(set(int(v) for v in values)):
            if not result or value-result[-1] > 6:
                result.append(value)
        return result
    xs = [x] + clustered(r["pos"] for r in rules if r.get("is_vertical") and x+4 < r["pos"] < x+w-4) + [x+w]
    ys = [y] + clustered(r["pos"] for r in rules if not r.get("is_vertical") and y+4 < r["pos"] < y+h-4) + [y+h]
    columns, rows = len(xs)-1, len(ys)-1
    sx, sy = image.shape[1]/w, image.shape[0]/h
    specs, merges, cells = [], [], []
    for r in range(rows):
        middle = (ys[r]+ys[r+1])/2
        active = [0]
        for c, boundary in enumerate(xs[1:-1], 1):
            if any(line.get("is_vertical") and abs(line["pos"]-boundary)<=6
                   and line.get("start", y)-2 <= middle <= line.get("end", y+h)+2 for line in rules):
                active.append(c)
        active.append(columns)
        for c, end in zip(active, active[1:]):
            span = end-c
            specs.append({"box": [round((xs[c]-x)*sx), round((ys[r]-y)*sy), round((xs[end]-x)*sx), round((ys[r+1]-y)*sy)],
                          "numeric": span == 1 and c > 0})
            cells.append({"row": r+1, "column": c+1, "x": xs[c], "y": ys[r], "width": xs[end]-xs[c], "height": ys[r+1]-ys[r], "colspan": span})
            if span > 1:
                merges.append({"row": r, "col": c, "rowspan": 1, "colspan": span})
    if len(specs) > 1200:
        raise ValueError("表の区切りが多すぎます。罫線の設定を確認してください。")
    with tempfile.TemporaryDirectory(prefix="ocr_table_") as tmp:
        folder = Path(tmp)
        image_path, request_path, response_path = folder/"image.png", folder/"request.json", folder/"result.json"
        cv2.imencode(".png", image)[1].tofile(str(image_path))
        request_path.write_text(json.dumps({"image": str(image_path), "model": str(model), "cells": specs, "doc_type": doc_type}), encoding="utf-8")
        from app.ocr_pipeline.ndlocr_engine import _NDLOCR_PROCESS_LOCK
        try:
            with _NDLOCR_PROCESS_LOCK:
                result = subprocess.run([command[0], str(worker), str(request_path), str(response_path)], cwd=get_project_root(),
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("表の認識が時間内に終わりませんでした。元の結果は保持しています。") from exc
        if result.returncode or not response_path.is_file():
            detail = result.stderr.decode("utf-8", errors="replace")[-1500:]
            raise RuntimeError("表のセル認識に失敗しました。元の結果は保持しています。 " + detail)
        recognized = json.loads(response_path.read_text(encoding="utf-8"))
    if len(recognized) != len(cells):
        raise RuntimeError("表の認識結果とセル数が一致しません。")
    grid = [["" for _ in range(columns)] for _ in range(rows)]
    for cell, value in zip(cells, recognized):
        cell.update(value)
        cell["ocr_count"] = 1 if value["text"] else 0
        grid[cell["row"]-1][cell["column"]-1] = value["text"]
    if not any(any(row) for row in grid):
        raise ValueError("表の文字を認識できませんでした。元の結果は保持しています。")
    return {"rule_lines": rules, "rows": grid, "cells": cells, "merges": merges, "merge_spans": merges,
            "columns": columns, "row_count": rows, "text": "\n".join("\t".join(row) for row in grid),
            "v_clustered": xs[1:-1], "h_clustered": ys[1:-1], "all_v_clustered": xs, "all_h_clustered": ys}
