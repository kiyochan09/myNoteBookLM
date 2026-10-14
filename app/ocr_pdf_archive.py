"""Portable OCR editing data and verified, non-destructive PDF text layers.

Imported originals are preserved; editing an archive replaces that saved PDF.
Raw data deletion requires a saved archive
and an unchanged inventory. Only text streams created by this module are replaced.
"""
from __future__ import annotations

import base64
import difflib
import functools
import hashlib
import json
import math
import os
import re
import shutil
import threading
import uuid
import unicodedata
import zlib
from datetime import datetime, timezone
from pathlib import Path

import fitz
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

STORAGE_LOCK = threading.RLock()
ATTACHMENT = "MyNotebookLM-OCR-edit-v1.json"
MAX_EDIT_BYTES = 64 * 1024 * 1024

# Embed a Unicode font with its own ToUnicode map. The predefined "japan"
# CMap cannot encode every CJK character (e.g. 陕 / 蓦), even though the
# bundled font contains their glyphs. ActualText over a missing glyph can
# then produce duplicated extraction instead of a faithful searchable layer.
OCR_FONT = fitz.Font("cjk")
OCR_FONT_BUFFER = OCR_FONT.buffer


class PdfEditRequest(BaseModel):
    pages: dict[str, dict]


def serialized(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with STORAGE_LOCK:
            return fn(*args, **kwargs)
    return wrapped


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def revision(pages, with_images=False):
    ignore = {"filename", "last_modified", "current_page", "page", "page_number",
              "total_pages", "archive_restored", "success", "file_name"}
    def stable(value):
        if isinstance(value, list):
            return [stable(v) for v in value]
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in ignore}
        return value
    values = stable(compact(pages))
    if with_images:
        values["_corrected_images"] = {k: digest(base64.b64decode(v["corrected_image"].split(",", 1)[-1], validate=True))
                                        for k, v in pages.items() if v.get("corrected_image")}
    return digest(json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def compact(data, custom_image=False):
    """Keep editing structure, not disposable page images or machine-local paths."""
    if isinstance(data, list):
        return [compact(v, custom_image) for v in data]
    if not isinstance(data, dict):
        return data
    custom_image = custom_image or bool(data.get("is_custom_image"))
    skip = {"corrected_image", "file_path", "last_modified", "file_size_kb"}
    return {k: compact(v, custom_image) for k, v in data.items()
            if k not in skip and (k != "image_base64" or custom_image)}


def clean_text(value):
    return "".join(c for c in logical_text(value) if not c.isspace())


def logical_text(value):
    text = str(value or "")
    text = re.sub(r"^(?:===|---)\s*ページ\s*\d+\s*(?:===|---)\s*", "", text)
    text = re.sub(r"^\[P\d+\]\s*", "", text)
    text = re.sub(r"【(?:大|中|小)?見出し】\s*", "", text)
    text = re.sub(r"⟦注:[^⟧]+⟧", "", text)
    return "".join(c for c in text if ord(c) >= 32).strip()


def rect_of(item):
    values = [float(item.get(k, 0)) for k in ("x", "y", "w", "h")]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("文字座標に不正な値があります。")
    x, y, w, h = values
    return fitz.Rect(x, y, x + w, y + h)


def corrected_runs(lines, target):
    """Keep edits in original boxes; added text follows its preceding anchor."""
    target = logical_text(target)
    source_parts = [logical_text(l.get("text")) for l in lines]
    source = "".join(source_parts)
    owners = [i for i, part in enumerate(source_parts) for _ in part]
    pieces = ["" for _ in lines]
    if not source:
        pieces[0] = target
    else:
        for op, a, b, c, d in difflib.SequenceMatcher(None, source, target, autojunk=False).get_opcodes():
            if op == "equal":
                for offset, char in enumerate(target[c:d]):
                    pieces[owners[a + offset]] += char
            elif op == "insert":
                # Inserted characters have no OCR coordinates. Continue from
                # the preceding character's line, including at line boundaries.
                # At the very beginning there is only the first line to anchor.
                pieces[owners[max(0, a - 1)]] += target[c:d]
            elif op == "replace":
                for offset, char in enumerate(target[c:d]):
                    pos = min(len(owners) - 1, a + (offset * (b - a) // max(1, d - c)))
                    pieces[owners[pos]] += char
    return [{**line, "text": text} for line, text in zip(lines, pieces) if text]


def make_runs(data):
    """Body text is authoritative. Region records anchor headings/notes/tables."""
    regions = sorted(data.get("regions") or [], key=lambda r: (r.get("reading_order") or 0))
    lines = [l for l in data.get("ocr_lines", []) if rect_of(l).get_area() > 0]
    used = set()
    groups = []
    for region in regions:
        if region.get("type") in ("image", "figure", "exclude", "excluded"):
            continue
        box = rect_of(region)
        selected = []
        for i, line in enumerate(lines):
            if i not in used and box.contains((rect_of(line).tl + rect_of(line).br) / 2):
                selected.append(line)
                used.add(i)
        vertical = region.get("orientation") == "vertical" or (
            region.get("orientation") != "horizontal" and
            (sum(bool(l.get("is_vertical")) for l in selected) > len(selected) / 2))
        selected.sort(key=lambda l: (-l.get("x", 0), l.get("y", 0)) if vertical else (l.get("y", 0), l.get("x", 0)))
        if not selected and box.get_area() > 0:
            # Region-only documents still remain editable; selection is coarser.
            selected = [{**{k: region.get(k, 0) for k in ("x", "y", "w", "h")},
                         "text": region.get("source_text", region.get("text", "")), "is_vertical": vertical,
                         "region_only": True}]
        groups.append((region, selected))
    body_groups = [(r, ls) for r, ls in groups if r.get("type") in (None, "body", "paragraph")]
    body_lines = [l for _, ls in body_groups for l in ls]
    # Explicit empty body is a user deletion, not a reason to revive raw OCR.
    body = data.get("body_text", "".join(r.get("text", "") for r, _ in body_groups))
    runs = []
    body_done = False
    for region, selected in groups:
        kind = region.get("type")
        if kind in (None, "body", "paragraph"):
            if not body_done and body_lines:
                runs.extend(corrected_runs(body_lines, body))
                body_done = True
            continue
        text = region.get("text", "")
        key = {"heading": "headings", "footnote": "footnotes", "table": "tables"}.get(kind)
        if key:
            records = data.get(key) or []
            related = next((v for v in records if isinstance(v, dict) and
                            str(v.get("region_id", v.get("id", ""))) == str(region.get("id"))), None)
            same_kind = [r for r, _ in groups if r.get("type") == kind]
            if related is None and len(records) == len(same_kind):
                related = records[same_kind.index(region)]
            if related is not None:
                text = related.get("text", related.get("title", text)) if isinstance(related, dict) else related
                if kind == "heading" and isinstance(related, dict):
                    text = related.get("title", text)
                if kind == "table" and isinstance(related, dict) and related.get("cells"):
                    cells = related["cells"]
                    rows = related.get("rows") or []
                    has_rows = isinstance(related.get("rows"), list)
                    known = {(int(c.get("row", 1))-1, int(c.get("column", 1))-1) for c in cells}
                    if has_rows and any((r, c) not in known for r, row in enumerate(rows) for c in range(len(row))):
                        # Newly added rows/columns have no OCR cell boxes. Retain
                        # all edited text using the table's existing line anchors.
                        text = " ".join(str(value.get("text", "") if isinstance(value, dict) else value)
                                        for row in rows for value in row)
                        if selected:
                            runs.extend(corrected_runs(selected, text))
                        elif clean_text(text):
                            raise ValueError("追加した表の文字を配置する座標がありません。")
                        continue
                    for cell in sorted(cells, key=lambda c: (c.get("row", 0), c.get("column", 0))):
                        row, col = int(cell.get("row", 1))-1, int(cell.get("column", 1))-1
                        value = rows[row][col] if 0 <= row < len(rows) and 0 <= col < len(rows[row]) else ("" if has_rows else cell.get("text", ""))
                        if isinstance(value, dict):
                            value = value.get("text", "")
                        run = {"x": cell.get("x", 0), "y": cell.get("y", 0),
                               "w": cell.get("width", cell.get("w", 0)),
                               "h": cell.get("height", cell.get("h", 0)), "text": value,
                               "is_vertical": cell.get("is_vertical", False)}
                        if clean_text(value):
                            if rect_of(run).is_empty:
                                raise ValueError("表のセルに文字座標がありません。")
                            runs.append(run)
                    continue
        if selected:
            runs.extend(corrected_runs(selected, text))
        elif clean_text(text):
            raise ValueError("本文のある領域に座標がありません。領域を確認してください。")
    if not groups:
        if lines:
            vertical = sum(bool(l.get("is_vertical")) for l in lines) > len(lines) / 2
            lines.sort(key=lambda l: (-l.get("x", 0), l.get("y", 0)) if vertical else (l.get("y", 0), l.get("x", 0)))
            runs = corrected_runs(lines, data.get("body_text", "".join(l.get("text", "") for l in lines)))
        elif clean_text(body):
            raise ValueError("本文の座標がありません。領域を設定してから保存してください。")
    elif clean_text(body) and not body_lines:
        raise ValueError("本文に対応する領域がありません。領域を確認してください。")
    return runs


def add_run(page, run, width, height, fontname="OCRUnicode"):
    text = logical_text(run.get("text"))
    if not text:
        return
    missing = next((c for c in text if not c.isspace() and not OCR_FONT.has_glyph(ord(c), fallback=False)), None)
    if missing is not None:
        raise ValueError(f"文字「{missing}」（U+{ord(missing):04X}）をPDFに保存できるフォントがありません。生データは保持します。")
    box = rect_of(run)
    sx, sy = page.rect.width / width, page.rect.height / height
    box = fitz.Rect(box.x0 * sx, box.y0 * sy, box.x1 * sx, box.y1 * sy) & page.rect
    if box.is_empty:
        raise ValueError("ページ外の文字座標があります。")
    # The invisible glyphs use a rotated horizontal text line. This preserves
    # contiguous phrase search in readers that split native CJK vertical glyphs.
    rotation = page.rotation
    transform = page.derotation_matrix
    if run.get("is_vertical"):
        size = min(box.width * .85, box.height / max(1, OCR_FONT.text_length(text, fontsize=1)))
        anchor = fitz.Point(box.x1 - size * .2, box.y0) * transform
        angle = (270 + rotation) % 360
    else:
        size = min(box.height * .75, box.width / max(1, OCR_FONT.text_length(text, fontsize=1)))
        anchor = fitz.Point(box.x0, box.y0 + size) * transform
        angle = rotation
    page.set_rotation(0)
    try:
        page.insert_font(fontname=fontname, fontbuffer=OCR_FONT_BUFFER)
        page.insert_text(anchor, text, fontname=fontname, fontsize=size, render_mode=3, rotate=angle)
    finally:
        page.set_rotation(rotation)
    stream = page.parent.xref_stream(page.get_contents()[-1])
    xref = page.get_contents()[-1]
    logical = (b"\xfe\xff" + text.encode("utf-16-be")).hex().encode("ascii")
    page.parent.update_stream(xref, b"/Span << /ActualText <" + logical + b"> >> BDC\n" + stream + b"\nEMC\n")


def read_edit_data(doc):
    if ATTACHMENT not in doc.embfile_names():
        return None
    info = doc.embfile_info(ATTACHMENT)
    if info["size"] > MAX_EDIT_BYTES:
        raise ValueError("PDF内の編集データが上限を超えています。")
    raw = doc.embfile_get(ATTACHMENT)
    if len(raw) > MAX_EDIT_BYTES:
        raise ValueError("PDF内の編集データが上限を超えています。")
    data = json.loads(raw)
    if data.get("format") != "MyNotebookLM-OCR" or data.get("version") != 1 or data.get("page_count") != len(doc):
        raise ValueError("PDF内の編集データの形式またはページ数が一致しません。")
    if not isinstance(data.get("pages"), dict) or not isinstance(data.get("streams"), dict):
        raise ValueError("PDF内の編集データを読み取れません。")
    for number, page in data["pages"].items():
        if not number.isdigit() or not 1 <= int(number) <= len(doc) or not isinstance(page, dict):
            raise ValueError("PDF内の編集ページ番号が不正です。")
    return data


def page_signature(page):
    # Low-resolution raster digest is small yet detects visibility changes.
    return digest(page.get_pixmap(matrix=fitz.Matrix(.5, .5), alpha=False).samples)


def optimize_lossless_images(doc):
    """Store repeated gray RGB channels once, preserving their exact color map.

    Indexed color retains the original RGB / ICC color space as its base. Each
    gray value maps to the same three channel values as before; no thresholding,
    resampling or lossy image encoding is involved. Existing bilevel scan images
    and color photographs keep their original encoding. When indexed color would
    alter rendering, recompress the exact original RGB bytes instead.
    """
    images = {}
    for number, page in enumerate(doc):
        for image in page.get_images(full=True):
            if image[0]:
                images.setdefault(image[0], set()).add(number)
    palette = bytes(value for gray in range(256) for value in (gray, gray, gray)).hex()
    saved_bytes = 0
    for xref, numbers in images.items():
        if (doc.xref_get_key(xref, "Filter") != ("name", "/FlateDecode") or
                doc.xref_get_key(xref, "BitsPerComponent") != ("int", "8") or
                any(doc.xref_get_key(xref, key)[0] != "null" for key in
                    ("Decode", "DecodeParms", "Mask", "SMask"))):
            continue
        color_type, color_space = doc.xref_get_key(xref, "ColorSpace")
        if color_type not in ("name", "xref", "array") or "/Indexed" in color_space:
            continue
        pix = fitz.Pixmap(doc, xref)
        if pix.n != 3 or pix.alpha or pix.colorspace.n != 3:
            continue
        samples = pix.samples
        gray = samples[::3]
        if gray != samples[1::3] or gray != samples[2::3]:
            continue
        # The stream must be plain RGB samples, without a predictor or decode map.
        if doc.xref_stream(xref) != samples:
            continue
        original_stream = doc.xref_stream_raw(xref)
        compressed = zlib.compress(gray, 9)
        if len(compressed) + 1600 >= len(original_stream):
            continue
        original_object = doc.xref_object(xref)
        original_rgb = fitz.Pixmap(fitz.csRGB, pix).samples
        signatures = {number: page_signature(doc[number]) for number in numbers}
        doc.update_stream(xref, compressed, compress=False)
        doc.xref_set_key(xref, "Filter", "/FlateDecode")
        doc.xref_set_key(xref, "ColorSpace", f"[/Indexed {color_space} 255 <{palette}>]")
        optimized_rgb = fitz.Pixmap(fitz.csRGB, fitz.Pixmap(doc, xref)).samples
        if (optimized_rgb != original_rgb or
                any(page_signature(doc[number]) != signature for number, signature in signatures.items())):
            recompressed = zlib.compress(samples, 9)
            replacement = recompressed if len(recompressed) < len(original_stream) else original_stream
            doc.update_stream(xref, replacement, compress=False)
            # update_stream removes /Filter; restore the original dictionary last.
            doc.update_object(xref, original_object)
            doc.xref_set_key(xref, "Length", str(len(replacement)))
            saved_bytes += len(original_stream) - len(replacement)
            continue
        saved_bytes += len(original_stream) - len(compressed) - 1600
    return saved_bytes


def validate_pdf(path, expected, signatures, expected_text):
    with fitz.open(path) as doc:
        saved = read_edit_data(doc)
        if saved != expected:
            raise ValueError("PDFの編集データの保存確認に失敗しました。生データは保持します。")
        for number, signature in signatures.items():
            page = doc[int(number) - 1]
            if page_signature(page) != signature:
                raise ValueError(f"P.{number}の表示確認に失敗しました。生データは保持します。")
            extracted = clean_text(page.get_text(sort=False))
            # Whitespace differs between PDF extractors, content must not.
            if extracted != expected_text[number]:
                raise ValueError(f"P.{number}の文字検証に失敗しました。生データは保持します。")
            records = expected["streams"].get(number, [])
            for record in records:
                if digest(doc.xref_stream(record["xref"])) != record["sha256"]:
                    raise ValueError("PDFの文字層の検証に失敗しました。")


class OcrPdfArchive:
    def __init__(self, ocr_dir, archive_dir, cache_dir):
        self.ocr_dir = Path(ocr_dir).absolute()
        self.archive_dir = Path(archive_dir).absolute()
        self.cache_dir = Path(cache_dir).absolute()
        self._verified = {}
        self._search_text = {}

    def storage_folders(self):
        # Read the paths actually used by this running app, including after moving PCs.
        return {"status": "success", "items": [
            {"id": "archives", "label": "透明テキスト付きPDF", "path": str(self.archive_dir)},
            {"id": "raw", "label": "OCR生データ", "path": str(self.ocr_dir)}]}

    def open_storage_folder(self, kind):
        folders = {item["id"]: item for item in self.storage_folders()["items"]}
        if kind not in folders:
            raise ValueError("保存フォルダの指定が不正です。")
        item = folders[kind]
        opener = getattr(os, "startfile", None)
        if opener is None:
            raise ValueError("このPCでは保存フォルダを直接開けません。表示されたパスから開いてください。")
        path = Path(item["path"])
        path.mkdir(parents=True, exist_ok=True)
        opener(str(path))
        return {"status": "success", **item}

    def source_dir(self, filename):
        if not filename or Path(filename).name != filename or any(c in filename for c in ("/", "\\", ":")):
            raise ValueError("ファイル名が不正です。")
        path = self.ocr_dir / Path(filename).stem
        self.check_path(path, self.ocr_dir)
        return path

    @staticmethod
    def check_path(path, root):
        path, root = Path(path), Path(root)
        if path == root or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("保存範囲外のパスです。")
        current = path
        while current != root:
            if current.exists() and (current.is_symlink() or current.is_junction()):
                raise ValueError("リンクを含むフォルダーは処理できません。")
            current = current.parent
        if root.is_symlink() or root.is_junction():
            raise ValueError("リンクを含む保存先は処理できません。")

    def inventory(self, source):
        result = {}
        if not source.exists():
            return result
        for folder, dirs, files in os.walk(source, followlinks=False):
            for name in dirs + files:
                path = Path(folder) / name
                self.check_path(path, self.ocr_dir)
                if path.is_file():
                    st = path.stat()
                    result[path.relative_to(source).as_posix()] = [st.st_size, st.st_mtime_ns, self.file_hash(path)]
        return result

    @staticmethod
    def file_hash(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for part in iter(lambda: f.read(1024 * 1024), b""):
                h.update(part)
        return h.hexdigest()

    @staticmethod
    def document_stem(filename):
        stem = re.sub(r"_OCR$", "", Path(filename).stem)
        stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).rstrip(" .") or "文書"
        if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", stem):
            stem = "_" + stem
        return stem

    def document_folder(self, filename):
        stem = self.document_stem(filename)
        folder = self.archive_dir / stem
        suffix = 2
        while folder.exists():
            folder = self.archive_dir / f"{stem}（{suffix}）"
            suffix += 1
        self.check_path(folder, self.archive_dir)
        return folder

    def read_receipt(self, folder):
        self.check_path(folder, self.archive_dir)
        receipt_path = folder / "receipt.json"
        self.check_path(receipt_path, folder)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("PDFの保存記録が不正です。")
        archive_id = receipt.get("archive_id", folder.name)
        if not isinstance(archive_id, str) or not re.fullmatch(r"[a-f0-9]{32}", archive_id):
            raise ValueError("保存IDが不正です。")
        return archive_id, receipt

    def receipts(self):
        if not self.archive_dir.exists():
            return
        for folder in sorted(self.archive_dir.iterdir(), key=lambda p: p.stat().st_mtime_ns, reverse=True):
            try:
                archive_id, receipt = self.read_receipt(folder)
                yield folder, archive_id, receipt
            except (ValueError, OSError, KeyError, json.JSONDecodeError):
                continue

    def receipt_pdf(self, folder, receipt):
        name = receipt["output_name"]
        if Path(name).name != name or any(c in name for c in ("/", "\\", ":")):
            raise ValueError("保存名が不正です。")
        path = folder / name
        self.check_path(path, self.archive_dir)
        return path, receipt

    def archive_path(self, archive_id):
        if not re.fullmatch(r"[a-f0-9]{32}", archive_id or ""):
            raise ValueError("保存IDが不正です。")
        # IDs belong to the receipt, so a readable folder can be renamed without
        # breaking existing download links or raw-data deletion authorization.
        for folder, saved_id, receipt in self.receipts():
            if saved_id == archive_id:
                return self.receipt_pdf(folder, receipt)
        raise ValueError("保存済みPDFが見つかりません。")

    def result(self, archive_id, path, receipt, operation):
        return {"status": "success", "archive_id": archive_id,
                "download_url": f"/api/batch/ocr-pdf/{archive_id}", "saved_path": str(path),
                "output_name": receipt["output_name"], "page_count": receipt["text_page_count"],
                "edit_page_count": receipt["page_count"], "total_pages": receipt["total_pages"],
                "corrected_pages": receipt["corrected_pages"], "size_bytes": path.stat().st_size,
                "operation": operation}

    def verified_pdf(self, archive_id, candidate=None):
        path, receipt = candidate if candidate is not None else self.archive_path(archive_id)
        st = path.stat()
        stamp = (str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns, receipt["pdf_sha256"])
        if self._verified.get(archive_id) != stamp:
            if self.file_hash(path) != receipt["pdf_sha256"]:
                raise ValueError("保存済みPDFの内容が変更されています。")
            if len(self._verified) >= 128:
                self._verified.clear()
            self._verified[archive_id] = stamp
        return path, receipt, stamp

    @serialized
    def list_pdfs(self, query="", include_text=False, sort="saved", descending=True):
        if sort not in ("name", "saved"):
            raise ValueError("並べ替え方法が不正です。")
        def normalized(value):
            return "".join(unicodedata.normalize("NFKC", str(value)).casefold().split())
        terms = [normalized(v) for v in query.split() if v.strip()]
        items = []
        for folder, archive_id, receipt in self.receipts():
            try:
                path, receipt, stamp = self.verified_pdf(archive_id, self.receipt_pdf(folder, receipt))
                title = self.document_stem(receipt["output_name"])
                name = normalized(title)
                if terms and not all(t in name for t in terms):
                    if not include_text:
                        continue
                    if self._search_text.get(archive_id, (None,))[0] != stamp:
                        with fitz.open(path) as doc:
                            text = normalized("\n".join(p.get_text() for p in doc))
                        if len(self._search_text) >= 32:
                            self._search_text.clear()
                        self._search_text[archive_id] = (stamp, text)
                    if not all(t in name + self._search_text[archive_id][1] for t in terms):
                        continue
                saved = path.stat().st_mtime
                items.append({"archive_id": archive_id, "document_name": title,
                              "folder_name": folder.name, "output_name": receipt["output_name"],
                              "saved_at": datetime.fromtimestamp(saved, timezone.utc).isoformat(),
                              "saved_timestamp": saved, "total_pages": receipt["total_pages"],
                              "text_page_count": receipt["text_page_count"]})
            except (ValueError, OSError, KeyError, RuntimeError):
                continue
        key = (lambda v: (normalized(v["document_name"]), v["saved_timestamp"])) if sort == "name" else (lambda v: (v["saved_timestamp"], normalized(v["document_name"])))
        return {"status": "success", "items": sorted(items, key=key, reverse=descending)}

    @serialized
    def render_archive_page(self, archive_id, number):
        path, receipt, _ = self.verified_pdf(archive_id)
        with fitz.open(path) as doc:
            if not 1 <= number <= len(doc):
                raise ValueError("ページ範囲外です。")
            pix = doc[number - 1].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            return {"filename": receipt["output_name"], "current_page": number, "total_pages": len(doc),
                    "image_width": pix.width, "image_height": pix.height,
                    "image_base64": "data:image/jpeg;base64," + base64.b64encode(pix.tobytes("jpg", jpg_quality=88)).decode("ascii")}

    @serialized
    def open_editor(self, archive_id):
        path, receipt, _ = self.verified_pdf(archive_id)
        restored = self.restore(receipt["output_name"], path=path)
        if not restored["pages"]:
            raise ValueError("PDF内に再編集用のデータがありません。")
        return {**restored, "archive_id": archive_id, "output_name": receipt["output_name"],
                "document_name": self.document_stem(receipt["output_name"]), "saved_path": str(path),
                "rendered_page": self.render_archive_page(archive_id, 1)}

    @serialized
    def save_editor(self, archive_id, pages):
        path, receipt, _ = self.verified_pdf(archive_id)
        return self.prepare(receipt["output_name"], path.read_bytes(), edit_pages=pages,
                            overwrite_archive_id=archive_id)

    def find_unchanged(self, filename, pdf_hash, pages, snapshot, target_id=None):
        fingerprint, portable = revision(pages), revision(pages, False)
        # Receipts are tiny; PDFs are hashed only after their source and revision match.
        for folder, archive_id, receipt in self.receipts():
            if target_id is not None and archive_id != target_id:
                continue
            try:
                path, receipt = self.receipt_pdf(folder, receipt)
                if self.document_stem(filename) != self.document_stem(receipt["output_name"]):
                    continue
                # Older saves baked corrected bitmaps into the page. A new save
                # must apply the current image policy before it can be reused.
                if receipt.get("page_images") not in ("original", "preserved"):
                    continue
                matched = ((receipt.get("source_pdf_sha256") == pdf_hash and receipt.get("revision") == fingerprint) or
                           (receipt.get("pdf_sha256") == pdf_hash and receipt.get("portable_revision") == portable))
                if not matched or self.file_hash(path) != receipt["pdf_sha256"]:
                    continue
                # Refresh deletion authorization against the exact current files.
                if snapshot is not None:
                    receipt["filename"] = filename
                    receipt["inventory"] = snapshot
                    receipt["editing_only"] = False
                    temporary = folder / "receipt.tmp"
                    temporary.write_bytes(encode(receipt))
                    os.replace(temporary, folder / "receipt.json")
                return self.result(archive_id, path, receipt, "unchanged")
            except (ValueError, OSError, KeyError, json.JSONDecodeError):
                continue
        return None

    @serialized
    def prepare(self, filename, pdf_bytes, edit_pages=None, overwrite_archive_id=None):
        self.source_dir(filename)  # validate even when editing a stored PDF
        target = None
        if overwrite_archive_id is not None:
            if edit_pages is None:
                raise ValueError("上書き保存には編集データが必要です。")
            target, target_receipt, _ = self.verified_pdf(overwrite_archive_id)
            if digest(pdf_bytes) != target_receipt["pdf_sha256"]:
                raise ValueError("PDFが更新されています。開き直してください。")
            original_receipt = (target.parent / "receipt.json").read_bytes()
        source = self.source_dir(filename) if edit_pages is None else None
        snapshot = self.inventory(source) if source is not None else None
        pages = {} if edit_pages is None else dict(edit_pages)
        if any(not key.isdigit() or not isinstance(value, dict) for key, value in pages.items()):
            raise ValueError("編集ページの形式が不正です。")
        for p in (sorted(source.glob("page_*/page_data.json")) if source is not None else []):
            data = json.loads(p.read_text(encoding="utf-8"))
            for index, figure in enumerate(data.get("figures") or []):
                if figure.get("is_custom_image") and not figure.get("image_base64"):
                    name = figure.get("file_name") or f"figure_{index:02d}.png"
                    if Path(name).name != name or any(c in name for c in ("/", "\\", ":")):
                        raise ValueError("差し替え図版のファイル名が不正です。")
                    image_path = p.parent / name
                    self.check_path(image_path, self.ocr_dir)
                    if not image_path.is_file():
                        raise ValueError("差し替え図版の画像が見つかりません。生データは保持します。")
                    mime = "image/jpeg" if image_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                    figure["image_base64"] = f"data:{mime};base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
            number = int(data.get("page_number") or data.get("current_page") or re.search(r"page_(\d+)", p.parent.name)[1])
            if str(number) in pages:
                raise ValueError("同じページの保存データが重複しています。")
            pages[str(number)] = data
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.needs_pass or doc.is_repaired:
                raise ValueError("暗号化または修復されたPDFは処理できません。")
            old = read_edit_data(doc)
            if old:
                # A sparse edit must retain every page from the portable archive.
                pages = {**old["pages"], **pages}
            if not pages:
                raise ValueError("保存するOCRデータがありません。")
            pdf_hash = digest(pdf_bytes)
            unchanged = self.find_unchanged(filename, pdf_hash, pages, snapshot, overwrite_archive_id)
            if unchanged:
                return unchanged
            if old:
                for key, records in old["streams"].items():
                    page = doc[int(key) - 1]
                    contents = page.get_contents()
                    for record in records:
                        xref = record["xref"]
                        if xref not in contents or digest(doc.xref_stream(xref)) != record["sha256"]:
                            raise ValueError("既存のOCR文字層が変更されています。元のOCR付きPDFを開いてください。")
                        doc.update_stream(xref, b"")
                doc.embfile_del(ATTACHMENT)
            meta = {"format": "MyNotebookLM-OCR", "version": 1, "page_count": len(doc),
                    "source_name": filename, "pages": {}, "streams": {},
                    "page_images": old.get("page_images", "preserved") if old else "original",
                    "original_pdf_sha256": old.get("original_pdf_sha256") if old else pdf_hash}
            signatures, expected_text, corrected_pages = {}, {}, []
            # A fresh resource name also avoids reusing an existing subset font
            # when an edited PDF introduces characters absent from that subset.
            fontname = "OCRUnicode" + uuid.uuid4().hex
            total_runs = 0
            text_page_count = 0
            for key, raw in sorted(pages.items(), key=lambda v: int(v[0])):
                number = int(key)
                if not 1 <= number <= len(doc):
                    raise ValueError(f"P.{number}が元PDFのページ範囲外です。")
                page = doc[number - 1]
                # Other OCR tools may already have created an invisible layer.
                # Remove it only if every existing glyph is invisible, and prove
                # that removing it leaves the rendered page unchanged.
                if clean_text(page.get_text()):
                    traces = page.get_texttrace()
                    if not traces or any(t.get("type") != 3 and t.get("opacity", 1) != 0 for t in traces):
                        raise ValueError(f"P.{number}には表示用のPDFテキストがあります。二重埋め込みを防ぐため処理を中止しました。生データは保持します。")
                    signature = page_signature(page)
                    page.add_redact_annot(page.rect, fill=False)
                    page.apply_redactions(images=0, graphics=0, text=0)
                    if page_signature(page) != signature or clean_text(page.get_text()):
                        raise ValueError(f"P.{number}の既存文字層を安全に置き換えられません。生データは保持します。")
                # Keep the original compressed page images. OCR correction
                # bitmaps are working data and are excluded from the PDF.
                # Coordinates still provide page-level search after correction.
                signatures[key] = page_signature(page)
                runs = make_runs(raw)
                width, height = raw.get("image_width"), raw.get("image_height")
                if not width or not height or not 0 < width <= 100000 or not 0 < height <= 100000:
                    if runs:
                        # Text placement must use its actual OCR coordinate size.
                        # Do not guess a scale for existing recognized text.
                        raise ValueError(f"P.{number}の画像サイズが不明です。")
                    # Older manually selected figure / blank pages can contain
                    # no OCR image metadata. They have no text to position; keep
                    # the PDF image and restore the app's zoom=2 canvas size for
                    # subsequent figure extraction and portable editing.
                    canvas = (page.rect * fitz.Matrix(2, 2)).irect
                    width, height = canvas.width, canvas.height
                    raw = {**raw, "image_width": width, "image_height": height}
                if runs:
                    text_page_count += 1
                before = set(page.get_contents())
                for run in runs:
                    add_run(page, run, width, height, fontname)
                meta["streams"][key] = [{"xref": x, "sha256": digest(doc.xref_stream(x))}
                                         for x in page.get_contents() if x not in before]
                meta["pages"][key] = compact(raw)
                expected_text[key] = "".join(clean_text(r["text"]) for r in runs)
                total_runs += len(runs)
            if not total_runs and not old and not self.has_prior(pdf_hash, ""):
                raise ValueError("埋め込むOCR本文がありません。生データは保持します。")
            raw_meta = encode(meta)
            if len(raw_meta) > MAX_EDIT_BYTES:
                raise ValueError("編集データが保存上限を超えています。")
            doc.embfile_add(ATTACHMENT, raw_meta, filename=ATTACHMENT, desc="OCR本文・座標・読み順（編集用）")
            archive_id = overwrite_archive_id or uuid.uuid4().hex
            folder = target.parent if target is not None else self.document_folder(filename)
            folder.mkdir(parents=True, exist_ok=target is not None)
            output_name = target.name if target is not None else self.document_stem(filename) + "_OCR.pdf"
            output = folder / output_name
            temp = folder / (".building_" + uuid.uuid4().hex + ".pdf" if target is not None else "building.pdf")
            # Keep xrefs stable: they identify only our generated streams on re-edit.
            try:
                doc.save(temp, garbage=1, deflate=True)
            except Exception:
                if target is not None:
                    temp.unlink(missing_ok=True)
                raise
        try:
            validate_pdf(temp, meta, signatures, expected_text)
            if source is not None and self.inventory(source) != snapshot:
                raise ValueError("保存中にOCRデータが更新されました。生データを残して中止しました。")
            receipt = {"archive_id": archive_id, "filename": filename, "output_name": output_name, "pdf_sha256": self.file_hash(temp),
                   "inventory": snapshot, "editing_only": source is None,
                   "page_count": len(meta["pages"]), "source_pdf_sha256": pdf_hash,
                   "revision": revision(pages), "portable_revision": revision(meta["pages"], False),
                   "text_page_count": text_page_count, "total_pages": meta["page_count"],
                   "corrected_pages": corrected_pages, "page_images": meta["page_images"],
                   "original_pdf_sha256": meta["original_pdf_sha256"]}
            if target is not None:
                self.replace_archive(output, temp, receipt, original_receipt, pdf_hash)
            else:
                os.replace(temp, output)
                (folder / "receipt.json").write_bytes(encode(receipt))
        finally:
            if target is not None:
                temp.unlink(missing_ok=True)
        self._verified.pop(archive_id, None)
        self._search_text.pop(archive_id, None)
        return self.result(archive_id, output, receipt, "updated" if old or self.has_prior(pdf_hash, archive_id) else "created")

    def replace_archive(self, output, temp, receipt, original_receipt, original_hash):
        """Commit a verified PDF and its receipt, rolling back if receipt saving fails."""
        receipt_path = output.parent / "receipt.json"
        token = uuid.uuid4().hex
        staged_receipt = output.parent / (".receipt_" + token + ".tmp")
        rollback = output.parent / (".rollback_" + token + ".pdf")
        keep_rollback = False
        try:
            staged_receipt.write_bytes(encode(receipt))
            if self.file_hash(output) != original_hash or receipt_path.read_bytes() != original_receipt:
                raise ValueError("保存中にPDFが更新されました。開き直してください。")
            # Keep a temporary recovery copy until both files have been committed.
            shutil.copyfile(output, rollback)
            if self.file_hash(rollback) != original_hash or self.file_hash(output) != original_hash:
                raise ValueError("保存中にPDFが更新されました。開き直してください。")
            os.replace(temp, output)
            try:
                os.replace(staged_receipt, receipt_path)
            except OSError as exc:
                try:
                    os.replace(rollback, output)
                except OSError as rollback_error:
                    keep_rollback = True
                    raise OSError(f"保存情報の更新に失敗しました。元PDFの復旧用ファイルを保持しました: {rollback}") from rollback_error
                raise OSError("保存情報を更新できなかったため、元のPDFに戻しました。再度保存してください。") from exc
        finally:
            staged_receipt.unlink(missing_ok=True)
            if not keep_rollback:
                rollback.unlink(missing_ok=True)

    def has_prior(self, pdf_hash, exclude_id):
        # Used only for the UI wording; never authorizes deletion or reuse.
        for folder, archive_id, r in self.receipts():
            if archive_id == exclude_id:
                continue
            if r.get("source_pdf_sha256") == pdf_hash and r.get("pdf_sha256") != pdf_hash:
                return True
        return False

    @serialized
    def delete_raw(self, filename, archive_id):
        source = self.source_dir(filename)
        path, receipt = self.archive_path(archive_id)
        if receipt.get("editing_only"):
            raise ValueError("透明テキスト編集で保存したPDFは、生データ削除の対象ではありません。")
        if receipt["filename"] != filename or self.file_hash(path) != receipt["pdf_sha256"]:
            raise ValueError("検証済みPDFが確認できません。生データは保持します。")
        if self.inventory(source) != receipt["inventory"]:
            raise ValueError("PDF作成後にOCRデータが更新されました。PDFを作り直してください。生データは保持します。")
        if not source.is_dir():
            raise ValueError("削除対象のOCRデータがありません。")
        self.check_path(source, self.ocr_dir)
        # Keep original PDFs and exported documents even if an older version put
        # them inside the raw-data folder. Only known intermediate formats go.
        raw_suffixes = {".json", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".txt", ".xml", ".npy", ".npz"}
        for name in receipt["inventory"]:
            item = source / name
            self.check_path(item, source)
            if item.suffix.lower() in raw_suffixes:
                item.unlink()
        for folder, dirs, files in os.walk(source, topdown=False, followlinks=False):
            item = Path(folder)
            self.check_path(item, self.ocr_dir)
            if not any(item.iterdir()):
                item.rmdir()
        return {"status": "success", "deleted": True, "filename": filename,
                "deleted_pages_count": receipt["page_count"], "saved_path": str(path)}

    def restore(self, filename, path=None):
        self.source_dir(filename)  # validate filename before reading cache
        if path is None:
            path = self.cache_dir / (Path(filename).stem + ".pdf")
            self.check_path(path, self.cache_dir)
        else:
            self.check_path(path, self.archive_dir)
        with fitz.open(path) as doc:
            meta = read_edit_data(doc)
            if meta is None:
                return {"status": "success", "pages": {}}
            pages = meta["pages"]
            for key, data in pages.items():
                data["filename"] = filename
                data["page_number"] = int(key)
                data["current_page"] = int(key)
                data["archive_restored"] = True
                # Ordinary figure crops can be regenerated from the PDF, not stored twice.
                for figure in data.get("figures") or []:
                    if figure.get("image_base64"):
                        continue
                    box = rect_of(figure)
                    if box.is_empty:
                        continue
                    page = doc[int(key) - 1]
                    sx = page.rect.width / data["image_width"]
                    sy = page.rect.height / data["image_height"]
                    clip = fitz.Rect(box.x0*sx, box.y0*sy, box.x1*sx, box.y1*sy) & page.rect
                    if not clip.is_empty:
                        pix = page.get_pixmap(matrix=fitz.Matrix(1/sx, 1/sy), clip=clip, alpha=False)
                        figure["image_base64"] = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode("ascii")
            return {"status": "success", "pages": pages, "source": "pdf"}


def create_archive_router(service):
    router = APIRouter()

    @router.get("/api/batch/storage-folders")
    def storage_folders():
        return service.storage_folders()

    @router.post("/api/batch/storage-folders/{kind}/open")
    def open_storage_folder(kind: str):
        try:
            return service.open_storage_folder(kind)
        except (ValueError, OSError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.get("/api/batch/ocr-pdfs")
    def list_pdfs(q: str = "", include_text: bool = False, sort: str = "saved", descending: bool = True):
        try:
            return service.list_pdfs(q, include_text, sort, descending)
        except ValueError as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.get("/api/batch/ocr-pdfs/{archive_id}/edit-data")
    def edit_data(archive_id: str):
        try:
            return service.open_editor(archive_id)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.get("/api/batch/ocr-pdfs/{archive_id}/page/{number}")
    def render(archive_id: str, number: int):
        try:
            return service.render_archive_page(archive_id, number)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.post("/api/batch/ocr-pdfs/{archive_id}/edit")
    def save_edit(archive_id: str, request: PdfEditRequest):
        try:
            return service.save_editor(archive_id, request.pages)
        except (ValueError, OSError, RuntimeError, KeyError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.post("/api/batch/inspect-ocr-pdf")
    async def inspect_pdf(file: UploadFile = File(...)):
        content = await file.read()
        def inspect():
            with fitz.open(stream=content, filetype="pdf") as doc:
                data = read_edit_data(doc)
                return {"status": "success", "editable": data is not None,
                        "page_count": len(data["pages"]) if data else 0}
        try:
            return await run_in_threadpool(inspect)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(400, detail="編集用PDFを読み取れません。") from exc

    @router.post("/api/batch/archive-pdf")
    async def archive_pdf(file: UploadFile = File(...), filename: str = Form(...)):
        try:
            content = await file.read()
            return await run_in_threadpool(service.prepare, filename, content)
        except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

    @router.get("/api/batch/ocr-pdf/{archive_id}")
    def download(archive_id: str):
        try:
            path, receipt = service.archive_path(archive_id)
            return FileResponse(path, media_type="application/pdf", filename=receipt["output_name"])
        except (ValueError, OSError) as exc:
            raise HTTPException(404, detail="保存済みPDFが見つかりません。") from exc

    @router.get("/api/batch/pdf-edit-data")
    def restore(filename: str):
        try:
            return service.restore(filename)
        except (ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
            raise HTTPException(400, detail="PDF内の編集データを復元できません。" + str(exc)) from exc

    return router
