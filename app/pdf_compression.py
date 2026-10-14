"""Image-only PDF compression with stable OCR xrefs and recoverable replacement."""
import base64
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import uuid

import pymupdf as fitz
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.ocr_pdf_archive import STORAGE_LOCK, read_edit_data, encode


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checked_path(value):
    path = Path(value).absolute()
    if path.suffix.lower() != '.pdf' or not path.is_file():
        raise ValueError('存在するPDFファイルを選択してください。')
    for item in (path, *path.parents):
        if item.is_symlink() or (hasattr(item, 'is_junction') and item.is_junction()):
            raise ValueError('リンクを含まないフォルダーのPDFを選択してください。')
    return path.resolve()


def verify_ocr(doc):
    data = read_edit_data(doc)
    if data:
        for key, records in data['streams'].items():
            contents = doc[int(key) - 1].get_contents()
            for record in records:
                xref = record['xref']
                if xref not in contents or hashlib.sha256(doc.xref_stream(xref)).hexdigest() != record['sha256']:
                    raise ValueError('OCR編集用の文字層を確認できません。元のPDFは変更しません。')
    return data


def fingerprint(doc):
    """Exact text instructions, attachments and geometry; ignore image streams."""
    pages = []
    for page in doc:
        pages.append((tuple(page.mediabox), tuple(page.cropbox), page.rotation,
                      hashlib.sha256(page.get_text().encode('utf-8')).hexdigest(),
                      [(xref, hashlib.sha256(doc.xref_stream(xref)).hexdigest()) for xref in page.get_contents()]))
    attachments = {name: hashlib.sha256(doc.embfile_get(name)).hexdigest() for name in doc.embfile_names()}
    verify_ocr(doc)
    return pages, attachments


def compress_pdf(source, target, progress=lambda *args: None, cancelled=lambda: False):
    with fitz.open(source) as doc:
        if doc.needs_pass or doc.is_repaired:
            raise ValueError('パスワード付き、または修復が必要なPDFは処理できません。')
        before = fingerprint(doc)
        images = {}
        for number, page in enumerate(doc):
            for image in page.get_images(full=True):
                if image[0]:
                    images.setdefault(image[0], set()).add(number)
        changed = 0
        for index, (xref, numbers) in enumerate(images.items(), 1):
            if cancelled():
                raise ValueError('圧縮をキャンセルしました。元のPDFは変更していません。')
            progress(index, len(images), '画像を圧縮中')
            if any(doc.xref_get_key(xref, key)[0] != 'null' for key in ('SMask', 'Mask')):
                continue
            pix = fitz.Pixmap(doc, xref)
            if pix.alpha or not pix.colorspace:
                continue
            if pix.colorspace.n != 1:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image = Image.frombytes('L' if pix.colorspace.n == 1 else 'RGB', (pix.width, pix.height), pix.samples)
            boxes = [box for number in numbers for box in doc[number].get_image_rects(xref)]
            if not boxes:
                continue
            def resized(dpi):
                scale = min(1, max(max(box.width * dpi / 72 / image.width, box.height * dpi / 72 / image.height) for box in boxes))
                return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
            replacement = resized(96)
            buffer = io.BytesIO()
            replacement.save(buffer, format='JPEG', quality=45, optimize=True)
            data, encoding = buffer.getvalue(), 'jpeg'
            if doc.xref_get_key(xref, 'BitsPerComponent') == ('int', '1') and image.mode == 'L':
                mono = resized(150).point(lambda value: 255 if value > 150 else 0, mode='1')
                tiff = io.BytesIO()
                mono.save(tiff, format='TIFF', compression='group4', strip_size=1024 * 1024 * 1024)
                encoded = tiff.getvalue()
                with Image.open(io.BytesIO(encoded)) as tiff_image:
                    offsets, lengths = tiff_image.tag_v2[273], tiff_image.tag_v2[279]
                if len(offsets) == 1:
                    group4 = encoded[offsets[0]:offsets[0] + lengths[0]]
                    if len(group4) + 200 < len(data):
                        data, replacement, encoding = group4, mono, 'group4'
            if len(data) + 200 >= len(doc.xref_stream_raw(xref)):
                continue
            color = '/DeviceGray' if replacement.mode in ('1', 'L') else '/DeviceRGB'
            bits = 1 if encoding == 'group4' else 8
            doc.update_object(xref, f'<< /Type /XObject /Subtype /Image /Width {replacement.width} /Height {replacement.height} /ColorSpace {color} /BitsPerComponent {bits} >>')
            doc.update_stream(xref, data, compress=False)
            doc.xref_set_key(xref, 'Filter', '/CCITTFaxDecode' if encoding == 'group4' else '/DCTDecode')
            if encoding == 'group4':
                doc.xref_set_key(xref, 'DecodeParms', f'<< /K -1 /Columns {replacement.width} /Rows {replacement.height} /BlackIs1 true >>')
                if fitz.Pixmap(doc, xref).samples != replacement.convert('L').tobytes():
                    raise ValueError('白黒画像の圧縮結果を確認できません。')
            changed += 1
        if cancelled():
            raise ValueError('圧縮をキャンセルしました。元のPDFは変更していません。')
        progress(len(images), len(images), '透明テキスト・編集情報を検証中')
        # Higher garbage levels renumber the OCR xrefs stored in the attachment.
        doc.save(target, garbage=1, deflate=True)
    with fitz.open(target) as result:
        if fingerprint(result) != before:
            raise ValueError('透明テキストまたは編集情報が一致しません。元のPDFは変更しません。')
    return {'images_changed': changed, 'pages': len(before[0]), 'editable': bool(before[1].get('MyNotebookLM-OCR-edit-v1.json'))}


def exclusive_backup(source, destination):
    """Hard link is released by atomic replacement; copying works on other volumes."""
    try:
        os.link(source, destination)
    except FileExistsError:
        raise ValueError(f'退避先がすでにあります。上書きしません: {destination.name}') from None
    except OSError:
        try:
            with open(source, 'rb') as original, open(destination, 'xb') as backup:
                shutil.copyfileobj(original, backup)
                backup.flush()
                os.fsync(backup.fileno())
        except FileExistsError:
            raise ValueError(f'退避先がすでにあります。上書きしません: {destination.name}') from None
        except Exception:
            destination.unlink(missing_ok=True)
            raise


def replace_compressed(source, candidate, original_hash, archive, details):
    backup = source.with_name(source.stem + 'old' + source.suffix)
    if os.path.lexists(backup):
        raise ValueError(f'{backup.name} がすでにあります。別の場所へ移動してから実行してください。')
    if file_hash(source) != original_hash:
        raise ValueError('処理中に元PDFが更新されました。選び直してください。')
    receipt_path = source.parent / 'receipt.json'
    old_receipt = None
    if receipt_path.exists():
        if receipt_path.is_symlink():
            raise ValueError('管理情報がリンクのため処理できません。')
        raw = receipt_path.read_bytes()
        try:
            receipt = json.loads(raw)
        except ValueError:
            raise ValueError('同じフォルダーのreceipt.jsonを読み取れません。') from None
        if receipt.get('output_name') == source.name:
            if receipt.get('pdf_sha256') != original_hash:
                raise ValueError('PDFと管理情報が一致しません。元のPDFは変更しません。')
            old_receipt = raw
    receipt_backup = source.parent / 'receiptold.json'
    if old_receipt is not None and os.path.lexists(receipt_backup):
        raise ValueError('receiptold.json がすでにあります。別の場所へ移動してから実行してください。')
    backup_created = receipt_created = False
    try:
        exclusive_backup(source, backup)
        backup_created = True
        if file_hash(backup) != original_hash or file_hash(source) != original_hash:
            raise ValueError('元PDFの退避を確認できません。元のPDFは変更しません。')
        if old_receipt is not None:
            with receipt_backup.open('xb') as stream:
                receipt_created = True
                stream.write(old_receipt)
                stream.flush()
                os.fsync(stream.fileno())
            updated = copy.deepcopy(receipt)
            updated['pdf_sha256'] = file_hash(candidate)
            updated['compression'] = {'jpeg_dpi': 96, 'bilevel_dpi': 150, 'jpeg_quality': 45,
                                      'original_sha256': original_hash, 'original_backup': backup.name,
                                      'text_and_edit_data_preserved': True}
            archive.replace_archive(source, candidate, updated, old_receipt, original_hash)
            archive._verified.clear()
            archive._search_text.clear()
        else:
            if file_hash(source) != original_hash:
                raise ValueError('元PDFが更新されました。選び直してください。')
            os.replace(candidate, source)
    except Exception:
        # Never remove the only remaining original if replacement/rollback failed.
        if backup_created and source.exists() and file_hash(source) == original_hash:
            backup.unlink(missing_ok=True)
            if receipt_created:
                receipt_backup.unlink(missing_ok=True)
        raise
    return {'saved_path': str(source), 'backup_path': str(backup), 'receipt_updated': old_receipt is not None, **details}


def choose_pdf(initial_dir):
    from app.runtime_paths import windows_powershell
    powershell = windows_powershell()
    if not powershell:
        raise ValueError('ファイル選択画面を開けません。Windowsで実行してください。')
    escaped = str(initial_dir).replace("'", "''")
    script = """Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = '圧縮するPDFを選択'
$dialog.Filter = 'PDFファイル (*.pdf)|*.pdf'
$dialog.InitialDirectory = '%s'
$dialog.Multiselect = $false
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
try { if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
 [Console]::Write([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($dialog.FileName)))
} } finally { $dialog.Dispose(); $owner.Dispose() }
""" % escaped
    result = subprocess.run([powershell, '-NoProfile', '-STA', '-EncodedCommand', base64.b64encode(script.encode('utf-16-le')).decode()],
                            capture_output=True, timeout=300, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    if result.returncode:
        raise ValueError('PDF選択画面を開けませんでした。')
    return base64.b64decode(result.stdout.strip()).decode('utf-8') if result.stdout.strip() else ''


class PdfCompression:
    def __init__(self, archive, gate):
        self.archive, self.gate = archive, gate
        self.lock = threading.Lock()
        self.jobs, self.selections = {}, {}
        self.active = None
        self.picking = False

    def is_busy(self):
        with self.lock:
            return self.active is not None or self.picking

    def select(self):
        with self.lock:
            if self.active or self.picking:
                raise ValueError('PDF圧縮またはファイル選択が進行中です。')
            self.picking = True
        try:
            value = choose_pdf(self.archive.archive_dir)
            if not value:
                return {'cancelled': True}
            source = checked_path(value)
            backup = source.with_name(source.stem + 'old' + source.suffix)
            if os.path.lexists(backup):
                raise ValueError(f'{backup.name} がすでにあります。上書きせず停止しました。')
            original = file_hash(source)
            token = secrets.token_urlsafe(32)
            with self.lock:
                self.selections = {token: (source, original, time.monotonic())}
            return {'selection': token, 'path': str(source), 'name': source.name,
                    'backup_name': backup.name, 'size_bytes': source.stat().st_size}
        finally:
            with self.lock:
                self.picking = False

    def start(self, token):
        # The gate lease outlives the HTTP response and blocks backup/restore.
        with self.gate.lock:
            if self.gate.reason:
                raise ValueError(self.gate.reason)
            with self.lock:
                if self.active:
                    raise ValueError('別のPDFを圧縮中です。')
                selected = self.selections.get(token)
                if not selected or time.monotonic() - selected[2] > 1800:
                    raise ValueError('PDFを選び直してください。')
                job = uuid.uuid4().hex
                self.jobs = {job: {'id': job, 'state': 'running', 'message': 'PDFを確認中', 'progress': 0}}
                self.active = job
                self.selections.clear()
                self.gate.active += 1
        try:
            threading.Thread(target=self.run, args=(job, *selected[:2]), daemon=True).start()
        except Exception:
            with self.lock:
                self.active = None
            with self.gate.lock:
                self.gate.active -= 1
            raise
        return {'job_id': job}

    def update(self, job, **values):
        with self.lock:
            self.jobs[job].update(values)

    def status(self, job):
        with self.lock:
            if job not in self.jobs:
                raise ValueError('圧縮処理が見つかりません。アプリを再起動した場合は保存フォルダーをご確認ください。')
            return dict(self.jobs[job])

    def cancel(self, job):
        with self.lock:
            if job not in self.jobs:
                raise ValueError('圧縮処理が見つかりません。')
            if self.jobs[job]['state'] == 'running' and not self.jobs[job].get('committing'):
                self.jobs[job]['cancelled'] = True
        return {'status': 'success'}

    def run(self, job, source, original_hash):
        candidate = source.parent / ('.compress_' + uuid.uuid4().hex + '.pdf')
        try:
            with STORAGE_LOCK:
                source = checked_path(source)
                if file_hash(source) != original_hash:
                    raise ValueError('選択後にPDFが更新されました。選び直してください。')
                def cancelled():
                    return self.status(job).get('cancelled', False)
                def progress(index, total, label):
                    self.update(job, progress=round(index / max(total, 1) * 85), message=f'{label}（{index}/{total}）')
                details = compress_pdf(source, candidate, progress, cancelled)
                if cancelled():
                    raise ValueError('圧縮をキャンセルしました。元のPDFは変更していません。')
                before, after = source.stat().st_size, candidate.stat().st_size
                if after >= before:
                    self.update(job, state='completed', message='これ以上小さくならないため、元のPDFを保持しました。oldファイルは作成していません。', result={'changed': False, 'before_bytes': before, 'after_bytes': before})
                    return
                # Commit is short and cannot be cancelled between PDF and receipt.
                with self.lock:
                    if self.jobs[job].get('cancelled'):
                        raise ValueError('圧縮をキャンセルしました。元のPDFは変更していません。')
                    self.jobs[job]['committing'] = True
                self.update(job, progress=95, message='元ファイルをoldへ退避し、圧縮PDFを保存中')
                result = replace_compressed(source, candidate, original_hash, self.archive, details)
                result.update(changed=True, before_bytes=before, after_bytes=after, reduction_percent=round((1 - after / before) * 100, 1))
                self.update(job, state='completed', progress=100, message='圧縮・保存が完了しました。', result=result)
        except Exception as exc:
            self.update(job, state='failed', message=str(exc) if isinstance(exc, (ValueError, OSError)) else '圧縮に失敗しました。元のPDFとoldファイルをご確認ください。')
        finally:
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
            finally:
                with self.lock:
                    self.active = None
                with self.gate.lock:
                    self.gate.active -= 1


class CompressionRequest(BaseModel):
    selection: str


def create_compression_router(service):
    router = APIRouter(prefix='/api/batch/pdf-compression')
    def action(function, *args):
        try:
            return function(*args)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            raise HTTPException(400, str(exc)) from None
    @router.post('/select')
    def select():
        return action(service.select)
    @router.post('/start')
    def start(request: CompressionRequest):
        return action(service.start, request.selection)
    @router.get('/{job}')
    def status(job: str):
        return action(service.status, job)
    @router.post('/{job}/cancel')
    def cancel(job: str):
        return action(service.cancel, job)
    return router
