"""Verified folder backups and offline, recoverable restoration on next startup."""
import base64
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import threading
import time
import uuid

DATABASES = ('my_notebook.db', 'knowledge.db', 'research.db', 'research_index.db')
DIRECTORIES = ('media', 'ocr_archives', 'tcy_templates')
SETTINGS = ('user_dictionary.json', 'tcy_registry.json')
SCOPES = DATABASES + DIRECTORIES + SETTINGS
FORMAT = 'MyNotebookLM-data-backup'


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def linked(path):
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def contained(path, parent):
    resolved = path.resolve()
    if resolved == parent.resolve() or not resolved.is_relative_to(parent.resolve()):
        raise ValueError('指定された場所を安全に確認できません。')
    return resolved


def remove_owned(path, parent):
    contained(path, parent)
    if linked(path):
        raise ValueError('リンクされたフォルダは処理できません。')
    if path.is_dir(): shutil.rmtree(path)
    elif path.exists(): path.unlink()


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024): value.update(chunk)
    return value.hexdigest()


def data_files(data):
    """Explicit scope excludes credentials, temporary files and OCR raw data."""
    if linked(data): raise ValueError('データ保存先がリンクのため処理できません。')
    result = []
    for name in SCOPES:
        path = data / name
        if not path.exists(): continue
        if linked(path): raise ValueError('保存対象にリンクがあります。通常のフォルダを使用してください。')
        if name in DIRECTORIES:
            if not path.is_dir(): raise ValueError('保存フォルダの形式が正しくありません。')
            for root, dirs, files in os.walk(path):
                for child in dirs + files:
                    candidate = Path(root) / child
                    if linked(candidate): raise ValueError('保存対象にリンクがあります。通常のフォルダを使用してください。')
                    contained(candidate, data)
                result.extend(Path(root) / child for child in sorted(files))
        else:
            if not path.is_file(): raise ValueError('保存ファイルの形式が正しくありません。')
            result.append(path)
    return sorted(result)


def validate_database(path, notebook=False):
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=15)) as conn:
            conn.execute('PRAGMA query_only=ON')
            if conn.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                raise ValueError('DBの整合性を確認できませんでした。')
            if notebook:
                tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if not {'documents', 'content_blocks'}.issubset(tables):
                    raise ValueError('文書DBの形式が異なります。')
    except sqlite3.Error as error:
        raise ValueError('DBを読み取れません。正常なバックアップを選択してください。') from error


def manifest_for(data):
    if not (data / 'my_notebook.db').is_file(): raise ValueError('文書DBが見つかりません。')
    entries = []
    for path in data_files(data):
        entries.append({'path':path.relative_to(data).as_posix(), 'size':path.stat().st_size, 'sha256':digest(path)})
    return {'format':FORMAT, 'version':1, 'complete':True,
            'created_at':datetime.now(timezone.utc).isoformat(), 'scopes':list(SCOPES),
            'directories':[name for name in DIRECTORIES if (data/name).is_dir()], 'files':entries}


def read_manifest(folder):
    if linked(folder) or not folder.is_dir(): raise ValueError('通常のバックアップフォルダを指定してください。')
    try:
        path = folder/'backup.json'
        if linked(path) or path.stat().st_size > 64*1024*1024: raise ValueError('バックアップ情報が不正です。')
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['format'] != FORMAT or data['version'] != 1 or data['complete'] is not True: raise ValueError()
        if data['scopes'] != list(SCOPES) or not isinstance(data['files'], list): raise ValueError()
        if not isinstance(data['directories'],list) or any(name not in DIRECTORIES for name in data['directories']): raise ValueError()
        seen = set()
        for entry in data['files']:
            name = entry['path']; relative = PurePosixPath(name)
            if not isinstance(name,str) or '\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts: raise ValueError()
            if relative.as_posix() != name or not relative.parts or relative.parts[0] not in SCOPES: raise ValueError()
            if relative.parts[0] not in DIRECTORIES and len(relative.parts) != 1: raise ValueError()
            if relative.parts[0] in DIRECTORIES and len(relative.parts) < 2: raise ValueError()
            if name.casefold() in seen: raise ValueError()
            seen.add(name.casefold())
            if type(entry['size']) is not int or entry['size'] < 0: raise ValueError()
            if not isinstance(entry['sha256'],str) or len(entry['sha256']) != 64 or any(c not in '0123456789abcdef' for c in entry['sha256']): raise ValueError()
        if 'my_notebook.db' not in seen: raise ValueError()
        if not isinstance(data['created_at'],str): raise ValueError()
        return data
    except (KeyError,TypeError,ValueError,OSError) as error:
        raise ValueError('このフォルダは完了済みのアプリバックアップではありません。backup.jsonがあるフォルダを選択してください。') from error


def checked_source(folder, relative):
    path = folder/'data'
    if linked(path): raise ValueError('バックアップにリンクが含まれています。')
    for part in PurePosixPath(relative).parts:
        path = path / part
        if linked(path): raise ValueError('バックアップにリンクが含まれています。')
    contained(path, folder/'data')
    if not path.is_file(): raise ValueError('バックアップ内のファイルが不足しています。')
    return path


def verify_backup(folder, manifest=None, progress=None):
    manifest = manifest or read_manifest(folder)
    for number, entry in enumerate(manifest['files'],1):
        path = checked_source(folder, entry['path'])
        if path.stat().st_size != entry['size'] or digest(path) != entry['sha256']:
            raise ValueError('バックアップの検証に失敗しました。ファイルが変更または破損しています。')
        if entry['path'] in DATABASES: validate_database(path,entry['path']=='my_notebook.db')
        if progress: progress(number, len(manifest['files']))
    return manifest


class MaintenanceGate:
    def __init__(self): self.lock=threading.Lock(); self.active=0; self.reason=''
    def enter(self):
        with self.lock:
            if self.reason: raise ValueError(self.reason)
            if self.active: raise ValueError('処理中の画面があります。処理が終わってから再実行してください。')
            self.reason='バックアップ・復元の処理中です。完了までお待ちください。'
    def release(self):
        with self.lock: self.reason=''


class BackupGateMiddleware:
    """Hold API/media access through the final streaming byte, not just headers."""
    def __init__(self, app, gate): self.app=app; self.gate=gate
    async def __call__(self, scope, receive, send):
        path=scope.get('path','')
        guarded=scope['type']=='http' and ((path.startswith('/api/') and not path.startswith('/api/backup/')) or path.startswith('/media/'))
        if not guarded: return await self.app(scope,receive,send)
        with self.gate.lock:
            reason=self.gate.reason
            if not reason: self.gate.active+=1
        if reason:
            from starlette.responses import JSONResponse
            return await JSONResponse({'detail':reason},status_code=503)(scope,receive,send)
        try: await self.app(scope,receive,send)
        finally:
            with self.gate.lock: self.gate.active-=1


class DataBackup:
    def __init__(self, root, gate=None, busy=None):
        self.root=Path(root).resolve(); self.data=self.root/'data'; self.state=self.root/'.backup_restore'
        self.gate=gate or MaintenanceGate(); self.busy=busy or (lambda:False)
        self.jobs={}; self.lock=threading.Lock(); self.operation=threading.Lock()
        if (self.state/'pending.json').exists(): self.gate.reason='復元の準備が完了しています。アプリを終了して再起動してください。'

    def path(self, value, allow_history=False):
        if not value or not Path(value).is_absolute(): raise ValueError('保存先の絶対パスを指定してください。')
        path=Path(value).resolve()
        history=self.root/'.restore_history'
        internal_history=allow_history and path!=history and path.is_relative_to(history)
        if (path==self.root or path.is_relative_to(self.root)) and not internal_history: raise ValueError('アプリ本体の外にあるフォルダを選択してください。')
        if not path.is_dir(): raise ValueError('フォルダが見つかりません。存在するフォルダを選択してください。')
        return path

    def update(self, job, **values):
        with self.lock: self.jobs[job].update(values)

    def status(self):
        result=None
        path=self.state/'result.json'
        if path.exists():
            try: result=json.loads(path.read_text(encoding='utf-8'))
            except (OSError,ValueError): pass
        with self.lock: jobs=[dict(job) for job in self.jobs.values()]
        return {'jobs':jobs,'pending':(self.state/'pending.json').exists(),'last_restore':result}

    def preview(self, value):
        folder=self.path(value,allow_history=True); manifest=read_manifest(folder)
        return {'path':str(folder),'created_at':manifest['created_at'],'file_count':len(manifest['files']),
                'bytes':sum(entry['size'] for entry in manifest['files']),
                'pdf_count':sum(entry['path'].lower().startswith('ocr_archives/') and entry['path'].lower().endswith('.pdf') for entry in manifest['files'])}

    def start(self, kind, value):
        if kind not in ('backup','restore'): raise ValueError('処理の指定が異なります。')
        folder=self.path(value,allow_history=kind=='restore')
        if not self.operation.acquire(blocking=False): raise ValueError('別のバックアップ・復元処理が進行中です。')
        entered=False
        try:
            self.gate.enter(); entered=True
            if self.busy(): raise ValueError('OCR・検索・回答作成の処理が終わってから再実行してください。')
            job=uuid.uuid4().hex
            with self.lock:
                self.jobs={job:{'id':job,'kind':kind,'state':'running','message':'保存対象を確認しています…','bytes':0,'total_bytes':0}}
            threading.Thread(target=self.run,args=(job,kind,folder),name='data-backup',daemon=True).start()
            return {'job_id':job}
        except Exception:
            if entered: self.gate.release()
            self.operation.release(); raise

    def run(self, job, kind, folder):
        try:
            destination=self.backup(job,folder) if kind=='backup' else self.prepare(job,folder)
            self.update(job,state='completed' if kind=='backup' else 'ready',path=str(destination),
                        message='バックアップが完了しました。' if kind=='backup' else '復元の準備が完了しました。アプリ本体を終了して再起動してください。ブラウザの再読み込みだけでは反映されません。')
        except Exception as error:
            message=str(error) if isinstance(error,ValueError) else '処理を完了できませんでした。保存先の空き容量・アクセス権・ファイルの使用状況を確認してください。'
            self.update(job,state='failed',message=message)
        finally:
            if not (self.state/'pending.json').exists(): self.gate.release()
            else: self.gate.reason='復元の準備が完了しています。アプリを終了して再起動してください。'
            self.operation.release()

    def copy(self, source, target, progress):
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as incoming, target.open('xb') as outgoing:
            while chunk:=incoming.read(1024*1024): outgoing.write(chunk); progress(len(chunk))
            outgoing.flush(); os.fsync(outgoing.fileno())
        shutil.copystat(source,target)

    def space(self, folder, size):
        if shutil.disk_usage(folder).free < size+16*1024*1024: raise ValueError('保存先の空き容量が不足しています。別のフォルダを選択してください。')

    def backup(self, job, parent):
        paths=data_files(self.data)
        total=sum(path.stat().st_size for path in paths); self.space(parent,total)
        self.update(job,total_bytes=total,message='DBとPDF・図版をバックアップしています…')
        stamp=datetime.now().strftime('%Y-%m-%d_%H%M%S')+'_'+job[:8]
        temporary=parent/('.MyNotebookLM_'+stamp+'.partial'); final=parent/('MyNotebookLM_バックアップ_'+stamp)
        temporary.mkdir(); done=0
        def progress(size):
            nonlocal done
            done+=size; self.update(job,bytes=done)
        try:
            for name in DIRECTORIES:
                if (self.data/name).is_dir(): (temporary/'data'/name).mkdir(parents=True)
            for source in paths:
                relative=source.relative_to(self.data); target=temporary/'data'/relative
                if relative.as_posix() in DATABASES:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with closing(sqlite3.connect(source.resolve().as_uri()+'?mode=ro',uri=True,timeout=30)) as incoming, closing(sqlite3.connect(target)) as outgoing:
                        incoming.backup(outgoing,pages=1024,sleep=0.05)
                        outgoing.execute('PRAGMA journal_mode=DELETE')
                    validate_database(target,relative.as_posix()=='my_notebook.db'); progress(source.stat().st_size)
                else: self.copy(source,target,progress)
            self.update(job,message='バックアップの整合性を検証しています…')
            manifest=manifest_for(temporary/'data'); write_json(temporary/'backup.json',manifest)
            verify_backup(temporary,manifest)
            temporary.rename(final)
            return final
        except Exception:
            remove_owned(temporary,parent); raise

    def prepare(self, job, folder):
        manifest=read_manifest(folder)
        self.state.mkdir(exist_ok=True); self.space(self.state,sum(entry['size'] for entry in manifest['files']))
        incoming=self.state/('incoming_'+job); incoming.mkdir()
        total=sum(entry['size'] for entry in manifest['files']); done=0
        self.update(job,total_bytes=total,message='バックアップを読み込み、復元用データを検証しています…')
        def progress(size):
            nonlocal done
            done+=size; self.update(job,bytes=done)
        try:
            for name in manifest['directories']: (incoming/'data'/name).mkdir(parents=True)
            for entry in manifest['files']:
                source=checked_source(folder,entry['path']); target=incoming/'data'/entry['path']
                if source.stat().st_size!=entry['size']: raise ValueError('バックアップのサイズが一致しません。')
                self.copy(source,target,progress)
            write_json(incoming/'backup.json',manifest); verify_backup(incoming,manifest)
            write_json(self.state/'pending.json',{'job_id':job,'incoming':incoming.name})
            return incoming
        except Exception:
            remove_owned(incoming,self.state); raise

    def cancel(self):
        if not self.operation.acquire(blocking=False): raise ValueError('処理中は取り消せません。完了までお待ちください。')
        try:
            pending=self.state/'pending.json'
            if pending.exists():
                info=json.loads(pending.read_text(encoding='utf-8')); incoming=self.state/info['incoming']
                contained(incoming,self.state); remove_owned(incoming,self.state); pending.unlink()
            self.gate.release()
            with self.lock:
                for job in self.jobs.values():
                    if job['state']=='ready': job.update(state='canceled',message='復元を取り消しました。現在のデータを引き続き使用できます。')
            return {'status':'success'}
        finally: self.operation.release()


def rollback(root, journal):
    history=root/'.restore_history'/journal['history']; contained(history,root/'.restore_history')
    for entry in reversed(journal['entries']):
        name=entry['name']
        if name not in SCOPES and name not in [db+suffix for db in DATABASES for suffix in ('-wal','-shm')]: raise ValueError('復元記録の形式が異なります。')
        target=root/'data'/name; old=history/'data'/name
        if old.exists():
            if target.exists(): remove_owned(target,root/'data')
            old.rename(target)
        elif not entry['existed'] and target.exists(): remove_owned(target,root/'data')


def apply_pending_restore(root):
    """Called before importing/initializing any app DB or starting any workers."""
    root=Path(root).resolve(); state=root/'.backup_restore'; pending=state/'pending.json'; journal_path=state/'journal.json'
    if journal_path.exists():
        journal=json.loads(journal_path.read_text(encoding='utf-8'))
        if journal['phase']!='committed':
            rollback(root,journal)
            write_json(state/'result.json',{'job_id':journal['job_id'],'state':'failed','message':'前回の復元が中断されたため、復元前のデータへ戻しました。'})
            if pending.exists(): pending.rename(state/('failed_'+uuid.uuid4().hex+'.json'))
        else:
            if pending.exists(): pending.unlink()
            write_json(state/'result.json',{'job_id':journal['job_id'],'state':'completed','message':'復元が完了しました。','history':str(root/'.restore_history'/journal['history'])})
        journal_path.unlink()
    if not pending.exists(): return
    info=json.loads(pending.read_text(encoding='utf-8')); incoming=state/info['incoming']; contained(incoming,state)
    history_name='復元前_'+datetime.now().strftime('%Y-%m-%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    history=root/'.restore_history'/history_name
    journal=None
    try:
        verify_backup(incoming)
        # All old processes/connections have exited. Flush committed WAL pages before
        # moving DBs, then retain the complete previous dataset as a valid backup.
        for name in DATABASES:
            path=root/'data'/name
            if path.exists():
                with closing(sqlite3.connect(path,timeout=15)) as conn:
                    if conn.execute('PRAGMA wal_checkpoint(TRUNCATE)').fetchone()[0]: raise ValueError('DBが使用中のため復元できません。')
        previous=manifest_for(root/'data')
        history.mkdir(parents=True); (history/'data').mkdir()
        names=list(SCOPES)+[name+suffix for name in DATABASES for suffix in ('-wal','-shm')]
        journal={'phase':'applying','job_id':info['job_id'],'history':history_name,'incoming':incoming.name,
                 'entries':[{'name':name,'existed':(root/'data'/name).exists()} for name in names]}
        write_json(journal_path,journal)
        for entry in journal['entries']:
            name=entry['name']; target=root/'data'/name; old=history/'data'/name; replacement=incoming/'data'/name
            if target.exists(): target.rename(old)
            if replacement.exists(): replacement.rename(target)
        write_json(history/'backup.json',previous)
        verify_backup(history,previous)
        journal['phase']='committed'; write_json(journal_path,journal)
        write_json(state/'result.json',{'job_id':info['job_id'],'state':'completed','message':'復元が完了しました。','history':str(history)})
        pending.unlink(); journal_path.unlink(); remove_owned(incoming,state)
    except Exception:
        if journal is not None:
            rollback(root,journal)
            if journal_path.exists(): journal_path.unlink()
        write_json(state/'result.json',{'job_id':info['job_id'],'state':'failed','message':'復元できなかったため、元のデータを保持しました。空き容量・ファイルの使用状況を確認してください。'})
        if pending.exists(): pending.rename(state/('failed_'+uuid.uuid4().hex+'.json'))
        if incoming.exists():
            try: remove_owned(incoming,state)
            except OSError: pass


def choose_folder(purpose):
    from app.runtime_paths import windows_powershell
    powershell=windows_powershell()
    if not powershell: raise ValueError('フォルダ選択画面を開けません。保存先のパスを直接入力してください。')
    title='バックアップの保存先を選択' if purpose=='backup' else 'backup.jsonがあるバックアップフォルダを選択'
    script="""Add-Type -AssemblyName System.Windows.Forms
$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
$dialog.Description = '%s'
$dialog.ShowNewFolderButton = $true
$owner = New-Object System.Windows.Forms.Form
$owner.TopMost = $true
try { if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
 [Console]::Write([Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($dialog.SelectedPath)))
} } finally { $dialog.Dispose(); $owner.Dispose() }
""" % title
    result=subprocess.run([powershell,'-NoProfile','-STA','-EncodedCommand',base64.b64encode(script.encode('utf-16-le')).decode()],capture_output=True,timeout=300,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if result.returncode: raise ValueError('フォルダ選択画面を開けません。パスを直接入力してください。')
    return base64.b64decode(result.stdout.strip()).decode('utf-8') if result.stdout.strip() else ''


def create_backup_router(service, token):
    from fastapi import APIRouter, Depends, HTTPException, Request
    from pydantic import BaseModel, Field
    import secrets
    class FolderRequest(BaseModel):
        path:str=Field(min_length=1,max_length=32767)
    class PickerRequest(BaseModel):
        purpose:str='backup'
    def authenticate(request:Request):
        if not secrets.compare_digest(request.headers.get('x-app-token',''),token): raise HTTPException(401,'画面を再読み込みしてください。')
    def action(function, *args):
        try: return function(*args)
        except ValueError as error: raise HTTPException(400,str(error)) from None
        except (OSError,subprocess.TimeoutExpired): raise HTTPException(503,'フォルダを利用できません。パスやアクセス権を確認してください。') from None
    router=APIRouter(prefix='/api/backup',dependencies=[Depends(authenticate)])
    @router.get('/status')
    def status(): return service.status()
    @router.post('/choose-folder')
    def choose(data:PickerRequest): return {'path':action(choose_folder,data.purpose)}
    @router.post('/preview')
    def preview(data:FolderRequest): return action(service.preview,data.path)
    @router.post('/create')
    def create(data:FolderRequest): return action(service.start,'backup',data.path)
    @router.post('/restore')
    def restore(data:FolderRequest): return action(service.start,'restore',data.path)
    @router.post('/cancel-restore')
    def cancel(): return action(service.cancel)
    return router
