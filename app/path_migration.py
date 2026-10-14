"""One-time, idempotent migration of application-owned SQLite file references."""
import argparse
from contextlib import closing
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from app.paths import APP_PATHS, AppPaths

COLUMNS = {
    'my_notebook.db': (
        ('documents', 'doc_metadata_json', 'metadata'),
        ('document_figures', 'file_path', 'store_file'),
        ('content_blocks', 'html_content', 'html'),
    ),
    'knowledge.db': (
        ('documents', 'file_path', 'store_file'),
        ('pages', 'image_path', 'store_file'),
        ('figures', 'image_path', 'store_file'),
        ('content_blocks', 'html_content', 'html'),
    ),
}

def unlinked(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise ValueError('リンクされた場所は移行・バックアップの対象にできません。')
    return path

def migrate(root=APP_PATHS.root, *, apply=False, source_roots=()):
    paths = AppPaths(unlinked(root), source_roots)
    result = {'applied':apply, 'databases':{}, 'changed_values':0}
    for filename, columns in COLUMNS.items():
        database = unlinked(paths.data / filename)
        if not database.is_file() or not database.stat().st_size:
            continue
        uri = database.resolve().as_uri() + ('?mode=rw' if apply else '?mode=ro')
        with closing(sqlite3.connect(uri, uri=True, timeout=60)) as conn:
            if not apply: conn.execute('PRAGMA query_only=ON')
            conn.execute('BEGIN IMMEDIATE' if apply else 'BEGIN')
            changes = []
            try:
                for table, column, method in columns:
                    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                        continue
                    if column not in {r[1] for r in conn.execute('PRAGMA table_info("'+table+'")')}:
                        continue
                    query = 'SELECT rowid,"'+column+'" FROM "'+table+'" WHERE typeof("'+column+'")=\'text\''
                    args = ()
                    if method == 'html':
                        roots=(paths.root,)+paths.source_roots
                        prefixes=[]
                        for prefix in roots:
                            prefixes.extend((str(prefix),prefix.as_posix(),prefix.as_uri()))
                        query += ' AND ('+' OR '.join('instr("'+column+'",?)>0' for _ in prefixes)+')'
                        args=tuple(prefixes)
                    updates=[]
                    for rowid, value in conn.execute(query,args):
                        converted=getattr(paths,method)(value)
                        if converted!=value: updates.append((converted,rowid))
                    if updates:
                        if apply: conn.executemany('UPDATE "'+table+'" SET "'+column+'"=? WHERE rowid=?',updates)
                        changes.append({'table':table,'column':column,'values':len(updates)})
                if apply: conn.commit()
                else: conn.rollback()
            except Exception:
                conn.rollback()
                raise
            result['databases'][filename]=changes
            result['changed_values']+=sum(r['values'] for r in changes)
    return result

def backup_databases(root, destination):
    paths=AppPaths(unlinked(root)); destination=unlinked(destination).resolve()
    if destination == paths.root or destination.is_relative_to(paths.root):
        raise ValueError('バックアップ先はアプリの外を指定してください。')
    if destination.exists():
        raise ValueError('上書きを防ぐため、新しいバックアップフォルダーを指定してください。')
    destination.mkdir(parents=True)
    for filename in COLUMNS:
        source=unlinked(paths.data/filename)
        if not source.is_file() or not source.stat().st_size: continue
        target=destination/filename
        with closing(sqlite3.connect(source.as_uri()+'?mode=ro',uri=True,timeout=60)) as incoming, closing(sqlite3.connect(target)) as outgoing:
            incoming.backup(outgoing,pages=1024,sleep=0.01)
            outgoing.execute('PRAGMA journal_mode=DELETE')
            if outgoing.execute('PRAGMA quick_check').fetchall()!=[('ok',)]:
                raise ValueError('バックアップDBの整合性を確認できません。')
    return destination

def main():
    parser=argparse.ArgumentParser(description='内部パスの相対化。既定は変更件数の確認のみ。')
    parser.add_argument('--root',type=Path,default=APP_PATHS.root)
    parser.add_argument('--source-root',type=Path,action='append',default=[])
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--check',action='store_true')
    parser.add_argument('--backup-directory',type=Path)
    args=parser.parse_args()
    backup=None
    if args.apply:
        destination=args.backup_directory or AppPaths(args.root).backup_directory/('パス相対化_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        backup=backup_databases(args.root,destination)
    result=migrate(args.root,apply=args.apply,source_roots=args.source_root)
    if backup:result['backup']=str(backup)
    print(json.dumps(result,ensure_ascii=True,indent=2))

if __name__=='__main__':
    main()
