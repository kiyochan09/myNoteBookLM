"""Research thread storage, separate from the source document database."""
from pathlib import Path
from contextlib import contextmanager
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


class ThreadData(BaseModel):
    title: str = Field(default="新しいリサーチ", max_length=100)
    groupIds: List[str] = Field(default_factory=list, max_length=500)
    tagIds: List[str] = Field(default_factory=list, max_length=500)
    question: str = Field(default="", max_length=12000)


class ThreadCreate(ThreadData):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1, max_length=100)


class ThreadUpdate(ThreadData):
    revision: int = Field(ge=1)


class ResearchThreads:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS research_threads (
                id TEXT PRIMARY KEY, title TEXT NOT NULL,
                group_ids_json TEXT NOT NULL, tag_ids_json TEXT NOT NULL,
                question TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT
            )''')

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @staticmethod
    def now():
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')

    @staticmethod
    def decode(row):
        return dict(id=row['id'], title=row['title'], groupIds=json.loads(row['group_ids_json']),
                    tagIds=json.loads(row['tag_ids_json']), question=row['question'], revision=row['revision'],
                    created_at=row['created_at'], updated_at=row['updated_at'])

    def list(self):
        with self.connect() as conn:
            return [self.decode(row) for row in conn.execute(
                'SELECT * FROM research_threads WHERE deleted_at IS NULL ORDER BY updated_at DESC, id')]

    def create(self, data):
        now = self.now()
        with self.connect() as conn:
            row = conn.execute('SELECT * FROM research_threads WHERE id=?', (data.id,)).fetchone()
            if row:
                if row['deleted_at']:
                    raise HTTPException(409, 'このスレッドは削除されています。')
                return self.decode(row)  # Safe retry after a lost create response.
            conn.execute('INSERT INTO research_threads VALUES (?,?,?,?,?,1,?,?,NULL)',
                         (data.id, data.title.strip() or '新しいリサーチ', json.dumps(data.groupIds),
                          json.dumps(data.tagIds), data.question, now, now))
            return self.decode(conn.execute('SELECT * FROM research_threads WHERE id=?', (data.id,)).fetchone())

    def update(self, thread_id, data):
        with self.connect() as conn:
            cursor = conn.execute('''UPDATE research_threads SET title=?, group_ids_json=?, tag_ids_json=?,
                question=?, revision=revision+1, updated_at=?
                WHERE id=? AND revision=? AND deleted_at IS NULL''',
                (data.title.strip() or '新しいリサーチ', json.dumps(data.groupIds), json.dumps(data.tagIds),
                 data.question, self.now(), thread_id, data.revision))
            if not cursor.rowcount:
                row = conn.execute('SELECT deleted_at FROM research_threads WHERE id=?', (thread_id,)).fetchone()
                if not row or row['deleted_at']:
                    raise HTTPException(404, 'スレッドが見つかりません。')
                raise HTTPException(409, '別の画面で更新されています。下書きを別スレッドに保存してください。')
            return self.decode(conn.execute('SELECT * FROM research_threads WHERE id=?', (thread_id,)).fetchone())

    def delete(self, thread_id, revision):
        with self.connect() as conn:
            cursor = conn.execute('''UPDATE research_threads SET deleted_at=?, revision=revision+1
                WHERE id=? AND revision=? AND deleted_at IS NULL''', (self.now(), thread_id, revision))
            if not cursor.rowcount:
                raise HTTPException(409, 'スレッドが更新または削除されています。再読み込みしてください。')

    def restore(self, thread_id):
        with self.connect() as conn:
            cursor = conn.execute('''UPDATE research_threads SET deleted_at=NULL, revision=revision+1, updated_at=?
                WHERE id=? AND deleted_at IS NOT NULL''', (self.now(), thread_id))
            if not cursor.rowcount:
                raise HTTPException(404, '削除したスレッドが見つかりません。')
            return self.decode(conn.execute('SELECT * FROM research_threads WHERE id=?', (thread_id,)).fetchone())


def create_research_router(path, db_service):
    store = ResearchThreads(path)
    router = APIRouter(prefix='/api/research/threads')

    def validate_scope(data):
        conn = db_service.get_connection()
        try:
            groups = {r['id'] for r in conn.execute('SELECT id FROM tag_groups')}
            tags = {r['id']: r['group_id'] for r in conn.execute('SELECT id, group_id FROM tags')}
        finally:
            conn.close()
        data.groupIds = list(dict.fromkeys(data.groupIds))
        data.tagIds = list(dict.fromkeys(data.tagIds))
        if not set(data.groupIds) <= groups or any(tags.get(t) not in data.groupIds for t in data.tagIds):
            raise HTTPException(400, 'グループやタグが変更されています。資料一覧を再読み込みしてください。')
        return data

    @router.get('')
    def list_threads():
        return {'threads': store.list()}

    @router.post('')
    def create_thread(data: ThreadCreate):
        return {'thread': store.create(validate_scope(data))}

    @router.patch('/{thread_id}')
    def update_thread(thread_id: str, data: ThreadUpdate):
        return {'thread': store.update(thread_id, validate_scope(data))}

    @router.delete('/{thread_id}')
    def delete_thread(thread_id: str, revision: int):
        store.delete(thread_id, revision)
        return {'status': 'success'}

    @router.post('/{thread_id}/restore')
    def restore_thread(thread_id: str):
        return {'thread': store.restore(thread_id)}

    return router
