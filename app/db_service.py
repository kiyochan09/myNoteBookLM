import sqlite3
import json
import uuid
import datetime
import html
from pathlib import Path
from typing import Dict, Any, List, Optional

def sanitize_fts_snippet(snippet_text: str) -> str:
    """SQLite FTS5のsnippet()結果から<mark>タグのみを維持し、他の全テキストを安全にHTMLエスケープ"""
    if not snippet_text:
        return ""
    token_open = "___FTS_MARK_OPEN___"
    token_close = "___FTS_MARK_CLOSE___"
    escaped_tokens = snippet_text.replace("<mark>", token_open).replace("</mark>", token_close)
    safe = html.escape(escaped_tokens)
    return safe.replace(token_open, "<mark>").replace(token_close, "</mark>")

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_NOTEBOOK_DB_PATH = BASE_DIR / "data" / "my_notebook.db"

NOTEBOOK_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- 1. ドキュメント基本テーブル
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_filename TEXT,
    total_pages INTEGER DEFAULT 0,
    doc_metadata_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. セクション / 見出し階層テーブル
CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    level INTEGER NOT NULL,
    order_idx INTEGER NOT NULL,
    page_number INTEGER
);

-- 3. コンテンツブロックテーブル
CREATE TABLE IF NOT EXISTS content_blocks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    block_type TEXT NOT NULL,
    text_content TEXT NOT NULL,
    html_content TEXT,
    reading_order INTEGER NOT NULL,
    page_number INTEGER,
    bbox_json TEXT
);

-- 4. 表テーブル
CREATE TABLE IF NOT EXISTS document_tables (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    caption TEXT,
    row_count INTEGER NOT NULL,
    col_count INTEGER NOT NULL,
    grid_json TEXT NOT NULL,
    reading_order INTEGER NOT NULL,
    page_number INTEGER
);

-- 5. 図版テーブル
CREATE TABLE IF NOT EXISTS document_figures (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    caption TEXT,
    file_path TEXT NOT NULL,
    file_hash TEXT,
    file_size_kb REAL,
    width INTEGER,
    height INTEGER,
    reading_order INTEGER NOT NULL,
    page_number INTEGER,
    is_custom BOOLEAN DEFAULT FALSE
);

-- 6. 注釈・リンク・ブックマークテーブル
CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    block_id TEXT REFERENCES content_blocks(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    anchor_text TEXT,
    target_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. 全文検索インデックス (SQLite FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_blocks USING fts5(
    block_id UNINDEXED,
    document_id UNINDEXED,
    title,
    text_content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS trg_notebook_blocks_ai AFTER INSERT ON content_blocks
WHEN new.text_content IS NOT NULL AND length(trim(new.text_content)) > 0
BEGIN
    INSERT INTO fts_blocks(block_id, document_id, title, text_content)
    VALUES (
        new.id,
        new.document_id,
        COALESCE((SELECT title FROM sections WHERE id = new.section_id), ''),
        new.text_content
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_notebook_blocks_ad AFTER DELETE ON content_blocks
BEGIN
    DELETE FROM fts_blocks WHERE block_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_notebook_blocks_au AFTER UPDATE ON content_blocks
BEGIN
    DELETE FROM fts_blocks WHERE block_id = old.id;
    INSERT INTO fts_blocks(block_id, document_id, title, text_content)
    VALUES (
        new.id,
        new.document_id,
        COALESCE((SELECT title FROM sections WHERE id = new.section_id), ''),
        new.text_content
    );
END;

-- 8. タググループテーブル
CREATE TABLE IF NOT EXISTS tag_groups (
    id TEXT PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    order_idx INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. タグ基本テーブル
CREATE TABLE IF NOT EXISTS tags (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    color TEXT DEFAULT '#3b82f6',
    group_id TEXT REFERENCES tag_groups(id) ON DELETE SET NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_tags_name_group ON tags(name, COALESCE(group_id, ''));

-- 10. ドキュメント・タグ中間テーブル
CREATE TABLE IF NOT EXISTS document_tags (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tag_id TEXT NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (document_id, tag_id)
);

-- 高速化インデックス
CREATE INDEX IF NOT EXISTS idx_content_blocks_doc ON content_blocks(document_id);
CREATE INDEX IF NOT EXISTS idx_content_blocks_section ON content_blocks(section_id);
CREATE INDEX IF NOT EXISTS idx_document_figures_doc ON document_figures(document_id);
CREATE INDEX IF NOT EXISTS idx_document_figures_section ON document_figures(section_id);
CREATE INDEX IF NOT EXISTS idx_document_tables_doc ON document_tables(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tables_section ON document_tables(section_id);
CREATE INDEX IF NOT EXISTS idx_annotations_block ON annotations(block_id);
CREATE INDEX IF NOT EXISTS idx_sections_parent ON sections(parent_id);
CREATE INDEX IF NOT EXISTS idx_document_tags_doc ON document_tags(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_group ON tags(group_id);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
"""


class DBService:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_NOTEBOOK_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # journal_mode の切替は必要な場合だけ初期化時に行う。
            # 既にWALなら再設定せず、別接続とのロック競合を避ける。
            current_journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
            if str(current_journal_mode).lower() != "wal":
                conn.execute("PRAGMA journal_mode = WAL;")
            conn.executescript(NOTEBOOK_SCHEMA_SQL)

            # 既存テーブルへの group_id カラム追加マイグレーション
            try:
                cur.execute("ALTER TABLE tags ADD COLUMN group_id TEXT REFERENCES tag_groups(id) ON DELETE SET NULL;")
            except sqlite3.OperationalError:
                pass

            # 単体 UNIQUE(name) から複合一意制約への安全な移行
            table_sql = cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='tags'").fetchone()
            if table_sql and "name TEXT UNIQUE" in table_sql[0]:
                cur.execute("PRAGMA foreign_keys = OFF;")
                cur.execute("""
                    CREATE TABLE tags_new (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        color TEXT DEFAULT '#3b82f6',
                        group_id TEXT REFERENCES tag_groups(id) ON DELETE SET NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("INSERT INTO tags_new (id, name, color, group_id, created_at) SELECT id, name, color, group_id, created_at FROM tags;")
                cur.execute("DROP TABLE tags;")
                cur.execute("ALTER TABLE tags_new RENAME TO tags;")
                cur.execute("PRAGMA foreign_keys = ON;")

            # 複合一意インデックスの作成（同一グループ内でのみタグ名重複禁止）
            cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_tags_name_group ON tags(name, COALESCE(group_id, ''));")

            # 高速化インデックスの確実な配備
            cur.execute("CREATE INDEX IF NOT EXISTS idx_content_blocks_doc ON content_blocks(document_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_figures_doc ON document_figures(document_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_tables_doc ON document_tables(document_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_tags_doc ON document_tags(document_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_tags_group ON tags(group_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(document_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);")

            conn.commit()
        finally:
            conn.close()

    def list_tag_groups(self) -> List[Dict[str, Any]]:
        """全タググループ一覧と、所属するタグ一覧を取得"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            group_rows = cur.execute(
                "SELECT id, name, order_idx, created_at FROM tag_groups ORDER BY order_idx ASC, name ASC"
            ).fetchall()
            groups = [dict(r) for r in group_rows]

            # 全タグを取得してグループごとに分類
            all_tags = self.list_tags()
            tags_by_group: Dict[str, List[Dict[str, Any]]] = {}
            unassigned_tags: List[Dict[str, Any]] = []

            for t in all_tags:
                gid = t.get("group_id")
                if gid:
                    if gid not in tags_by_group:
                        tags_by_group[gid] = []
                    tags_by_group[gid].append(t)
                else:
                    unassigned_tags.append(t)

            for g in groups:
                g["tags"] = tags_by_group.get(g["id"], [])
                g["tag_count"] = len(g["tags"])

            return {
                "groups": groups,
                "unassigned_tags": unassigned_tags
            }
        finally:
            conn.close()

    def create_tag_group(self, name: str) -> Dict[str, Any]:
        """新規タググループ作成（グループ名・タグ名と重複しないようチェック）"""
        name = name.strip()
        if not name:
            raise ValueError("Tag group name cannot be empty")

        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # 1. 同名のグループが存在しないかチェック
            existing = cur.execute("SELECT * FROM tag_groups WHERE name = ?", (name,)).fetchone()
            if existing:
                raise ValueError(f"同名のグループ「{name}」が既に存在します。別の名前を指定してください。")

            # 2. 同名のタグが存在しないかチェック（タグ名とグループ名の重複禁止）
            tag_conflict = cur.execute("SELECT * FROM tags WHERE name = ?", (name,)).fetchone()
            if tag_conflict:
                raise ValueError(f"同名のタグ「{name}」が既に存在するため、グループ名として使用できません。別の名前を指定してください。")

            group_id = f"tgrp-{uuid.uuid4().hex[:8]}"
            max_order = cur.execute("SELECT COALESCE(MAX(order_idx), 0) FROM tag_groups").fetchone()[0]
            order_idx = max_order + 1
            cur.execute(
                "INSERT INTO tag_groups (id, name, order_idx) VALUES (?, ?, ?)",
                (group_id, name, order_idx)
            )
            conn.commit()
            return {"id": group_id, "name": name, "order_idx": order_idx}
        finally:
            conn.close()

    def delete_tag_group(self, group_id: str) -> bool:
        """タググループ削除（所属タグのgroup_idはNULLへ）"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("UPDATE tags SET group_id = NULL WHERE group_id = ?", (group_id,))
            cur.execute("DELETE FROM tag_groups WHERE id = ?", (group_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def assign_tag_to_group(self, tag_id: str, group_id: Optional[str]) -> bool:
        """タグをグループに所属させる（group_idがNoneならグループ解除。移動先グループ内重複チェック付き）"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            tag = cur.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
            if not tag:
                raise ValueError(f"Tag with id '{tag_id}' not found")

            tag_name = tag["name"]
            # 移動先グループ内に既に同名のタグが存在しないか検証
            if group_id:
                conflict = cur.execute(
                    "SELECT * FROM tags WHERE name = ? AND group_id = ? AND id != ?",
                    (tag_name, group_id, tag_id)
                ).fetchone()
                if conflict:
                    grp_row = cur.execute("SELECT name FROM tag_groups WHERE id = ?", (group_id,)).fetchone()
                    grp_name = grp_row["name"] if grp_row else group_id
                    raise ValueError(f"グループ「{grp_name}」内に既に同名のタグ「{tag_name}」が存在するため、移動できません。")
            else:
                conflict = cur.execute(
                    "SELECT * FROM tags WHERE name = ? AND group_id IS NULL AND id != ?",
                    (tag_name, tag_id)
                ).fetchone()
                if conflict:
                    raise ValueError(f"未所属タグに既に同名のタグ「{tag_name}」が存在するため、未分類へ変更できません。")

            cur.execute("UPDATE tags SET group_id = ? WHERE id = ?", (group_id, tag_id))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def list_tags(self, group_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """全タグの一覧（グループ情報・紐づく文書件数付き）を取得"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            query = """
                SELECT t.id, t.name, t.color, t.group_id, tg.name AS group_name, t.created_at,
                       COUNT(dt.document_id) AS doc_count
                FROM tags t
                LEFT JOIN tag_groups tg ON t.group_id = tg.id
                LEFT JOIN document_tags dt ON t.id = dt.tag_id
            """
            params = []
            if group_id:
                query += " WHERE t.group_id = ?"
                params.append(group_id)
            query += """
                GROUP BY t.id, t.name, t.color, t.group_id, tg.name, t.created_at
                ORDER BY tg.order_idx ASC, t.name ASC
            """
            rows = cur.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_document_counts(self) -> Dict[str, int]:
        """総ドキュメント数と未分類ドキュメント数を軽量集計"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            total = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            unassigned = cur.execute("""
                SELECT COUNT(*) FROM documents d
                WHERE NOT EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id)
            """).fetchone()[0]
            return {"total_docs_count": total, "unassigned_docs_count": unassigned}
        finally:
            conn.close()

    def create_tag(self, name: str, color: Optional[str] = None, group_id: Optional[str] = None) -> Dict[str, Any]:
        """新規タグの作成（グループ名との重複禁止、同一グループ内での重複禁止）"""
        name = name.strip()
        if not name:
            raise ValueError("Tag name cannot be empty")

        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # 1. 同名のグループが存在しないかチェック（タグ名とグループ名の重複禁止）
            group_conflict = cur.execute("SELECT * FROM tag_groups WHERE name = ?", (name,)).fetchone()
            if group_conflict:
                raise ValueError(f"同名のグループ「{name}」が既に存在するため、タグ名として使用できません。別の名前を指定してください。")

            # 2. 同一グループ内（または未所属内）に同名のタグが存在しないかチェック
            if group_id:
                tag_conflict = cur.execute("SELECT * FROM tags WHERE name = ? AND group_id = ?", (name, group_id)).fetchone()
            else:
                tag_conflict = cur.execute("SELECT * FROM tags WHERE name = ? AND group_id IS NULL", (name,)).fetchone()

            if tag_conflict:
                grp_name = "未所属"
                if group_id:
                    grp_row = cur.execute("SELECT name FROM tag_groups WHERE id = ?", (group_id,)).fetchone()
                    if grp_row:
                        grp_name = f"グループ「{grp_row['name']}」"
                raise ValueError(f"{grp_name}内に同名のタグ「{name}」が既に存在します。別の名前を指定してください。")

            tag_id = f"tag-{uuid.uuid4().hex[:8]}"
            default_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#06b6d4', '#64748b', '#ef4444']
            import random
            selected_color = color or random.choice(default_colors)
            cur.execute(
                "INSERT INTO tags (id, name, color, group_id) VALUES (?, ?, ?, ?)",
                (tag_id, name, selected_color, group_id)
            )
            conn.commit()
            return {"id": tag_id, "name": name, "color": selected_color, "group_id": group_id}
        finally:
            conn.close()

    def update_tag(self, tag_id: str, name: Optional[str] = None, color: Optional[str] = None, group_id: Optional[str] = None) -> Dict[str, Any]:
        """タグ名・色・所属グループの更新（重複チェック付き）"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            existing = cur.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
            if not existing:
                raise ValueError(f"Tag with id '{tag_id}' not found")

            target_name = name.strip() if name is not None else existing["name"]
            target_group_id = group_id if group_id is not None else existing["group_id"]

            if not target_name:
                raise ValueError("Tag name cannot be empty")

            # 1. 同名グループが存在しないかチェック（タグ名とグループ名の重複禁止）
            group_conflict = cur.execute("SELECT * FROM tag_groups WHERE name = ?", (target_name,)).fetchone()
            if group_conflict:
                raise ValueError(f"同名のグループ「{target_name}」が既に存在するため、タグ名として使用できません。別の名前を指定してください。")

            # 2. 対象グループ内で同名タグが存在しないかチェック
            if target_group_id:
                tag_conflict = cur.execute(
                    "SELECT * FROM tags WHERE name = ? AND group_id = ? AND id != ?",
                    (target_name, target_group_id, tag_id)
                ).fetchone()
            else:
                tag_conflict = cur.execute(
                    "SELECT * FROM tags WHERE name = ? AND group_id IS NULL AND id != ?",
                    (target_name, tag_id)
                ).fetchone()

            if tag_conflict:
                grp_name = "未所属"
                if target_group_id:
                    grp_row = cur.execute("SELECT name FROM tag_groups WHERE id = ?", (target_group_id,)).fetchone()
                    if grp_row:
                        grp_name = f"グループ「{grp_row['name']}」"
                raise ValueError(f"{grp_name}内に同名のタグ「{target_name}」が既に存在します。別の名前を指定してください。")

            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(target_name)

            if color is not None:
                updates.append("color = ?")
                params.append(color)

            if group_id is not None:
                updates.append("group_id = ?")
                params.append(group_id)

            if updates:
                params.append(tag_id)
                cur.execute(f"UPDATE tags SET {', '.join(updates)} WHERE id = ?", params)
                conn.commit()

            updated = cur.execute("SELECT * FROM tags WHERE id = ?", (tag_id,)).fetchone()
            return dict(updated)
        finally:
            conn.close()

    def delete_tag(self, tag_id: str) -> bool:
        """タグの削除（中間テーブルもCASCADE削除）"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM tags WHERE id = ?", (tag_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def assign_tags_to_documents(self, doc_ids: List[str], tag_id: str) -> int:
        """複数ドキュメントにタグを一括付与"""
        if not doc_ids:
            return 0
        conn = self.get_connection()
        cur = conn.cursor()
        count = 0
        try:
            tag = cur.execute("SELECT id FROM tags WHERE id = ?", (tag_id,)).fetchone()
            if not tag:
                raise ValueError(f"Tag '{tag_id}' not found")

            for doc_id in doc_ids:
                doc = cur.execute("SELECT id FROM documents WHERE id = ?", (doc_id,)).fetchone()
                if not doc:
                    continue
                cur.execute(
                    "INSERT OR IGNORE INTO document_tags (document_id, tag_id) VALUES (?, ?)",
                    (doc_id, tag_id)
                )
                if cur.rowcount > 0:
                    count += 1
            conn.commit()
            return count
        finally:
            conn.close()

    def remove_tags_from_documents(self, doc_ids: List[str], tag_id: str) -> int:
        """複数ドキュメントからタグを一括解除"""
        if not doc_ids:
            return 0
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            placeholders = ",".join(["?"] * len(doc_ids))
            cur.execute(
                f"DELETE FROM document_tags WHERE tag_id = ? AND document_id IN ({placeholders})",
                [tag_id] + doc_ids
            )
            conn.commit()
            return cur.rowcount
        finally:
            conn.close()

    def assign_group_to_documents_tags(self, doc_ids: List[str], group_id: Optional[str] = None) -> Dict[str, Any]:
        """選択した複数ドキュメントに設定されているタグ群の所属グループを一括変更"""
        if not doc_ids:
            return {"updated_tag_count": 0, "affected_tags": [], "group": None}

        conn = self.get_connection()
        cur = conn.cursor()
        try:
            group_dict = None
            if group_id:
                group = cur.execute("SELECT * FROM tag_groups WHERE id = ?", (group_id,)).fetchone()
                if not group:
                    raise ValueError(f"Tag group '{group_id}' not found")
                group_dict = dict(group)

            placeholders = ",".join(["?"] * len(doc_ids))
            tag_rows = cur.execute(
                f"""
                SELECT DISTINCT t.id, t.name, t.color, t.group_id
                FROM tags t
                JOIN document_tags dt ON t.id = dt.tag_id
                WHERE dt.document_id IN ({placeholders})
                """,
                doc_ids
            ).fetchall()

            affected_tags = [dict(r) for r in tag_rows]
            if not affected_tags:
                return {
                    "updated_tag_count": 0,
                    "affected_tags": [],
                    "group": group_dict
                }

            tag_ids = [t["id"] for t in affected_tags]
            tag_placeholders = ",".join(["?"] * len(tag_ids))
            cur.execute(
                f"UPDATE tags SET group_id = ? WHERE id IN ({tag_placeholders})",
                [group_id] + tag_ids
            )
            conn.commit()

            for t in affected_tags:
                t["group_id"] = group_id
                t["group_name"] = group_dict["name"] if group_dict else None

            return {
                "updated_tag_count": len(affected_tags),
                "affected_tags": affected_tags,
                "group": group_dict
            }
        finally:
            conn.close()

    def assign_documents_to_group(self, doc_ids: List[str], group_id: str, tag_id: Optional[str] = None) -> Dict[str, Any]:
        """後方互換用: 選択した複数ドキュメントのタグ群にグループを一括設定"""
        return self.assign_group_to_documents_tags(doc_ids, group_id)

    def list_documents(
        self,
        tag_id: Optional[str] = None,
        tag_ids: Optional[List[str]] = None,
        tag_op: str = "and",
        group_id: Optional[str] = None,
        unassigned: bool = False,
        sort_by: str = "updated_at",
        order: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            sort_field = "d.updated_at"
            if sort_by == "created_at":
                sort_field = "d.created_at"
            elif sort_by == "title":
                sort_field = "d.title"
            elif sort_by == "source_type":
                sort_field = "d.source_type"

            sort_order = "DESC" if order.lower() == "desc" else "ASC"

            query = f"""
                SELECT d.id, d.title, d.source_type, d.source_filename, d.total_pages,
                       d.created_at, d.updated_at
                FROM documents d
            """
            where_clauses = []
            params = []

            if unassigned or tag_id == "unclassified" or tag_id == "unassigned":
                where_clauses.append("NOT EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id)")
            elif tag_id:
                where_clauses.append("EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id AND dt.tag_id = ?)")
                params.append(tag_id)
            elif tag_ids and len(tag_ids) > 0:
                if (tag_op or "").lower() == "or":
                    placeholders = ",".join(["?"] * len(tag_ids))
                    where_clauses.append(f"EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id AND dt.tag_id IN ({placeholders}))")
                    params.extend(tag_ids)
                else:  # "and" (default)
                    for tid in tag_ids:
                        where_clauses.append("EXISTS (SELECT 1 FROM document_tags dt WHERE dt.document_id = d.id AND dt.tag_id = ?)")
                        params.append(tid)
            elif group_id:
                where_clauses.append("EXISTS (SELECT 1 FROM document_tags dt JOIN tags t ON dt.tag_id = t.id WHERE dt.document_id = d.id AND t.group_id = ?)")
                params.append(group_id)

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += f" ORDER BY {sort_field} {sort_order}"

            if limit is not None and limit > 0:
                query += " LIMIT ? OFFSET ?"
                params.extend([limit, offset])

            rows = cur.execute(query, params).fetchall()
            doc_list = [dict(r) for r in rows]

            # 各ドキュメントに紐づくタグを一括取得
            doc_ids = [d["id"] for d in doc_list]
            tags_map = {d_id: [] for d_id in doc_ids}
            if doc_ids:
                placeholders = ",".join(["?"] * len(doc_ids))
                tag_rows = cur.execute(
                    f"""
                    SELECT dt.document_id, t.id, t.name, t.color, t.group_id, tg.name AS group_name
                    FROM document_tags dt
                    JOIN tags t ON dt.tag_id = t.id
                    LEFT JOIN tag_groups tg ON t.group_id = tg.id
                    WHERE dt.document_id IN ({placeholders})
                    ORDER BY tg.order_idx ASC, t.name ASC
                    """,
                    doc_ids
                ).fetchall()
                for tr in tag_rows:
                    tags_map[tr["document_id"]].append({
                        "id": tr["id"],
                        "name": tr["name"],
                        "color": tr["color"],
                        "group_id": tr["group_id"],
                        "group_name": tr["group_name"]
                    })

            for d in doc_list:
                d["tags"] = tags_map.get(d["id"], [])

            return doc_list
        finally:
            conn.close()

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            doc_row = cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not doc_row:
                return None
            return dict(doc_row)
        finally:
            conn.close()

    def get_document_bundle(self, doc_id: str) -> Optional[Dict[str, Any]]:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            doc_row = cur.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if not doc_row:
                return None
            doc = dict(doc_row)

            # ドキュメントのタグ一覧を取得
            tag_rows = cur.execute(
                """
                SELECT t.id, t.name, t.color, t.group_id, tg.name AS group_name
                FROM document_tags dt
                JOIN tags t ON dt.tag_id = t.id
                LEFT JOIN tag_groups tg ON t.group_id = tg.id
                WHERE dt.document_id = ?
                ORDER BY tg.order_idx ASC, t.name ASC
                """,
                (doc_id,)
            ).fetchall()
            doc["tags"] = [dict(tr) for tr in tag_rows]

            sec_rows = cur.execute(
                "SELECT * FROM sections WHERE document_id = ? ORDER BY order_idx ASC",
                (doc_id,)
            ).fetchall()
            sections = [dict(s) for s in sec_rows]

            sec_map = {s["id"]: {**s, "children": []} for s in sections}
            root_sections = []
            for s in sections:
                pid = s["parent_id"]
                if pid and pid in sec_map:
                    sec_map[pid]["children"].append(sec_map[s["id"]])
                else:
                    root_sections.append(sec_map[s["id"]])

            blk_rows = cur.execute(
                "SELECT * FROM content_blocks WHERE document_id = ? ORDER BY reading_order ASC",
                (doc_id,)
            ).fetchall()
            blocks = [dict(b) for b in blk_rows]

            tbl_rows = cur.execute(
                "SELECT * FROM document_tables WHERE document_id = ? ORDER BY reading_order ASC",
                (doc_id,)
            ).fetchall()
            tables = []
            for t in tbl_rows:
                td = dict(t)
                try:
                    td["grid"] = json.loads(td["grid_json"])
                except Exception:
                    td["grid"] = {}
                tables.append(td)

            fig_rows = cur.execute(
                "SELECT * FROM document_figures WHERE document_id = ? ORDER BY reading_order ASC",
                (doc_id,)
            ).fetchall()
            figures = [dict(f) for f in fig_rows]

            ann_rows = cur.execute(
                "SELECT * FROM annotations WHERE document_id = ? ORDER BY created_at ASC",
                (doc_id,)
            ).fetchall()
            annotations = [dict(a) for a in ann_rows]

            return {
                "document": doc,
                "sections": sections,
                "section_tree": root_sections,
                "blocks": blocks,
                "tables": tables,
                "figures": figures,
                "annotations": annotations
            }
        finally:
            conn.close()

    def save_document_bundle(self, bundle: Dict[str, Any]) -> str:
        doc = bundle.get("document", {})
        doc_id = doc.get("id") or f"doc-{uuid.uuid4().hex[:12]}"
        title = doc.get("title") or "無題のドキュメント"
        source_type = doc.get("source_type") or "manual"
        source_filename = doc.get("source_filename") or ""
        total_pages = int(doc.get("total_pages") or 0)
        metadata = doc.get("doc_metadata_json")
        if isinstance(metadata, dict):
            metadata_str = json.dumps(metadata, ensure_ascii=False)
        else:
            metadata_str = metadata or "{}"

        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE;")

            cur.execute(
                """
                INSERT INTO documents (id, title, source_type, source_filename, total_pages, doc_metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    source_type = excluded.source_type,
                    source_filename = excluded.source_filename,
                    total_pages = excluded.total_pages,
                    doc_metadata_json = excluded.doc_metadata_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (doc_id, title, source_type, source_filename, total_pages, metadata_str)
            )

            cur.execute("DELETE FROM annotations WHERE document_id = ?", (doc_id,))
            cur.execute("DELETE FROM document_tables WHERE document_id = ?", (doc_id,))
            cur.execute("DELETE FROM document_figures WHERE document_id = ?", (doc_id,))
            # block_id is UNINDEXED in FTS5. Its row-delete trigger scans the full
            # FTS table once per block, so bulk-delete this document's FTS rows once.
            cur.execute("DROP TRIGGER IF EXISTS trg_notebook_blocks_ad;")
            cur.execute("DELETE FROM fts_blocks WHERE document_id = ?", (doc_id,))
            cur.execute("DELETE FROM content_blocks WHERE document_id = ?", (doc_id,))
            cur.execute("""
                CREATE TRIGGER trg_notebook_blocks_ad AFTER DELETE ON content_blocks
                BEGIN
                    DELETE FROM fts_blocks WHERE block_id = old.id;
                END;
            """)
            cur.execute("DELETE FROM sections WHERE document_id = ?", (doc_id,))

            sections = bundle.get("sections", [])
            valid_section_ids = set()
            sec_id_map: Dict[str, str] = {}
            for idx, sec in enumerate(sections):
                orig_id = sec.get("id") or f"sec-{idx}"
                if orig_id.startswith(f"sec-{doc_id}"):
                    new_id = orig_id
                else:
                    new_id = f"sec-{doc_id}-{orig_id.replace('sec-', '')}"
                sec_id_map[orig_id] = new_id
                valid_section_ids.add(new_id)

            for idx, sec in enumerate(sections):
                orig_id = sec.get("id") or f"sec-{idx}"
                s_id = sec_id_map.get(orig_id, f"sec-{doc_id}-{idx}")
                orig_pid = sec.get("parent_id")
                mapped_pid = sec_id_map.get(orig_pid) if orig_pid else None
                p_id = mapped_pid if (mapped_pid in valid_section_ids and mapped_pid != s_id) else None
                s_title = sec.get("title") or f"見出し {idx+1}"
                s_level = int(sec.get("level") or 1)
                s_order = int(sec.get("order_idx", idx))
                s_page = sec.get("page_number")
                cur.execute(
                    """
                    INSERT INTO sections (id, document_id, parent_id, title, level, order_idx, page_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (s_id, doc_id, p_id, s_title, s_level, s_order, s_page)
                )

            blocks = bundle.get("blocks", [])
            valid_block_ids = set()
            blk_id_map: Dict[str, str] = {}
            for idx, blk in enumerate(blocks):
                orig_bid = blk.get("id") or f"blk-{idx}"
                if orig_bid.startswith(f"blk-{doc_id}"):
                    b_id = orig_bid
                else:
                    b_id = f"blk-{doc_id}-{orig_bid.replace('blk-', '')}-{uuid.uuid4().hex[:4]}"
                blk_id_map[orig_bid] = b_id
                valid_block_ids.add(b_id)
                
                orig_sec_id = blk.get("section_id")
                mapped_sec_id = sec_id_map.get(orig_sec_id) if orig_sec_id else None
                sec_id = mapped_sec_id if mapped_sec_id in valid_section_ids else None

                b_type = blk.get("block_type") or "paragraph"
                t_content = blk.get("text_content") or ""
                h_content = blk.get("html_content") or f"<p>{t_content}</p>"
                r_order = int(blk.get("reading_order", idx + 1))
                p_num = blk.get("page_number")
                bbox = blk.get("bbox_json")
                if isinstance(bbox, (dict, list)):
                    bbox = json.dumps(bbox)

                cur.execute(
                    """
                    INSERT INTO content_blocks (id, document_id, section_id, block_type, text_content, html_content, reading_order, page_number, bbox_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (b_id, doc_id, sec_id, b_type, t_content, h_content, r_order, p_num, bbox)
                )

            tables = bundle.get("tables", [])
            for idx, tbl in enumerate(tables):
                t_id = tbl.get("id") or f"tbl-{uuid.uuid4().hex[:12]}"
                orig_sec_id = tbl.get("section_id")
                mapped_sec_id = sec_id_map.get(orig_sec_id) if orig_sec_id else None
                sec_id = mapped_sec_id if mapped_sec_id in valid_section_ids else None

                caption = tbl.get("caption") or f"表 {idx+1}"
                row_cnt = int(tbl.get("row_count") or len(tbl.get("grid", {}).get("rows", [])))
                col_cnt = int(tbl.get("col_count") or 0)
                grid = tbl.get("grid_json") or tbl.get("grid") or {}
                if isinstance(grid, (dict, list)):
                    grid_str = json.dumps(grid, ensure_ascii=False)
                else:
                    grid_str = str(grid)
                r_order = int(tbl.get("reading_order", idx + 1))
                p_num = tbl.get("page_number")

                cur.execute(
                    """
                    INSERT INTO document_tables (id, document_id, section_id, caption, row_count, col_count, grid_json, reading_order, page_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (t_id, doc_id, sec_id, caption, row_cnt, col_cnt, grid_str, r_order, p_num)
                )

            figures = bundle.get("figures", [])
            for idx, fig in enumerate(figures):
                f_id = fig.get("id") or f"fig-{uuid.uuid4().hex[:12]}"
                orig_sec_id = fig.get("section_id")
                mapped_sec_id = sec_id_map.get(orig_sec_id) if orig_sec_id else None
                sec_id = mapped_sec_id if mapped_sec_id in valid_section_ids else None

                caption = fig.get("caption") or f"図 {idx+1}"
                f_path = fig.get("file_path") or ""
                f_hash = fig.get("file_hash") or ""
                f_size = float(fig.get("file_size_kb") or 0.0)
                w = int(fig.get("width") or 0)
                h = int(fig.get("height") or 0)
                r_order = int(fig.get("reading_order", idx + 1))
                p_num = fig.get("page_number")
                is_custom = bool(fig.get("is_custom", False))

                cur.execute(
                    """
                    INSERT INTO document_figures (id, document_id, section_id, caption, file_path, file_hash, file_size_kb, width, height, reading_order, page_number, is_custom)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (f_id, doc_id, sec_id, caption, f_path, f_hash, f_size, w, h, r_order, p_num, is_custom)
                )

            annotations = bundle.get("annotations", [])
            for idx, ann in enumerate(annotations):
                a_id = ann.get("id") or f"ann-{uuid.uuid4().hex[:12]}"
                orig_bid = ann.get("block_id")
                mapped_bid = blk_id_map.get(orig_bid) if orig_bid else None
                b_id = mapped_bid if mapped_bid in valid_block_ids else None

                kind = ann.get("kind") or "footnote"
                anchor = ann.get("anchor_text") or ""
                target = ann.get("target_value") or ""

                cur.execute(
                    """
                    INSERT INTO annotations (id, document_id, block_id, kind, anchor_text, target_value)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (a_id, doc_id, b_id, kind, anchor, target)
                )

            conn.commit()
            return doc_id
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def delete_document(self, doc_id: str) -> bool:
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def rename_document(self, doc_id: str, new_title: str) -> bool:
        title = new_title.strip() if new_title else ""
        if not title:
            raise ValueError("ドキュメントタイトルを空にすることはできません。")
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE documents SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, doc_id)
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def search_blocks(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            q_clean = f'"{query.strip()}"'
            rows = cur.execute(
                """
                SELECT f.block_id, f.document_id, f.title as section_title,
                       snippet(fts_blocks, 3, '<mark>', '</mark>', '...', 12) as snippet,
                       cb.block_type, cb.reading_order, cb.page_number,
                       d.title as doc_title, d.source_type
                FROM fts_blocks f
                JOIN content_blocks cb ON f.block_id = cb.id
                JOIN documents d ON f.document_id = d.id
                WHERE fts_blocks MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (q_clean, limit)
            ).fetchall()
            results = []
            for r in rows:
                item = dict(r)
                if "snippet" in item and item["snippet"]:
                    item["snippet"] = sanitize_fts_snippet(item["snippet"])
                results.append(item)
            return results
        except Exception:
            return []
        finally:
            conn.close()

