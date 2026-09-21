-- ====================================================================
-- 個人知識基盤システム 統合SQLiteデータベーススキーマ
-- (KaisoRTextEditor 階層エディタ & OCR-REPOS & 将来のNotebookLM/RAG 共通)
-- ====================================================================

PRAGMA foreign_keys = ON;

-- 1. 文書マスター (PDF, docx, 手動作成ノートなど)
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    file_path TEXT,
    source_type TEXT CHECK(source_type IN ('pdf_ocr', 'docx', 'manual', 'web')),
    page_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT
);

-- 2. ページ情報 (主にPDF用。docx等の単一文書でも1ページとして記録)
CREATE TABLE IF NOT EXISTS pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    width INTEGER,
    height INTEGER,
    image_path TEXT,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- 3. 階層エディタ用ノード (KaisoRTextEditorのツリー構造 TreeNode と 1:1 対応)
CREATE TABLE IF NOT EXISTS tree_nodes (
    id TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL,
    parent_id TEXT,
    title TEXT NOT NULL,
    node_type TEXT CHECK(node_type IN ('rich', 'spreadsheet', 'code', 'bookmark', 'encrypted')) DEFAULT 'rich',
    sort_order INTEGER DEFAULT 0,
    icon TEXT,
    color_badge TEXT,
    tags_json TEXT,
    is_locked INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(parent_id) REFERENCES tree_nodes(id) ON DELETE CASCADE
);

-- 4. 構造化コンテンツブロック (見出し、本文段落、表、図、脚注などの素片)
CREATE TABLE IF NOT EXISTS content_blocks (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    page_id TEXT,
    node_id TEXT,
    block_type TEXT CHECK(block_type IN ('h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'body', 'table', 'figure', 'map', 'footnote')),
    reading_order INTEGER NOT NULL,
    text_content TEXT,
    html_content TEXT,
    bbox_json TEXT,
    confidence REAL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE SET NULL,
    FOREIGN KEY(page_id) REFERENCES pages(id) ON DELETE SET NULL,
    FOREIGN KEY(node_id) REFERENCES tree_nodes(id) ON DELETE CASCADE
);

-- 5. 表データ (KaisoRTextEditorの SpreadsheetData と完全互換)
CREATE TABLE IF NOT EXISTS tables (
    id TEXT PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    row_count INTEGER NOT NULL,
    col_count INTEGER NOT NULL,
    headers_json TEXT,
    matrix_json TEXT NOT NULL,
    has_header_row INTEGER DEFAULT 1,
    FOREIGN KEY(block_id) REFERENCES content_blocks(id) ON DELETE CASCADE
);

-- 6. 図版・画像・地図 (OCRでクロップされた画像やdocx埋め込み画像)
CREATE TABLE IF NOT EXISTS figures (
    id TEXT PRIMARY KEY,
    block_id TEXT NOT NULL UNIQUE,
    figure_type TEXT CHECK(figure_type IN ('figure', 'table_img', 'photo', 'map', 'diagram')) DEFAULT 'figure',
    image_path TEXT NOT NULL,
    caption TEXT,
    anchor_id TEXT,
    FOREIGN KEY(block_id) REFERENCES content_blocks(id) ON DELETE CASCADE
);

-- 7. 注釈・マーカー・ブックマーク (KaisoRエディタの SentenceBookmark と完全連動)
CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    block_id TEXT,
    anchor_id TEXT,
    text_excerpt TEXT NOT NULL,
    comment TEXT,
    color TEXT DEFAULT '#ffeb3b',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(node_id) REFERENCES tree_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(block_id) REFERENCES content_blocks(id) ON DELETE SET NULL
);

-- 8. 全文検索インデックス (SQLite FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS fts_blocks USING fts5(
    block_id UNINDEXED,
    document_id UNINDEXED,
    node_id UNINDEXED,
    title,
    text_content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS trg_blocks_ai AFTER INSERT ON content_blocks
WHEN new.text_content IS NOT NULL AND length(trim(new.text_content)) > 0
BEGIN
    INSERT INTO fts_blocks(block_id, document_id, node_id, title, text_content)
    VALUES (
        new.id,
        new.document_id,
        new.node_id,
        (SELECT title FROM tree_nodes WHERE id = new.node_id),
        new.text_content
    );
END;

CREATE TRIGGER IF NOT EXISTS trg_blocks_ad AFTER DELETE ON content_blocks
BEGIN
    DELETE FROM fts_blocks WHERE block_id = old.id;
END;

CREATE TRIGGER IF NOT EXISTS trg_blocks_au AFTER UPDATE ON content_blocks
BEGIN
    DELETE FROM fts_blocks WHERE block_id = old.id;
    INSERT INTO fts_blocks(block_id, document_id, node_id, title, text_content)
    VALUES (
        new.id,
        new.document_id,
        new.node_id,
        (SELECT title FROM tree_nodes WHERE id = new.node_id),
        new.text_content
    );
END;

-- 高速化インデックス
CREATE INDEX IF NOT EXISTS idx_content_blocks_doc ON content_blocks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_figures_doc ON document_figures(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tables_doc ON document_tables(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tags_doc ON document_tags(document_id);
CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id);
CREATE INDEX IF NOT EXISTS idx_tags_group ON tags(group_id);
CREATE INDEX IF NOT EXISTS idx_sections_doc ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_documents_updated ON documents(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_title ON documents(title);
