import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.core.db import get_db_connection, DEFAULT_DB_PATH

# 簡易な日英キーワード同義語マッピング
SYNONYMS = {
    "水": ["water", "水"],
    "危機": ["crisis", "bankruptcy", "危機"],
    "システム": ["system", "システム"],
    "知識": ["knowledge", "知識"],
    "基盤": ["platform", "base", "基盤"],
    "報告": ["report", "foreword", "報告"],
    "開発": ["development", "開発"]
}


class KnowledgeRetriever:
    """共通SQLiteデータベースから関連ブロックを検索し、元PDF/Wordの出典情報を取得するリトリーバー"""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH

    def search_sources(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        conn = get_db_connection(self.db_path)
        cur = conn.cursor()

        raw_words = re.findall(r"[\w\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]+", query)
        search_terms = set()
        for w in raw_words:
            search_terms.add(w.lower())
            for k, syns in SYNONYMS.items():
                if k in w:
                    for s in syns:
                        search_terms.add(s.lower())

        results = []
        seen_block_ids = set()

        for term in search_terms:
            if len(term) < 2:
                continue

            # 1. FTS5 全文検索
            try:
                fts_query = f'"{term}"'
                rows = cur.execute(
                    """
                    SELECT cb.id as block_id, cb.document_id, cb.page_id, cb.node_id,
                           cb.block_type, cb.text_content, cb.bbox_json,
                           doc.title as doc_title, doc.source_type,
                           p.page_number, p.image_path as page_image_path,
                           tn.title as node_title
                    FROM fts_blocks f
                    JOIN content_blocks cb ON f.block_id = cb.id
                    LEFT JOIN documents doc ON cb.document_id = doc.id
                    LEFT JOIN pages p ON cb.page_id = p.id
                    LEFT JOIN tree_nodes tn ON cb.node_id = tn.id
                    WHERE fts_blocks MATCH ?
                    LIMIT ?
                    """,
                    (fts_query, top_k)
                ).fetchall()

                for r in rows:
                    bid = r["block_id"]
                    if bid not in seen_block_ids:
                        seen_block_ids.add(bid)
                        bbox = json.loads(r["bbox_json"]) if r["bbox_json"] else None
                        results.append({
                            "block_id": bid,
                            "document_title": r["doc_title"] or "未分類",
                            "source_type": r["source_type"] or "unknown",
                            "node_title": r["node_title"] or "",
                            "page_number": r["page_number"] or 1,
                            "page_image_path": r["page_image_path"],
                            "text_content": r["text_content"],
                            "bbox": bbox,
                            "block_type": r["block_type"]
                        })
            except Exception:
                pass

            # 2. LIKE 部分一致フォールバック
            like_pat = f"%{term}%"
            rows = cur.execute(
                """
                SELECT cb.id as block_id, cb.document_id, cb.page_id, cb.node_id,
                       cb.block_type, cb.text_content, cb.bbox_json,
                       doc.title as doc_title, doc.source_type,
                       p.page_number, p.image_path as page_image_path,
                       tn.title as node_title
                FROM content_blocks cb
                LEFT JOIN documents doc ON cb.document_id = doc.id
                LEFT JOIN pages p ON cb.page_id = p.id
                LEFT JOIN tree_nodes tn ON cb.node_id = tn.id
                WHERE cb.text_content LIKE ?
                LIMIT ?
                """,
                (like_pat, top_k)
            ).fetchall()

            for r in rows:
                bid = r["block_id"]
                if bid not in seen_block_ids:
                    seen_block_ids.add(bid)
                    bbox = json.loads(r["bbox_json"]) if r["bbox_json"] else None
                    results.append({
                        "block_id": bid,
                        "document_title": r["doc_title"] or "未分類",
                        "source_type": r["source_type"] or "unknown",
                        "node_title": r["node_title"] or "",
                        "page_number": r["page_number"] or 1,
                        "page_image_path": r["page_image_path"],
                        "text_content": r["text_content"],
                        "bbox": bbox,
                        "block_type": r["block_type"]
                    })
                if len(results) >= top_k:
                    break

        conn.close()
        return results[:top_k]
