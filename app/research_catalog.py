"""Read-only document metadata for the research screen.

No OCR text, indexes, provider calls, or schema changes are needed at this stage.
"""


def get_research_catalog(db_service):
    conn = db_service.get_connection()
    try:
        # Keep groups and memberships in the same read snapshot.
        conn.execute("BEGIN")
        groups = [dict(row) for row in conn.execute("""
            SELECT g.id, g.name, COUNT(DISTINCT t.id) AS tag_count,
                   COUNT(DISTINCT d.id) AS document_count
            FROM tag_groups g
            LEFT JOIN tags t ON t.group_id = g.id
            LEFT JOIN document_tags dt ON dt.tag_id = t.id
            LEFT JOIN documents d ON d.id = dt.document_id
            GROUP BY g.id
            ORDER BY g.order_idx, g.name, g.id
        """)]
        tags = [dict(row) for row in conn.execute("""
            SELECT t.id, t.name, t.group_id, COUNT(DISTINCT d.id) AS document_count
            FROM tags t
            JOIN tag_groups g ON g.id = t.group_id
            LEFT JOIN document_tags dt ON dt.tag_id = t.id
            LEFT JOIN documents d ON d.id = dt.document_id
            GROUP BY t.id
            ORDER BY g.order_idx, g.name, t.name, t.id
        """)]
        documents = [dict(row) for row in conn.execute("""
            SELECT id, title, source_type, source_filename, total_pages, updated_at
            FROM documents ORDER BY title, id
        """)]
        by_id = {doc["id"]: doc for doc in documents}
        for doc in documents:
            doc["group_ids"] = []
            doc["tag_ids"] = []
        for row in conn.execute("""
            SELECT DISTINCT dt.document_id, t.group_id
            FROM document_tags dt
            JOIN tags t ON t.id = dt.tag_id
            JOIN tag_groups g ON g.id = t.group_id
            ORDER BY dt.document_id, t.group_id
        """):
            if row["document_id"] in by_id:
                by_id[row["document_id"]]["group_ids"].append(row["group_id"])
        for row in conn.execute("""
            SELECT DISTINCT dt.document_id, t.id AS tag_id
            FROM document_tags dt
            JOIN tags t ON t.id = dt.tag_id
            JOIN tag_groups g ON g.id = t.group_id
            ORDER BY dt.document_id, t.id
        """):
            if row["document_id"] in by_id:
                by_id[row["document_id"]]["tag_ids"].append(row["tag_id"])
        return {"groups": groups, "tags": tags, "documents": documents}
    finally:
        conn.close()
