import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from app.core.db import get_db_connection


def build_kaisor_database_profile(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """SQLite DB内の全データをKaisoRTextEditorのDatabaseProfile形式に整形"""
    conn = get_db_connection(db_path)
    cur = conn.cursor()

    # 1. tree_nodes 取得
    node_rows = cur.execute("SELECT * FROM tree_nodes ORDER BY sort_order ASC").fetchall()
    # 2. content_blocks 取得
    block_rows = cur.execute("SELECT * FROM content_blocks ORDER BY reading_order ASC").fetchall()
    # 3. tables 取得
    table_rows = cur.execute("SELECT * FROM tables").fetchall()
    # 4. figures 取得
    figure_rows = cur.execute("SELECT * FROM figures").fetchall()
    # 5. annotations 取得
    annotation_rows = cur.execute("SELECT * FROM annotations ORDER BY created_at ASC").fetchall()

    conn.close()

    # マップ作成
    tables_by_block = {t["block_id"]: dict(t) for t in table_rows}
    figures_by_block = {f["block_id"]: dict(f) for f in figure_rows}
    annotations_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for a in annotation_rows:
        annotations_by_node.setdefault(a["node_id"], []).append({
            "id": a["id"],
            "nodeId": a["node_id"],
            "text": a["text_excerpt"],
            "comment": a["comment"] or "",
            "color": a["color"] or "#ffeb3b",
            "anchorId": a["anchor_id"] or "",
            "noteTitle": "",
            "createdAt": str(a["created_at"])
        })

    # ブロックをノード毎にグループ化
    blocks_by_node: Dict[str, List[Dict[str, Any]]] = {}
    for b in block_rows:
        if b["node_id"]:
            blocks_by_node.setdefault(b["node_id"], []).append(dict(b))

    nodes_dict: Dict[str, Any] = {}
    children_map: Dict[Optional[str], List[str]] = {}

    for nr in node_rows:
        nid = nr["id"]
        pid = nr["parent_id"]
        children_map.setdefault(pid, []).append(nid)

        # ブロックからHTMLおよび表データを組み立て
        node_blocks = blocks_by_node.get(nid, [])
        rich_html_parts = []
        spreadsheet_data = None

        for nb in node_blocks:
            bid = nb["id"]
            b_type = nb["block_type"]
            text = nb["text_content"] or ""

            if b_type == "table" and bid in tables_by_block:
                tbl = tables_by_block[bid]
                headers = json.loads(tbl["headers_json"]) if tbl["headers_json"] else []
                matrix = json.loads(tbl["matrix_json"]) if tbl["matrix_json"] else []
                spreadsheet_data = {
                    "headers": headers,
                    "rows": matrix,
                    "hasHeaderRow": bool(tbl["has_header_row"]),
                    "lockHeader": False
                }
                # HTML内にもプレビュー表示用テーブルを配置
                table_html = "<table class='border-collapse border border-slate-300 my-4 w-full'><thead><tr>"
                for h in headers:
                    table_html += f"<th class='border border-slate-300 bg-slate-100 p-2 text-left'>{h}</th>"
                table_html += "</tr></thead><tbody>"
                for row in matrix:
                    table_html += "<tr>"
                    for cell in row:
                        table_html += f"<td class='border border-slate-300 p-2'>{cell.get('value', '')}</td>"
                    table_html += "</tr>"
                table_html += "</tbody></table>"
                rich_html_parts.append(table_html)

            elif b_type == "figure" and bid in figures_by_block:
                fig = figures_by_block[bid]
                img_p = fig["image_path"]
                fname = Path(img_p).name
                cap = fig["caption"] or "図版"
                rich_html_parts.append(f"""
                <figure class="my-4 text-center">
                    <img src="/media/{fname}" alt="{cap}" class="max-w-full rounded shadow mx-auto max-h-96">
                    <figcaption class="text-sm text-slate-500 font-medium mt-2">{cap}</figcaption>
                </figure>
                """)

            elif b_type.startswith("h"):
                rich_html_parts.append(f"<{b_type} class='font-bold my-3'>{text}</{b_type}>")

            elif b_type == "footnote":
                rich_html_parts.append(f"<p class='text-xs text-slate-500 border-t border-slate-200 pt-1 mt-4'><em>{text}</em></p>")

            else:
                rich_html_parts.append(f"<p class='my-2 leading-relaxed'>{text}</p>")

        combined_html = "\n".join(rich_html_parts) if rich_html_parts else f"<p>{nr['title']}</p>"

        # ノードタイプ決定
        ntype = nr["node_type"] or "rich"
        if spreadsheet_data and not rich_html_parts:
            ntype = "spreadsheet"

        nodes_dict[nid] = {
            "id": nid,
            "notebookId": nr["notebook_id"],
            "parentId": pid,
            "title": nr["title"],
            "type": ntype,
            "icon": nr["icon"],
            "colorBadge": nr["color_badge"],
            "tags": json.loads(nr["tags_json"]) if nr["tags_json"] else [],
            "created": str(nr["created_at"]),
            "updated": str(nr["updated_at"]),
            "isLocked": bool(nr["is_locked"]),
            "sentenceBookmarks": annotations_by_node.get(nid, []),
            "content": {
                "richHtml": combined_html,
                "spreadsheet": spreadsheet_data,
                "plainText": "\n".join(b["text_content"] for b in node_blocks if b.get("text_content"))
            },
            "children": []  # 後で設定
        }

    # 各ノードの子ノードID配列を設定
    for nid, node in nodes_dict.items():
        node["children"] = children_map.get(nid, [])

    # ルートノードID（parent_id が None）
    root_node_ids = children_map.get(None, [])

    notebooks = [
        {
            "id": "nb-default",
            "name": "統合個人ナレッジ",
            "color": "#3b82f6",
            "bgClass": "bg-blue-500",
            "borderClass": "border-blue-500",
            "description": "Word (.docx) および PDF (NDLOCR-Lite) から取り込んだ知識ベース",
            "nodeIds": root_node_ids
        }
    ]

    profile = {
        "id": "profile-main",
        "name": "個人知識ベース (SQLite)",
        "createdAt": "2026-09-11T00:00:00.000Z",
        "updatedAt": "2026-09-11T00:00:00.000Z",
        "storageType": "local_folder",
        "notebooks": notebooks,
        "nodes": nodes_dict,
        "tags": [],
        "sentenceBookmarks": [a for sublist in annotations_by_node.values() for a in sublist],
        "figureCaptions": [],
        "activeNotebookId": "nb-default",
        "activeNodeId": root_node_ids[0] if root_node_ids else None
    }

    return profile
