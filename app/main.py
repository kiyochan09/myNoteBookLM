import io
import sys
import shutil
import base64
import json
import urllib.parse
import secrets
import uuid
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
import re
import fitz
import cv2
import numpy as np
from PIL import Image as PILImage

# ndlocr_core パス
NDLOCR_CORE_DIR = Path(__file__).resolve().parent / "ocr_pipeline" / "ndlocr_core"
if str(NDLOCR_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(NDLOCR_CORE_DIR))

from app.core.db import get_db_connection, DEFAULT_DB_PATH
from app.db_service import DBService
from app.importers.docx_importer import DocxImporter
from app.importers.odt_importer import OdtImporter
from app.importers.pdf_importer import PdfImporter
from app.importers.ocr_batch_importer import OcrBatchImporter
from app.export_service import ExportService

from app.ocr_pipeline.layout_engine import (
    process_page_layout,
    recognize_custom_regions,
    extract_table_cells_and_matrix,
    detect_table_rule_lines,
    format_lines_into_paragraphs
)
from app.ocr_pipeline.ndlocr_engine import run_ndlocr_on_image

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_DIR = BASE_DIR / "data" / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR = BASE_DIR / "data" / "temp_uploads"
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PDF_CACHE_DIR = BASE_DIR / "data" / "pdf_cache"
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# アプリケーション共通セキュリティトークン (暗号学的一意性 & リロード耐久性)
TOKEN_FILE = BASE_DIR / "data" / ".app_token"
try:
    if TOKEN_FILE.exists():
        APP_SECURITY_TOKEN = TOKEN_FILE.read_text("utf-8").strip()
    else:
        APP_SECURITY_TOKEN = secrets.token_hex(32)
        TOKEN_FILE.write_text(APP_SECURITY_TOKEN, "utf-8")
except Exception:
    APP_SECURITY_TOKEN = secrets.token_hex(32)

def is_allowed_origin(origin: Optional[str]) -> bool:
    if not origin or origin == "null":
        return True
    try:
        parsed = urllib.parse.urlparse(origin)
        return parsed.hostname in {"127.0.0.1", "localhost", "testserver"}
    except Exception:
        return False

app = FastAPI(
    title="MyNotebookLM & OCR Knowledge Base System",
    description="MyNotebookLM 統合ドキュメントエディタ & OCR-REPOS システム",
    version="2.0.0"
)

# 1. セキュリティ検査ミドルウェア (Host / Sec-Fetch-Site / Origin / Token認証)
class SecurityEnforcementMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # (1) Hostヘッダ検証 (DNS Rebinding 防御)
        host_header = request.headers.get("host", "").split(":")[0]
        if host_header and host_header not in {"127.0.0.1", "localhost", "testserver"}:
            return Response("Bad Request: Invalid Host", status_code=400)

        # (2) 更新系リクエストに対する CSRF & 外部Origin防御
        if request.method in ("POST", "PUT", "DELETE", "PATCH"):
            # Sec-Fetch-Site 検証: cross-site からの更新は即時遮断
            sec_fetch_site = request.headers.get("sec-fetch-site")
            if sec_fetch_site == "cross-site":
                return Response("Forbidden: Cross-site request rejected", status_code=403)

            # Origin 検証: 外部ドメインからの更新は即時遮断 (ローカルOrigin/nullは許可)
            origin = request.headers.get("origin")
            if origin and not is_allowed_origin(origin):
                return Response("Forbidden: Origin not allowed", status_code=403)

            # /api/ 配下に対する X-App-Token 検証
            if request.url.path.startswith("/api/"):
                token = request.headers.get("x-app-token", "")
                client_host = request.client.host if request.client else ""
                is_local_client = client_host in {"127.0.0.1", "::1", "localhost", "testclient"}
                if token:
                    if not secrets.compare_digest(token, APP_SECURITY_TOKEN):
                        return Response("Unauthorized: Valid X-App-Token required", status_code=401)
                elif not is_local_client:
                    return Response("Unauthorized: Valid X-App-Token required", status_code=401)

        response = await call_next(request)
        return response

app.add_middleware(SecurityEnforcementMiddleware)

# 2. CORS設定 (ローカル開発・動的ポート対応)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
    allow_headers=["X-Requested-With", "X-App-Token", "Content-Type", "Accept"],
)

app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")

db_service = DBService()
export_service = ExportService(db_service)

# ==========================================
# 画面配信 (HTML Routes) とセキュリティトークン注入
# ==========================================
EDITOR_HTML_PATH = BASE_DIR / "editor_form.html"
OCR_HTML_PATH = BASE_DIR / "ocr_form1_designer.html"
OCR_TEST_HTML_PATH = BASE_DIR / "ocr_test_full.html"

def _inject_security_token(html_text: str) -> str:
    """HTMLの<head>直後にセキュリティトークンを安全に注入"""
    token_script = f'<meta name="app-token" content="{APP_SECURITY_TOKEN}"><script>window.__APP_SECURITY_TOKEN__ = "{APP_SECURITY_TOKEN}";</script>'
    if "<head>" in html_text:
        return html_text.replace("<head>", f"<head>{token_script}", 1)
    elif "<HEAD>" in html_text:
        return html_text.replace("<HEAD>", f"<HEAD>{token_script}", 1)
    return token_script + html_text

def _render_editor():
    if EDITOR_HTML_PATH.exists():
        with open(EDITOR_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=_inject_security_token(f.read()))
    return HTMLResponse(content="<h1>MyNotebookLM Backend Running</h1>")

def _render_ocr():
    if OCR_HTML_PATH.exists():
        with open(OCR_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=_inject_security_token(f.read()))
    return HTMLResponse(content="<h1>OCR Form1 Designer Not Found</h1>", status_code=404)

def _render_ocr_test():
    if OCR_TEST_HTML_PATH.exists():
        with open(OCR_TEST_HTML_PATH, "r", encoding="utf-8") as f:
            return HTMLResponse(content=_inject_security_token(f.read()))
    return HTMLResponse(content="<h1>OCR Test Full Not Found</h1>", status_code=404)

@app.get("/", response_class=HTMLResponse)
def read_root():
    return _render_editor()

@app.get("/editor_form.html", response_class=HTMLResponse)
@app.get("/editor_form", response_class=HTMLResponse)
@app.get("/editor.html", response_class=HTMLResponse)
@app.get("/editor", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
@app.get("/index", response_class=HTMLResponse)
@app.get("/app", response_class=HTMLResponse)
def get_editor_form():
    return _render_editor()

@app.get("/ocr_form1_designer.html", response_class=HTMLResponse)
@app.get("/ocr_form1_designer", response_class=HTMLResponse)
@app.get("/ocr_workspace.html", response_class=HTMLResponse)
@app.get("/ocr_form.html", response_class=HTMLResponse)
@app.get("/ocr.html", response_class=HTMLResponse)
@app.get("/ocr", response_class=HTMLResponse)
@app.get("/ocr_app.html", response_class=HTMLResponse)
def get_ocr_form():
    return _render_ocr()

@app.get("/ocr_test_full.html", response_class=HTMLResponse)
@app.get("/ocr_test_full", response_class=HTMLResponse)
def get_ocr_test_form():
    return _render_ocr_test()


# ==========================================
# MyNotebookLM エディタ & タグ管理 REST API
# ==========================================

class CreateTagRequest(BaseModel):
    name: str
    color: Optional[str] = None
    group_id: Optional[str] = None


class UpdateTagRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None
    group_id: Optional[str] = None


class CreateTagGroupRequest(BaseModel):
    name: str


class AssignTagGroupRequest(BaseModel):
    group_id: Optional[str] = None


class BatchTagAssignRequest(BaseModel):
    doc_ids: List[str]
    tag_id: str


class BatchTagRemoveRequest(BaseModel):
    doc_ids: List[str]
    tag_id: str


class BatchAssignGroupRequest(BaseModel):
    doc_ids: List[str]
    group_id: Optional[str] = None
    tag_id: Optional[str] = None


@app.get("/api/editor/documents")
def list_editor_documents(
    tag_id: Optional[str] = Query(None),
    tag_ids: Optional[str] = Query(None),
    tag_op: str = Query("and"),
    group_id: Optional[str] = Query(None),
    unassigned: bool = Query(False),
    sort_by: str = Query("updated_at"),
    order: str = Query("desc"),
    limit: Optional[int] = Query(None),
    offset: int = Query(0)
):
    """DB内のドキュメント一覧を返却（タグ/グループ/未分類絞り込み・ソート対応、AND/OR演算対応）"""
    parsed_tag_ids = [tid.strip() for tid in tag_ids.split(",") if tid.strip()] if tag_ids else None
    docs = db_service.list_documents(
        tag_id=tag_id,
        tag_ids=parsed_tag_ids,
        tag_op=tag_op,
        group_id=group_id,
        unassigned=unassigned,
        sort_by=sort_by,
        order=order,
        limit=limit,
        offset=offset
    )
    return {"status": "success", "documents": docs, "count": len(docs)}


@app.get("/api/tag-groups")
@app.get("/api/tag_groups")
def list_tag_groups_endpoint():
    """全タググループ一覧と所属タグを取得"""
    data = db_service.list_tag_groups()
    return {"status": "success", **data}


@app.post("/api/tag-groups")
def create_tag_group_endpoint(req: CreateTagGroupRequest):
    """新規タググループを作成"""
    try:
        grp = db_service.create_tag_group(req.name)
        return {"status": "success", "group": grp}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/tag-groups/{group_id}")
def delete_tag_group_endpoint(group_id: str):
    """タググループを削除"""
    success = db_service.delete_tag_group(group_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tag group not found")
    return {"status": "success", "deleted_group_id": group_id}


@app.post("/api/tags/{tag_id}/set-group")
def assign_tag_group_endpoint(tag_id: str, req: AssignTagGroupRequest):
    """タグのグループ所属を設定・変更・解除"""
    try:
        success = db_service.assign_tag_to_group(tag_id, req.group_id)
        if not success:
            raise HTTPException(status_code=404, detail="Tag not found")
        return {"status": "success", "tag_id": tag_id, "group_id": req.group_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tags")
def list_tags_endpoint(group_id: Optional[str] = Query(None)):
    """全タグ一覧を取得（ドキュメント件数・グループ情報・未分類件数付き）"""
    tags = db_service.list_tags(group_id=group_id)
    counts = db_service.get_document_counts()
    return {"status": "success", "tags": tags, **counts}


@app.post("/api/tags")
def create_tag_endpoint(req: CreateTagRequest):
    """新規タグを作成"""
    try:
        tag = db_service.create_tag(req.name, req.color, req.group_id)
        return {"status": "success", "tag": tag}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/tags/{tag_id}")
def update_tag_endpoint(tag_id: str, req: UpdateTagRequest):
    """タグ名や属性を更新"""
    try:
        tag = db_service.update_tag(tag_id, name=req.name, color=req.color, group_id=req.group_id)
        return {"status": "success", "tag": tag}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/tags/{tag_id}")
def delete_tag_endpoint(tag_id: str):
    """タグを削除"""
    success = db_service.delete_tag(tag_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tag not found")
    return {"status": "success", "deleted_tag_id": tag_id}


@app.post("/api/documents/tags/batch-assign")
def batch_assign_tags_endpoint(req: BatchTagAssignRequest):
    """選択した複数ドキュメントにタグを一括付与"""
    try:
        count = db_service.assign_tags_to_documents(req.doc_ids, req.tag_id)
        return {"status": "success", "assigned_count": count}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/documents/tags/batch-remove")
def batch_remove_tags_endpoint(req: BatchTagRemoveRequest):
    """選択した複数ドキュメントからタグを一括解除"""
    try:
        count = db_service.remove_tags_from_documents(req.doc_ids, req.tag_id)
        return {"status": "success", "removed_count": count}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/documents/groups/batch-assign")
def batch_assign_group_endpoint(req: BatchAssignGroupRequest):
    """選択した複数ドキュメントに付与されているタグ群のグループを一括設定"""
    try:
        res = db_service.assign_group_to_documents_tags(req.doc_ids, req.group_id)
        return {"status": "success", **res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/editor/document/{doc_id}")
def get_editor_document(doc_id: str):
    """エディタ用のドキュメント全文・見出しツリー・図表・注釈データを返却"""
    bundle = db_service.get_document_bundle(doc_id)
    if not bundle:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"status": "success", "bundle": bundle}


class SaveDocumentRequest(BaseModel):
    bundle: Dict[str, Any]


@app.post("/api/editor/document/{doc_id}/save")
def save_editor_document(doc_id: str, req: SaveDocumentRequest):
    """エディタで編集されたドキュメント全体を一括トランザクション保存"""
    bundle = req.bundle
    if "document" not in bundle:
        bundle["document"] = {}
    bundle["document"]["id"] = doc_id
    try:
        saved_id = db_service.save_document_bundle(bundle)
        return {"status": "success", "document_id": saved_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/editor/document/{doc_id}")
def delete_editor_document(doc_id: str):
    """ドキュメントを削除"""
    success = db_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "deleted_id": doc_id}


class RenameDocumentRequest(BaseModel):
    title: str


@app.put("/api/editor/document/{doc_id}/rename")
@app.post("/api/editor/document/{doc_id}/rename")
def rename_editor_document(doc_id: str, req: RenameDocumentRequest):
    """ドキュメントのファイル名（タイトル）を変更"""
    new_title = req.title.strip() if req.title else ""
    if not new_title:
        raise HTTPException(status_code=400, detail="ドキュメント名は空にできません")
    try:
        success = db_service.rename_document(doc_id, new_title)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "success", "document_id": doc_id, "title": new_title}
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {".docx", ".odt", ".pdf"}

async def _read_file_safely(file: UploadFile, max_size: int = MAX_UPLOAD_SIZE) -> bytes:
    """ストリーミングサイズを監視しながら安全にファイルを読み込み (50MB超過で即時切断)"""
    content = bytearray()
    chunk_size = 1024 * 1024  # 1MB
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        content.extend(chunk)
        if len(content) > max_size:
            raise HTTPException(status_code=413, detail="ファイルサイズが上限(50MB)を超過しています")
    return bytes(content)

def _verify_magic_number(ext: str, content: bytes) -> None:
    """ファイルのマジックナンバー (先頭シグネチャ) を検証"""
    if ext == ".pdf":
        if not content.startswith(b"%PDF-"):
            raise ValueError("不正なPDFファイルです: マジックナンバー (%PDF-) が一致しません")
    elif ext in (".docx", ".odt"):
        if not content.startswith(b"PK\x03\x04"):
            raise ValueError(f"不正な{ext.upper()}ファイルです: PKヘッダ (PK\\x03\\x04) が一致しません")

def _verify_zip_safety(content: bytes, max_uncompressed_bytes: int = 200 * 1024 * 1024, max_ratio: float = 50.0) -> None:
    """Zip爆弾防御 (展開時合計200MB上限 & 圧縮比率1:50上限)"""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            total_uncompressed = 0
            total_compressed = 0
            for info in zf.infolist():
                total_uncompressed += info.file_size
                total_compressed += info.compress_size
                if total_uncompressed > max_uncompressed_bytes:
                    raise ValueError("Zip爆弾の疑いがあります: 展開後サイズが200MBを超過しています")
            if total_compressed > 0 and (total_uncompressed / total_compressed) > max_ratio:
                raise ValueError("Zip爆弾の疑いがあります: 圧縮比率が異常に高すぎます (>50:1)")
    except zipfile.BadZipFile:
        raise ValueError("Zipアーカイブが破損しているか形式が不正です")

def _process_single_import(filename: str, content: bytes) -> Dict[str, Any]:
    """多層防御を備えた1ファイルの安全なインポートとDB保存"""
    # 1. パストラバーサル防止（純粋なファイル名のみ抽出）
    safe_basename = Path(filename).name.strip()
    if not safe_basename:
        safe_basename = "uploaded_file"
    ext = Path(safe_basename).suffix.lower()

    # 2. 拡張子ホワイトリスト検証
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"許可されていない拡張子です: {ext} (許可: .docx, .odt, .pdf)")

    # 3. マジックナンバー検証
    _verify_magic_number(ext, content)

    # 4. Zip爆弾検証 (DOCX / ODT)
    if ext in (".docx", ".odt"):
        _verify_zip_safety(content)

    # 5. UUID隔離保存 & 境界検証 (is_relative_to)
    unique_name = f"{uuid.uuid4().hex}_{safe_basename}"
    temp_path = (TEMP_UPLOAD_DIR / unique_name).resolve()
    if not temp_path.is_relative_to(TEMP_UPLOAD_DIR.resolve()):
        raise ValueError("セキュリティエラー: 隔離ディレクトリ外への書き込みは拒絶されました")

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        if ext == ".docx":
            importer = DocxImporter(media_dir=MEDIA_DIR)
            bundle = importer.parse_file(temp_path, original_filename=safe_basename)
        elif ext == ".odt":
            importer = OdtImporter(media_dir=MEDIA_DIR)
            bundle = importer.parse_file(temp_path, original_filename=safe_basename)
        elif ext == ".pdf":
            importer = PdfImporter(media_dir=MEDIA_DIR)
            bundle = importer.parse_file(temp_path, original_filename=safe_basename)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # タイトルおよびファイル名からUUIDなどの一時プレフィックスが確実に排除されていることを保証
        expected_title = Path(safe_basename).stem
        if bundle.get("document"):
            bundle["document"]["title"] = expected_title
            bundle["document"]["source_filename"] = safe_basename
        if bundle.get("sections") and len(bundle["sections"]) > 0:
            if bundle["sections"][0].get("level") == 1 and bundle["sections"][0].get("parent_id") is None:
                bundle["sections"][0]["title"] = expected_title

        doc_id = db_service.save_document_bundle(bundle)
        return {
            "document_id": doc_id,
            "title": bundle["document"]["title"],
            "source_type": bundle["document"]["source_type"],
            "sections_count": len(bundle["sections"]),
            "blocks_count": len(bundle["blocks"]),
            "tables_count": len(bundle["tables"]),
            "figures_count": len(bundle["figures"])
        }
    finally:
        # 6. 必ずクリーンアップ (finally unlink)
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/api/editor/import-file")
async def import_editor_file(file: UploadFile = File(...)):
    """ODT / DOCX / PDF ファイルを受信して自動解析しDBに格納"""
    try:
        content = await _read_file_safely(file)
        res = _process_single_import(file.filename, content)
        return {"status": "success", **res}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@app.post("/api/editor/import-files")
async def import_editor_files_batch(files: List[UploadFile] = File(...)):
    """複数の ODT / DOCX / PDF ファイルを一括受信して自動解析しDBに格納"""
    results = []
    success_count = 0
    error_count = 0

    for file in files:
        filename = file.filename
        try:
            content = await _read_file_safely(file)
            res = _process_single_import(filename, content)
            results.append({
                "filename": filename,
                "status": "success",
                **res
            })
            success_count += 1
        except Exception as e:
            results.append({
                "filename": filename,
                "status": "error",
                "error": str(e)
            })
            error_count += 1

    return {
        "status": "success" if error_count == 0 else ("partial" if success_count > 0 else "error"),
        "total": len(files),
        "success_count": success_count,
        "error_count": error_count,
        "results": results
    }


class ImportOcrBatchRequest(BaseModel):
    batch_name: str


@app.get("/api/editor/ocr-batches")
def list_ocr_batches():
    """data/ocr_results 配下のOCRバッチ一覧とページ数を返却"""
    batches = []
    if OCR_RESULTS_DIR.exists():
        for p in OCR_RESULTS_DIR.iterdir():
            if p.is_dir():
                page_files = list(p.glob("page_*.json"))
                page_dirs = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("page_")]
                count = max(len(page_files), len(page_dirs))
                batches.append({
                    "batch_name": p.name,
                    "pages_count": count,
                    "path": str(p),
                    "updated_at": p.stat().st_mtime
                })
    batches.sort(key=lambda x: -x["updated_at"])
    return {"status": "success", "batches": batches}


@app.post("/api/editor/import-ocr-batch")
def import_ocr_batch_endpoint(req: ImportOcrBatchRequest):
    """既存のOCRバッチフォルダ（data/ocr_results/<batch_name>）を結合してDB格納"""
    batch_dir = OCR_RESULTS_DIR / req.batch_name
    if not batch_dir.exists():
        raise HTTPException(status_code=404, detail=f"Batch directory '{req.batch_name}' not found")

    try:
        importer = OcrBatchImporter(media_dir=MEDIA_DIR)
        bundle = importer.parse_batch_dir(batch_dir)
        doc_id = db_service.save_document_bundle(bundle)
        return {
            "status": "success",
            "document_id": doc_id,
            "title": bundle["document"]["title"],
            "source_type": bundle["document"]["source_type"],
            "sections_count": len(bundle["sections"]),
            "blocks_count": len(bundle["blocks"]),
            "tables_count": len(bundle["tables"]),
            "figures_count": len(bundle["figures"])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR Batch Import failed: {str(e)}")


@app.get("/api/editor/document/{doc_id}/export")
def export_document_endpoint(doc_id: str, format: str = Query("md")):
    """指定フォーマット（md / html / docx / odt）でドキュメントを出力ダウンロード"""
    bundle = db_service.get_document_bundle(doc_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Document not found")
    
    title = bundle["document"].get("title", "document")
    safe_title = urllib.parse.quote(re.sub(r'[\/\\:\*\?"<>\|]', '_', title))

    fmt = format.lower()
    if fmt in ["md", "markdown"]:
        md_text = export_service.export_markdown(doc_id)
        return Response(
            content=md_text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_title}.md"}
        )
    elif fmt == "html":
        html_text = export_service.export_html(doc_id)
        return Response(
            content=html_text.encode("utf-8"),
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_title}.html"}
        )
    elif fmt == "docx":
        docx_stream = export_service.export_docx(doc_id)
        return Response(
            content=docx_stream.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_title}.docx"}
        )
    elif fmt == "odt":
        odt_stream = export_service.export_odt(doc_id)
        return Response(
            content=odt_stream.getvalue(),
            media_type="application/vnd.oasis.opendocument.text",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_title}.odt"}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")



@app.post("/api/editor/upload-image")
async def upload_editor_image(file: UploadFile = File(...)):
    """エディタから手動挿入または貼り付けられた画像を保存してURLを返す"""
    content = await file.read()
    ext = Path(file.filename or "image.png").suffix or ".png"
    importer = DocxImporter(media_dir=MEDIA_DIR)
    saved = importer.save_image_bytes(content, original_ext=ext, prefix="user_upload")
    return {
        "status": "success",
        "file_name": saved["file_name"],
        "file_path": saved["file_path"],
        "image_base64": saved.get("image_base64", ""),
        "file_size_kb": saved["file_size_kb"]
    }


@app.get("/api/editor/search")
def search_editor_blocks(q: str):
    """FTS5 全文検索"""
    results = db_service.search_blocks(q)
    return {"query": q, "count": len(results), "results": results}


# ==========================================
# 既存 OCR API 群 (完全互換保持)
# ==========================================
OCR_RESULTS_DIR = BASE_DIR / "data" / "ocr_results"
OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def analyze_pdf_core(doc, filename: str, page_number: int = 1, deck_count: int = 2, orientation: str = "auto", doc_type: str = "japanese"):
    return process_page_layout(
        doc=doc,
        page_number=page_number,
        deck_count=deck_count,
        orientation=orientation,
        doc_type=doc_type,
        filename=filename
    )

def save_figure_images_to_disk(pdf_name: str, page_number: int, figures: List[Dict[str, Any]], regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_dir = OCR_RESULTS_DIR / pdf_name
    page_dir = doc_dir / f"page_{page_number:04d}"
    figures_dir = doc_dir / "figures"
    page_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    reg_img_map = {r.get("id"): r.get("image_base64") for r in regions if r.get("type") == "image" and r.get("image_base64")}
    updated_figures = []
    fig_idx = 0
    for fig in figures:
        fig_dict = dict(fig)
        fig_id = fig_dict.get("id") or f"fig-{fig_idx+1}"
        fig_dict["id"] = fig_id
        fig_dict["page"] = page_number
        b64 = fig_dict.get("image_base64") or reg_img_map.get(fig_id)

        if b64 and isinstance(b64, str) and b64.startswith("data:image"):
            try:
                _, encoded = b64.split(",", 1) if "," in b64 else ("", b64)
                img_bytes = base64.b64decode(encoded)
                img_pil = PILImage.open(io.BytesIO(img_bytes))

                if len(img_bytes) > 500 * 1024:
                    buf = io.BytesIO()
                    img_pil.save(buf, format="JPEG", quality=85)
                    img_bytes = buf.getvalue()
                    ext = "jpg"
                else:
                    ext = "png"

                clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(fig_id))
                fig_name_on_disk = f"figure_{fig_idx:02d}.{ext}"
                fig_page_path = page_dir / fig_name_on_disk
                with open(fig_page_path, "wb") as f:
                    f.write(img_bytes)

                fig_doc_name = f"p{page_number:04d}_{clean_id}.{ext}"
                fig_doc_path = figures_dir / fig_doc_name
                with open(fig_doc_path, "wb") as f:
                    f.write(img_bytes)

                fig_dict["file_name"] = fig_name_on_disk
                fig_dict["file_path"] = str(fig_page_path.relative_to(BASE_DIR)).replace("\\", "/")
                fig_dict["file_size_kb"] = round(len(img_bytes) / 1024.0, 1)
                fig_dict["w"] = img_pil.width
                fig_dict["h"] = img_pil.height
                fig_dict["image_base64"] = b64
            except Exception as ex:
                print(f"Error saving figure PNG for p.{page_number} fig {fig_id}: {ex}")

        updated_figures.append(fig_dict)
        fig_idx += 1

    return updated_figures

def save_ocr_page_to_disk(filename: str, page_number: int, data: Dict[str, Any]) -> bool:
    try:
        pdf_name = Path(filename).stem
        page_dir = OCR_RESULTS_DIR / pdf_name / f"page_{page_number:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        save_dict = {}
        for k, v in data.items():
            if k == "image_base64":
                continue
            save_dict[k] = v
        save_dict["page_number"] = page_number
        save_dict["filename"] = filename

        figures = data.get("figures", [])
        regions = data.get("regions", [])
        if figures or any(r.get("type") == "image" for r in regions):
            if not figures:
                figures = [
                    {
                        "id": r.get("id"),
                        "name": r.get("name") or "図版",
                        "page": page_number,
                        "x": r.get("x", 0),
                        "y": r.get("y", 0),
                        "w": r.get("w", 0),
                        "h": r.get("h", 0),
                        "image_base64": r.get("image_base64")
                    }
                    for r in regions if r.get("type") == "image"
                ]
            updated_figs = save_figure_images_to_disk(pdf_name, page_number, figures, regions)
            save_dict["figures"] = updated_figs

        body_text = data.get("body_text", "")
        # 本文系 regions に最新の body_text を同期し、古いテキストの残存を根絶
        if body_text and regions:
            body_regs = [r for r in regions if r.get("type") in ["body", "paragraph", None]]
            if len(body_regs) == 1:
                body_regs[0]["text"] = body_text
            elif len(body_regs) > 1:
                # 複数領域の場合、各領域にすでにテキストがあれば保持し、全領域が空の場合のみ第1領域に設定
                has_any_text = any(bool(br.get("text")) for br in body_regs)
                if not has_any_text:
                    body_regs[0]["text"] = body_text
            save_dict["regions"] = regions

        json_path = page_dir / "page_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False, indent=2)

        if body_text:
            with open(page_dir / "body_reading_order.txt", "w", encoding="utf-8") as f:
                f.write(body_text)

        if regions:
            with open(page_dir / "auto_regions.json", "w", encoding="utf-8") as f:
                json.dump(regions, f, ensure_ascii=False, indent=2)

        layout_info = {
            "page_number": page_number,
            "filename": filename,
            "orientation": data.get("orientation", "auto"),
            "doc_type": data.get("doc_type", "japanese"),
            "headings": data.get("headings", []),
            "footnotes": data.get("footnotes", []),
            "tables": data.get("tables", []),
            "figures": save_dict.get("figures", [])
        }
        with open(page_dir / "auto_layout.json", "w", encoding="utf-8") as f:
            json.dump(layout_info, f, ensure_ascii=False, indent=2)

        return True
    except Exception as e:
        print(f"Error saving page data to disk: {e}")
        return False

def get_pdf_bytes_sync(filename: str) -> Optional[bytes]:
    clean_stem = Path(filename).stem
    cache_path = PDF_CACHE_DIR / f"{clean_stem}.pdf"
    if cache_path.exists():
        return cache_path.read_bytes()
    candidates = [
        BASE_DIR / "data" / "ocr_results" / clean_stem / f"{clean_stem}.pdf",
        BASE_DIR / "data" / "raw_pdfs" / filename,
        Path(r"C:\Users\natur\Downloads") / filename,
        Path(r"C:\Users\natur\Downloads") / f"{clean_stem}.pdf"
    ]
    for c in candidates:
        if c.exists():
            content = c.read_bytes()
            try:
                cache_path.write_bytes(content)
            except Exception:
                pass
            return content
    return None

def _render_page_only_core(content: bytes, filename: str, page_number: int) -> dict:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDFファイルの展開に失敗しました: {str(e)}")
    try:
        total_pages = len(doc)
        if total_pages == 0:
            raise HTTPException(status_code=400, detail="PDF内にページが存在しません")
        pno = max(0, min(page_number - 1, total_pages - 1))
        page = doc[pno]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("jpg", jpg_quality=88)
        b64_img = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"
        w, h = pix.width, pix.height
        del pix
        del img_bytes
        return {
            "filename": filename,
            "total_pages": total_pages,
            "current_page": pno + 1,
            "image_width": w,
            "image_height": h,
            "image_base64": b64_img
        }
    finally:
        doc.close()

def _analyze_pdf_page_core(content: bytes, filename: str, page_number: int, orientation: str, doc_type: str, deck_num: int) -> dict:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDFファイルの展開に失敗しました: {str(e)}")
    try:
        res = analyze_pdf_core(doc, filename=filename, page_number=page_number, deck_count=deck_num, orientation=orientation, doc_type=doc_type)
        save_ocr_page_to_disk(filename, page_number, res)
        return res
    finally:
        doc.close()

def _recognize_regions_core(content: bytes, filename: str, page_number: int, regions: list, orientation: str, doc_type: str) -> dict:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDFファイルの展開に失敗しました: {str(e)}")
    try:
        updated_data = recognize_custom_regions(doc=doc, page_number=page_number, regions=regions, orientation=orientation, doc_type=doc_type, filename=filename)
        save_ocr_page_to_disk(filename, page_number, updated_data)
        return updated_data
    finally:
        doc.close()

@app.post("/api/ocr/render-page-only")
async def render_page_only(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(None),
    page_number: int = Form(1)
):
    target_filename = file.filename if (file and file.filename) else filename
    if not target_filename or not target_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFファイルを指定してください (.pdf)")

    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="PDFファイルの内容が空です")
        clean_stem = Path(target_filename).stem
        try:
            (PDF_CACHE_DIR / f"{clean_stem}.pdf").write_bytes(content)
        except Exception:
            pass
    else:
        content = await run_in_threadpool(get_pdf_bytes_sync, target_filename)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDFファイル「{target_filename}」がサーバー上に見つかりません。初回はファイルを選択して開いてください。")

    res = await run_in_threadpool(_render_page_only_core, content, target_filename, page_number)
    return res

@app.post("/api/ocr/analyze-pdf-page")
async def analyze_pdf_page(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(None),
    page_number: int = Form(1),
    orientation: str = Form("auto"),
    doc_type: str = Form("japanese"),
    deck: str = Form("2")
):
    target_filename = file.filename if (file and file.filename) else filename
    if not target_filename or not target_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFファイルを指定してください (.pdf)")

    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="PDFファイルの内容が空です")
        clean_stem = Path(target_filename).stem
        try:
            (PDF_CACHE_DIR / f"{clean_stem}.pdf").write_bytes(content)
        except Exception:
            pass
    else:
        content = await run_in_threadpool(get_pdf_bytes_sync, target_filename)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDFファイル「{target_filename}」がサーバー上に見つかりません。")

    try:
        deck_num = int(re.sub(r"\D", "", deck)) if re.sub(r"\D", "", deck) else 2
    except Exception:
        deck_num = 2

    res = await run_in_threadpool(_analyze_pdf_page_core, content, target_filename, page_number, orientation, doc_type, deck_num)
    return res

@app.post("/api/ocr/recognize-regions")
async def recognize_regions_endpoint(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(None),
    page_number: int = Form(1),
    regions_json: str = Form(...),
    orientation: str = Form("auto"),
    doc_type: str = Form("japanese")
):
    target_filename = file.filename if (file and file.filename) else filename
    if not target_filename or not target_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFファイルを指定してください (.pdf)")

    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="PDFファイルの内容が空です")
        clean_stem = Path(target_filename).stem
        try:
            (PDF_CACHE_DIR / f"{clean_stem}.pdf").write_bytes(content)
        except Exception:
            pass
    else:
        content = await run_in_threadpool(get_pdf_bytes_sync, target_filename)
        if not content:
            raise HTTPException(status_code=404, detail=f"PDFファイル「{target_filename}」がサーバー上に見つかりません。")

    try:
        regions = json.loads(regions_json)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid regions JSON: {e}")

    updated_data = await run_in_threadpool(_recognize_regions_core, content, target_filename, page_number, regions, orientation, doc_type)
    return updated_data

class ExtractTableGridRequest(BaseModel):
    filename: str
    page_number: int
    region: Dict[str, Any]
    doc_type: str = "japanese"
    orientation: str = "auto"

@app.post("/api/ocr/extract-table-grid")
def extract_table_grid_endpoint(req: ExtractTableGridRequest):
    pdf_stem = Path(req.filename).stem
    page_dir = OCR_RESULTS_DIR / pdf_stem / f"page_{req.page_number:04d}"
    page_json = page_dir / "page_data.json"
    ocr_lines = []
    page_data = {}
    if page_json.exists():
        try:
            with open(page_json, "r", encoding="utf-8") as f:
                page_data = json.load(f)
                ocr_lines = page_data.get("ocr_lines", [])
        except Exception:
            pass
    reg = req.region
    tx, ty, tw, th = int(reg.get("x", 0)), int(reg.get("y", 0)), int(reg.get("w", 0)), int(reg.get("h", 0))
    rl = reg.get("rule_lines", [])
    img_bgr = None
    img_path = page_dir / f"page_{req.page_number:04d}.png"
    if img_path.exists():
        img_data = np.fromfile(str(img_path), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img_bgr is None:
        img_w = page_data.get("image_width", 1200)
        img_h = page_data.get("image_height", 1600)
        img_bgr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    if not rl:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        rl = detect_table_rule_lines(binary_img, tx, ty, tw, th)

    t_res = extract_table_cells_and_matrix(img_bgr, tx, ty, tw, th, existing_rule_lines=rl, doc_type=req.doc_type, ocr_lines=ocr_lines)
    table_entry = {
        "id": reg.get("id", "tbl"),
        "name": reg.get("name", "表"),
        "page": req.page_number,
        "x": tx, "y": ty, "w": tw, "h": th,
        "rule_lines": t_res["rule_lines"],
        "columns": t_res["columns"],
        "row_count": t_res["row_count"],
        "rows": t_res["rows"],
        "cells": t_res["cells"],
        "merge_spans": t_res.get("merge_spans", []),
        "text": t_res["text"]
    }
    return {"success": True, "table": table_entry}

class RecognizeSingleRegionRequest(BaseModel):
    filename: str
    page_number: int
    region: Dict[str, Any]
    doc_type: str = "japanese"
    default_orientation: str = "auto"

@app.post("/api/ocr/recognize-single-region")
def recognize_single_region_endpoint(req: RecognizeSingleRegionRequest):
    """
    指定された単一領域のみを高精度に再認識・整形する。
    領域固有の orientation (vertical / horizontal / auto) を尊重し、他の領域やタブデータには一切影響を与えない。
    """
    pdf_stem = Path(req.filename).stem
    page_dir = OCR_RESULTS_DIR / pdf_stem / f"page_{req.page_number:04d}"
    
    reg = req.region
    rx = int(reg.get("x", 0))
    ry = int(reg.get("y", 0))
    rw = int(reg.get("w", 0))
    rh = int(reg.get("h", 0))
    rtype = reg.get("type", "body")
    
    # 領域固有の組方向（未指定や auto ならデフォルト方向）
    reg_orient = reg.get("orientation")
    if not reg_orient or reg_orient == "auto":
        reg_orient = req.default_orientation
    if reg_orient not in ("vertical", "horizontal"):
        reg_orient = "vertical" if req.doc_type == "japanese" else "horizontal"
    
    is_vert = (reg_orient == "vertical")
    
    # ページ画像の取得
    img_bgr = None
    img_path = page_dir / f"page_{req.page_number:04d}.png"
    if img_path.exists():
        try:
            img_data = np.fromfile(str(img_path), dtype=np.uint8)
            img_bgr = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception:
            pass
        
    if img_bgr is None:
        # PDF原本からのレンダリング取得
        candidate_dirs = [
            Path(r"C:\Users\natur\Downloads"),
            BASE_DIR / "data",
            BASE_DIR / "data" / "ocr_results" / pdf_stem
        ]
        for c_dir in candidate_dirs:
            raw_pdf = c_dir / req.filename
            if not raw_pdf.exists() and not req.filename.endswith(".pdf"):
                raw_pdf = c_dir / f"{req.filename}.pdf"
            if raw_pdf.exists():
                try:
                    doc = fitz.open(raw_pdf)
                    pno = max(0, min(req.page_number - 1, len(doc) - 1))
                    page = doc[pno]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
                    nparr = np.frombuffer(pix.tobytes("png"), np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    doc.close()
                    break
                except Exception:
                    pass

    if img_bgr is not None and rw > 6 and rh > 6:
        h_img, w_img = img_bgr.shape[:2]
        c_x1, c_y1 = max(0, rx), max(0, ry)
        c_x2, c_y2 = min(w_img, rx + rw), min(h_img, ry + rh)
        crop = img_bgr[c_y1:c_y2, c_x1:c_x2]
        
        if crop.shape[0] > 6 and crop.shape[1] > 6:
            # クロップ画像に対して指定の組方向でOCRを実行
            crop_lines = run_ndlocr_on_image(crop, orientation=reg_orient, doc_type=req.doc_type)
            
            # 行座標をページ全体座標系にシフト
            shifted_lines = []
            for cl in crop_lines:
                shifted_lines.append({
                    "x": c_x1 + cl.get("x", 0),
                    "y": c_y1 + cl.get("y", 0),
                    "w": cl.get("w", 0),
                    "h": cl.get("h", 0),
                    "text": cl.get("text", ""),
                    "is_vertical": cl.get("is_vertical", is_vert)
                })
                
            if is_vert:
                shifted_lines.sort(key=lambda item: (-item["x"], item["y"]))
            else:
                shifted_lines.sort(key=lambda item: (item["y"], item["x"]))
                
            formatted_text = format_lines_into_paragraphs(shifted_lines, is_vertical=is_vert, doc_type=req.doc_type)
            
            return {
                "success": True,
                "status": "success",
                "region_id": reg.get("id"),
                "text": formatted_text,
                "ocr_lines": shifted_lines,
                "orientation": reg_orient
            }

    return {
        "success": False,
        "status": "error",
        "message": "画像が見つからないか領域サイズが無効です",
        "region_id": reg.get("id"),
        "text": reg.get("text", ""),
        "orientation": reg_orient
    }

class SavePageDataRequest(BaseModel):
    filename: str
    page_number: int
    data: Dict[str, Any]

class BatchSaveRequest(BaseModel):
    filename: str
    pages: List[Dict[str, Any]]

@app.post("/api/batch/save-page-data")
def save_page_data_api(req: SavePageDataRequest):
    success = save_ocr_page_to_disk(req.filename, req.page_number, req.data)
    return {"status": "success" if success else "error", "page_number": req.page_number}

@app.post("/api/batch/save-batch")
def save_batch_api(req: BatchSaveRequest):
    saved_count = 0
    for p in req.pages:
        p_num = p.get("page_number") or p.get("page") or p.get("current_page")
        if p_num and save_ocr_page_to_disk(req.filename, int(p_num), p):
            saved_count += 1
    return {"status": "success", "saved_count": saved_count}

@app.get("/api/batch/load-all-pages")
def load_all_pages_api(filename: str):
    pdf_name = Path(filename).stem
    doc_dir = OCR_RESULTS_DIR / pdf_name
    pages = {}
    if doc_dir.exists():
        for p_dir in sorted(doc_dir.glob("page_*")):
            json_file = p_dir / "page_data.json"
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        p_data = json.load(f)
                        p_num = p_data.get("page_number") or p_data.get("current_page")
                        if not p_num:
                            m = re.search(r"page_(\d+)", p_dir.name)
                            if m: p_num = int(m.group(1))
                        if p_num:
                            p_data["last_modified"] = json_file.stat().st_mtime
                            pages[str(p_num)] = p_data
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    return {"filename": filename, "pages": pages}

class DeleteDiskDataRequest(BaseModel):
    filename: Optional[str] = None

@app.post("/api/batch/delete-disk-data")
async def delete_disk_data_api(
    filename: Optional[str] = Form(None),
    body: Optional[DeleteDiskDataRequest] = None
):
    target_filename = filename or (body.filename if body else None)
    if not target_filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    pdf_name = Path(target_filename).stem
    doc_dir = OCR_RESULTS_DIR / pdf_name
    
    deleted = False
    deleted_pages_count = 0
    if doc_dir.exists() and doc_dir.is_dir():
        try:
            if doc_dir.resolve().is_relative_to(OCR_RESULTS_DIR.resolve()):
                pages = list(doc_dir.glob("page_*"))
                deleted_pages_count = len(pages)
                shutil.rmtree(doc_dir)
                deleted = True
        except Exception as e:
            print(f"Error deleting doc_dir {doc_dir}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    # デスクトップ版OCR結果ディレクトリも存在すれば削除
    desk_doc_dir = Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\ocr_results") / pdf_name
    if desk_doc_dir.exists() and desk_doc_dir.is_dir():
        try:
            shutil.rmtree(desk_doc_dir)
        except Exception:
            pass

    return {
        "status": "success",
        "deleted": deleted,
        "filename": target_filename,
        "deleted_pages_count": deleted_pages_count,
        "message": f"Successfully deleted OCR raw data for {pdf_name}"
    }

class ExportBatchDocxRequest(BaseModel):
    filename: str
    merge_cross_page: Optional[bool] = True
    insert_page_break: Optional[bool] = False
    line_char_count: Optional[int] = 0
    memory_pages: Optional[List[Dict[str, Any]]] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    target_page: Optional[int] = None
    pages: Optional[List[int]] = None

@app.post("/api/batch/export-docx")
def export_batch_docx_post(req: ExportBatchDocxRequest):
    """OCRデータ（単一ページ・選択バッチ・全ページ）からWord (.docx) を安全に出力（READ-ONLY）"""
    clean_stem = Path(req.filename).stem

    # 出力ファイル名サフィックス判定
    if req.target_page is not None:
        file_suffix = f"_p{req.target_page}"
    elif req.page_start is not None and req.page_end is not None:
        if req.page_start == req.page_end:
            file_suffix = f"_p{req.page_start}"
        else:
            file_suffix = f"_p{req.page_start}-{req.page_end}"
    elif req.pages and len(req.pages) == 1:
        file_suffix = f"_p{req.pages[0]}"
    elif req.pages and len(req.pages) > 1:
        file_suffix = f"_p{min(req.pages)}-{max(req.pages)}"
    else:
        file_suffix = ""

    out_filename = f"{clean_stem}{file_suffix}"
    safe_title = urllib.parse.quote(re.sub(r'[\/\\:\*\?"<>\|]', '_', out_filename))
    options = {
        "merge_cross_page": req.merge_cross_page,
        "insert_page_break": req.insert_page_break,
        "line_char_count": req.line_char_count,
        "page_start": req.page_start,
        "page_end": req.page_end,
        "target_page": req.target_page,
        "pages": req.pages
    }
    try:
        docx_stream = export_service.export_ocr_batch_docx(
            pdf_stem=req.filename,
            options=options,
            memory_pages=req.memory_pages
        )
        return Response(
            content=docx_stream.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_title}.docx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word export failed: {str(e)}")

@app.get("/api/batch/export-docx")
def export_batch_docx_get(
    filename: str,
    merge_cross_page: bool = True,
    insert_page_break: bool = False,
    line_char_count: int = 0,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    target_page: Optional[int] = None
):
    """GETリクエストによる直接Word (.docx) ダウンロード（READ-ONLY）"""
    clean_stem = Path(filename).stem
    if target_page is not None:
        file_suffix = f"_p{target_page}"
    elif page_start is not None and page_end is not None:
        if page_start == page_end:
            file_suffix = f"_p{page_start}"
        else:
            file_suffix = f"_p{page_start}-{page_end}"
    else:
        file_suffix = ""

    out_filename = f"{clean_stem}{file_suffix}"
    safe_title = urllib.parse.quote(re.sub(r'[\/\\:\*\?"<>\|]', '_', out_filename))
    options = {
        "merge_cross_page": merge_cross_page,
        "insert_page_break": insert_page_break,
        "line_char_count": line_char_count,
        "page_start": page_start,
        "page_end": page_end,
        "target_page": target_page
    }
    try:
        docx_stream = export_service.export_ocr_batch_docx(
            pdf_stem=filename,
            options=options
        )
        return Response(
            content=docx_stream.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{safe_title}.docx"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word export failed: {str(e)}")

class SaveFigurePngRequest(BaseModel):
    filename: str
    page_number: int
    fig_id: str
    fig_name: Optional[str] = "図版"
    image_base64: str
    is_custom: Optional[bool] = True

@app.post("/api/figures/save-png")
def save_figure_png_endpoint(req: SaveFigurePngRequest):
    pdf_stem = Path(req.filename).stem
    doc_dir = OCR_RESULTS_DIR / pdf_stem
    page_dir = doc_dir / f"page_{req.page_number:04d}"
    figures_dir = doc_dir / "figures"
    page_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        b64 = req.image_base64
        _, encoded = b64.split(",", 1) if "," in b64 else ("", b64)
        img_bytes = base64.b64decode(encoded)
        img_pil = PILImage.open(io.BytesIO(img_bytes))

        if len(img_bytes) > 500 * 1024:
            buf = io.BytesIO()
            img_pil.save(buf, format="JPEG", quality=85)
            img_bytes = buf.getvalue()
            ext = "jpg"
        else:
            ext = "png"

        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', str(req.fig_id))
        fig_name_on_disk = f"figure_{clean_id}.{ext}"
        fig_page_path = page_dir / fig_name_on_disk
        with open(fig_page_path, "wb") as f:
            f.write(img_bytes)

        fig_doc_name = f"p{req.page_number:04d}_{clean_id}.{ext}"
        fig_doc_path = figures_dir / fig_doc_name
        with open(fig_doc_path, "wb") as f:
            f.write(img_bytes)

        page_json = page_dir / "page_data.json"
        if page_json.exists():
            with open(page_json, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            figs = p_data.get("figures", [])
            target_fig = next((f for f in figs if f.get("id") == req.fig_id), None)
            if not target_fig:
                target_fig = {"id": req.fig_id, "name": req.fig_name or "図版", "page": req.page_number}
                figs.append(target_fig)
            target_fig["image_base64"] = req.image_base64
            target_fig["file_name"] = fig_name_on_disk
            target_fig["file_path"] = str(fig_page_path.relative_to(BASE_DIR)).replace("\\", "/")
            target_fig["file_size_kb"] = round(len(img_bytes) / 1024.0, 1)
            target_fig["w"] = img_pil.width
            target_fig["h"] = img_pil.height
            target_fig["is_custom_image"] = req.is_custom
            p_data["figures"] = figs
            with open(page_json, "w", encoding="utf-8") as f:
                json.dump(p_data, f, ensure_ascii=False, indent=2)

        return {"status": "success", "page_number": req.page_number, "fig_id": req.fig_id, "file_name": fig_name_on_disk}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/figures/batch-list")
def get_batch_figures_list(filename: str, start_page: Optional[int] = None, end_page: Optional[int] = None):
    pdf_stem = Path(filename).stem
    doc_dir = OCR_RESULTS_DIR / pdf_stem
    if not doc_dir.exists():
        return {"filename": filename, "figures": []}
    figure_list = []
    for p_dir in sorted(doc_dir.glob("page_*")):
        m = re.search(r"page_(\d+)", p_dir.name)
        if not m: continue
        p_num = int(m.group(1))
        if start_page is not None and p_num < start_page: continue
        if end_page is not None and p_num > end_page: continue
        page_json = p_dir / "page_data.json"
        if not page_json.exists(): continue
        try:
            with open(page_json, "r", encoding="utf-8") as f:
                p_data = json.load(f)
            figs = p_data.get("figures", [])
            for fig in figs:
                b64 = fig.get("image_base64")
                if not b64 and fig.get("file_name"):
                    fpath = p_dir / fig["file_name"]
                    if fpath.exists():
                        with open(fpath, "rb") as im_f:
                            b64_bytes = base64.b64encode(im_f.read()).decode("utf-8")
                            ext = fpath.suffix.lower().replace(".", "")
                            mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
                            b64 = f"data:{mime};base64,{b64_bytes}"
                if b64:
                    figure_list.append({
                        "id": fig.get("id", f"fig_p{p_num}"),
                        "name": fig.get("name", "図版"),
                        "page": p_num,
                        "w": fig.get("w", 0),
                        "h": fig.get("h", 0),
                        "file_name": fig.get("file_name", f"figure_p{p_num}.png"),
                        "file_size_kb": fig.get("file_size_kb", 0),
                        "image_base64": b64,
                        "is_custom_image": fig.get("is_custom_image", False)
                    })
        except Exception as e:
            print(f"Error loading figures from {page_json}: {e}")
    return {"filename": filename, "figures": figure_list}


# ==========================================
# 文書校正 (Proofreading) API エンドポイント (KWJA / ルール / ハイブリッド)
# ==========================================
class ProofreadScanRequest(BaseModel):
    filename: Optional[str] = None
    pages: Dict[str, Dict[str, Any]]
    engine: str = "hybrid"  # rule, kwja, hybrid


class ProofreadApplyRequest(BaseModel):
    filename: Optional[str] = None
    pages: Dict[str, Dict[str, Any]]
    engine: str = "hybrid"


@app.get("/api/proofread/engine-status")
def get_proofread_engine_status():
    """校正エンジンの利用可能状態（ルール、KWJA）を取得"""
    from app.ocr_pipeline.kwja_corrector import is_kwja_available
    kwja_ok = is_kwja_available()
    return {
        "status": "success",
        "rule_available": True,
        "kwja_available": kwja_ok,
        "default_engine": "hybrid" if kwja_ok else "rule"
    }


@app.post("/api/proofread/scan-batch")
def scan_batch_proofread(req: ProofreadScanRequest):
    """指定バッチまたは全ページの校正候補を一括抽出"""
    from app.ocr_pipeline.katakana_corrector import extract_proofread_candidates
    candidates_by_page = {}
    total_candidates = 0

    for p_str, p_data in req.pages.items():
        page_candidates = []
        p_num = int(p_str) if p_str.isdigit() else 1

        # 1. 本文
        body_text = p_data.get("body_text", "")
        if body_text:
            body_cands = extract_proofread_candidates(body_text, engine=req.engine)
            for c in body_cands:
                c["tab"] = "body"
                c["page"] = p_num
                page_candidates.append(c)

        # 2. 見出し
        headings = p_data.get("headings", [])
        for h_idx, h in enumerate(headings):
            h_text = h if isinstance(h, str) else h.get("title", "")
            if h_text:
                h_cands = extract_proofread_candidates(h_text, engine=req.engine)
                for c in h_cands:
                    c["tab"] = "heading"
                    c["page"] = p_num
                    c["item_index"] = h_idx
                    page_candidates.append(c)

        # 3. 注釈文
        footnotes = p_data.get("footnotes", [])
        for f_idx, fn in enumerate(footnotes):
            fn_text = fn if isinstance(fn, str) else fn.get("text", "")
            if fn_text:
                fn_cands = extract_proofread_candidates(fn_text, engine=req.engine)
                for c in fn_cands:
                    c["tab"] = "footnote"
                    c["page"] = p_num
                    c["item_index"] = f_idx
                    page_candidates.append(c)

        candidates_by_page[p_str] = page_candidates
        total_candidates += len(page_candidates)

    return {
        "status": "success",
        "engine": req.engine,
        "total_candidates": total_candidates,
        "candidates_by_page": candidates_by_page
    }


@app.post("/api/proofread/apply-batch")
def apply_batch_proofread(req: ProofreadApplyRequest):
    """指定バッチまたは全ページに対して校正を一括適用"""
    from app.ocr_pipeline.katakana_corrector import correct_japanese_text
    from app.ocr_pipeline.kwja_corrector import is_kwja_available, correct_text_with_kwja

    corrected_pages = {}
    modified_count = 0

    for p_str, p_data in req.pages.items():
        doc = dict(p_data)
        page_mod = False

        # 本文
        orig_body = doc.get("body_text", "")
        if orig_body:
            if "raw_body_text" not in doc or not doc["raw_body_text"]:
                doc["raw_body_text"] = orig_body

            corr_body = correct_japanese_text(orig_body)
            if req.engine in ("kwja", "hybrid") and is_kwja_available():
                corr_body = correct_text_with_kwja(corr_body)

            if corr_body != orig_body:
                doc["body_text"] = corr_body
                page_mod = True

        # 見出し
        headings = doc.get("headings", [])
        new_headings = []
        for h in headings:
            if isinstance(h, str):
                c_h = correct_japanese_text(h)
                new_headings.append(c_h)
                if c_h != h: page_mod = True
            elif isinstance(h, dict):
                nh = dict(h)
                c_t = correct_japanese_text(nh.get("title", ""))
                if c_t != nh.get("title", ""):
                    nh["title"] = c_t
                    page_mod = True
                new_headings.append(nh)
            else:
                new_headings.append(h)
        doc["headings"] = new_headings

        # 注釈文
        footnotes = doc.get("footnotes", [])
        new_footnotes = []
        for fn in footnotes:
            if isinstance(fn, str):
                c_fn = correct_japanese_text(fn)
                new_footnotes.append(c_fn)
                if c_fn != fn: page_mod = True
            elif isinstance(fn, dict):
                nfn = dict(fn)
                c_t = correct_japanese_text(nfn.get("text", ""))
                if c_t != nfn.get("text", ""):
                    nfn["text"] = c_t
                    page_mod = True
                new_footnotes.append(nfn)
            else:
                new_footnotes.append(fn)
        doc["footnotes"] = new_footnotes

        if page_mod:
            modified_count += 1
            doc["is_proofread"] = True
            doc["proofread_engine"] = req.engine

        corrected_pages[p_str] = doc

    return {
        "status": "success",
        "engine": req.engine,
        "modified_count": modified_count,
        "corrected_pages": corrected_pages
    }


# ==========================================
# ユーザー補正辞書 & 縦中横（10〜99）REST API
# ==========================================
USER_DICT_PATHS = [
    BASE_DIR / "data" / "user_dictionary.json",
    Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\config\user_dictionary.json")
]
TCY_REGISTRY_PATHS = [
    BASE_DIR / "data" / "tcy_registry.json",
    Path(r"C:\Users\natur\source\repos\OCR_Translator\ocr_engine\config\tcy_registry.json")
]

@app.get("/api/user-dict")
def get_user_dict():
    for p in USER_DICT_PATHS:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"rules": [], "status": "default"}

@app.post("/api/user-dict")
def save_user_dict(payload: Dict[str, Any]):
    for p in USER_DICT_PATHS:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    try:
        from app.ocr_pipeline.user_dict_service import UserDictService
        UserDictService.get_instance().reload_if_needed(force=True)
    except Exception:
        pass
    return {"status": "success", "count": len(payload.get("rules", []))}

@app.get("/api/tcy-registry")
def get_tcy_registry():
    for p in TCY_REGISTRY_PATHS:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {"tcy": {}, "status": "default"}

@app.post("/api/tcy-registry")
def save_tcy_registry(payload: Dict[str, Any]):
    for p in TCY_REGISTRY_PATHS:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return {"status": "success"}

