# Restore only after the previous process has exited, before any DB is opened.
from pathlib import Path as _StartupPath
from app.data_backup import apply_pending_restore
apply_pending_restore(_StartupPath(__file__).resolve().parent.parent)

from app.ocr_pipeline.line_tilt import decode_page_image, correct_image_request
import io
from contextlib import asynccontextmanager
import logging
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
from app.ocr_pdf_archive import OcrPdfArchive, create_archive_router, serialized
from app.pdf_compression import PdfCompression, create_compression_router
import cv2
import numpy as np
from PIL import Image as PILImage

# ndlocr_core パス
NDLOCR_CORE_DIR = Path(__file__).resolve().parent / "ocr_pipeline" / "ndlocr_core"
if str(NDLOCR_CORE_DIR) not in sys.path:
    sys.path.insert(0, str(NDLOCR_CORE_DIR))

from app.core.db import get_db_connection, DEFAULT_DB_PATH
from app.db_service import DBService
from app.research_catalog import get_research_catalog
from app.research_threads import create_research_router
from app.research_search import create_search_router
from app.research_embeddings import OfflineEmbeddingProvider
from app.research_index import SemanticIndex
from app.research_answers import ResearchAnswers, create_answers_router
from app.research_ai_settings import AISettings
from app.importers.docx_importer import DocxImporter
from app.importers.odt_importer import OdtImporter
from app.importers.pdf_importer import PdfImporter
from app.importers.ocr_batch_importer import OcrBatchImporter
from app.export_service import ExportService

from app.ocr_pipeline.layout_engine import (
    recognize_custom_regions,
    extract_table_cells_and_matrix,
    detect_table_rule_lines,
    format_lines_into_paragraphs
)
from app.ocr_pipeline.ndlocr_engine import run_ndlocr_on_image
from app.ocr_pipeline.region_classifier import split_vertical_heading_regions

from app.paths import APP_PATHS

BASE_DIR = APP_PATHS.root
DATA_DIR = APP_PATHS.data
MEDIA_DIR = APP_PATHS.media
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_UPLOAD_DIR = APP_PATHS.uploads
TEMP_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PDF_CACHE_DIR = APP_PATHS.pdf_cache
PDF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# アプリケーション共通セキュリティトークン (暗号学的一意性 & リロード耐久性)
TOKEN_FILE = APP_PATHS.token
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

@asynccontextmanager
async def research_lifespan(app):
    try:
        yield
    finally:
        await research_answers.close()
        research_index.close()


app = FastAPI(
    title="MyNotebookLM & OCR Knowledge Base System",
    description="MyNotebookLM 統合ドキュメントエディタ & OCR-REPOS システム",
    version="2.0.0",
    lifespan=research_lifespan
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
app.include_router(create_research_router(APP_PATHS.research_db, db_service))
research_index = SemanticIndex(APP_PATHS.research_index_db, db_service, OfflineEmbeddingProvider(BASE_DIR))
app.include_router(create_search_router(APP_PATHS.research_db, db_service, semantic_index=research_index))
research_answers = ResearchAnswers(APP_PATHS.research_db, db_service, AISettings(APP_PATHS.ai_settings))
app.include_router(create_answers_router(research_answers, APP_SECURITY_TOKEN))

from app.data_backup import DataBackup, MaintenanceGate, BackupGateMiddleware, create_backup_router
backup_gate = MaintenanceGate()
def backup_background_busy():
    with research_index.lock:
        indexing = research_index.active_job is not None or bool(research_index.pending)
    return indexing or any(not task.done() for task in research_answers.tasks.values()) or ("pdf_compression" in globals() and pdf_compression.is_busy())
data_backup = DataBackup(BASE_DIR, backup_gate, backup_background_busy)
app.include_router(create_backup_router(data_backup, APP_SECURITY_TOKEN))
app.add_middleware(BackupGateMiddleware, gate=backup_gate)

export_service = ExportService(db_service)

# ==========================================
# 画面配信 (HTML Routes) とセキュリティトークン注入
# ==========================================
EDITOR_HTML_PATH = BASE_DIR / "editor_form.html"
RESEARCH_HTML_PATH = BASE_DIR / "research_form.html"
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

@app.get("/research_form.html", response_class=HTMLResponse)
@app.get("/research", response_class=HTMLResponse)
def get_research_form():
    if not RESEARCH_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="Research screen not found")
    return HTMLResponse(content=_inject_security_token(RESEARCH_HTML_PATH.read_text(encoding="utf-8")))


@app.get("/api/research/catalog")
def research_catalog_endpoint():
    return {"status": "success", **get_research_catalog(db_service)}


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


class BatchMultiTagAssignRequest(BaseModel):
    doc_ids: List[str]
    tag_ids: List[str]


@app.post("/api/documents/tags/batch-assign-many")
def batch_assign_multiple_tags_endpoint(req: BatchMultiTagAssignRequest):
    try:
        count = db_service.assign_multiple_tags_to_documents(req.doc_ids, req.tag_ids)
        return {"status": "success", "assigned_count": count}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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
        logging.getLogger(__name__).exception("Editor document save failed: %s", doc_id)
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

        import hashlib
        metadata = bundle["document"].get("doc_metadata_json") or {}
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        metadata["import_sha256"] = hashlib.sha256(content).hexdigest()
        bundle["document"]["doc_metadata_json"] = metadata
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


class ImportPreflightFile(BaseModel):
    name: str
    sha256: str = ""


class ImportPreflightRequest(BaseModel):
    files: List[ImportPreflightFile]


@app.post("/api/editor/import-preflight")
def preflight_editor_import(request: ImportPreflightRequest):
    from app.import_duplicates import find_duplicates
    conn = db_service.get_connection()
    try:
        documents = [dict(r) for r in conn.execute(
            "SELECT id,title,source_filename,doc_metadata_json FROM documents")]
    finally:
        conn.close()
    files = [{"name": f.name, "sha256": f.sha256} for f in request.files]
    return {"status": "success", "duplicates": find_duplicates(files, documents)}


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
    content = await _read_file_safely(file)
    from app.editor_image import prepare_editor_image
    try:
        content, ext, width, height = await run_in_threadpool(prepare_editor_image, content)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    importer = DocxImporter(media_dir=MEDIA_DIR)
    saved = importer.save_image_bytes(content, original_ext=ext, prefix="user_upload")
    return {
        "status": "success",
        "file_name": saved["file_name"],
        "file_path": saved["file_path"],
        "image_base64": saved.get("image_base64", ""),
        "file_size_kb": saved["file_size_kb"],
        "file_size_bytes": len(content),
        "file_hash": saved["file_hash"],
        "width": width, "height": height
    }


@app.get("/api/search")
def search_global_documents(q: str = "", group: str = "all"):
    results = db_service.search_global(q, group)
    return {"status": "success", "query": q, "count": len(results), "results": results}


@app.get("/api/editor/search")
def search_editor_blocks(q: str):
    """FTS5 全文検索"""
    results = db_service.search_blocks(q)
    return {"query": q, "count": len(results), "results": results}


# ==========================================
# 既存 OCR API 群 (完全互換保持)
# ==========================================
OCR_RESULTS_DIR = APP_PATHS.ocr_results
OCR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ocr_pdf_archive = OcrPdfArchive(OCR_RESULTS_DIR, APP_PATHS.ocr_archives, PDF_CACHE_DIR)
app.include_router(create_archive_router(ocr_pdf_archive))
pdf_compression = PdfCompression(ocr_pdf_archive, backup_gate)
app.include_router(create_compression_router(pdf_compression))

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

@serialized
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
        body_regions = [r for r in regions if r.get("type") in ["body", "paragraph", None]]
        figure_only_page = any(r.get("type") == "image" for r in regions) and not body_regions and not any(r.get("type") in ["heading", "footnote", "table"] for r in regions)
        if figure_only_page:
            # OCR fragments from map labels/captions must not be imported as article body text.
            body_text = ""
            save_dict["body_text"] = ""
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

        if body_text or figure_only_page:
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
        APP_PATHS.ocr_results / clean_stem / f"{clean_stem}.pdf",
        APP_PATHS.raw_pdfs / filename,
        APP_PATHS.raw_pdfs / f"{clean_stem}.pdf"
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

@serialized
def _recognize_regions_core(content: bytes, filename: str, page_number: int, regions: list, orientation: str, doc_type: str, severe_skew: bool = False, corrected_image: Optional[str] = None) -> dict:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDFファイルの展開に失敗しました: {str(e)}")
    try:
        updated_data = recognize_custom_regions(doc=doc, page_number=page_number, regions=regions, orientation=orientation, doc_type=doc_type, filename=filename, severe_skew=severe_skew, image_override=decode_page_image(corrected_image) if corrected_image else None)
        if corrected_image:
            updated_data["corrected_image"] = corrected_image
        save_ocr_page_to_disk(filename, page_number, updated_data)
        return updated_data
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        doc.close()

class LineTiltRequest(BaseModel):
    image_base64: str
    mode: str = "manual"
    boundaries: List[Dict[str, float]] = []


@app.post("/api/ocr/correct-line-tilt")
async def correct_line_tilt_endpoint(req: LineTiltRequest):
    try:
        return await run_in_threadpool(correct_image_request, req.image_base64, req.boundaries, req.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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

@app.post("/api/ocr/recognize-regions")
async def recognize_regions_endpoint(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(None),
    page_number: int = Form(1),
    regions_json: str = Form(...),
    orientation: str = Form("auto"),
    doc_type: str = Form("japanese"),
    severe_skew: bool = Form(False),
    corrected_image: Optional[str] = Form(None),
    corrected_image_file: Optional[UploadFile] = File(None)
):
    # A PNG file avoids multipart text-field size limits for full-page scans.
    if corrected_image_file is not None:
        corrected_bytes = await _read_file_safely(corrected_image_file)
        if not corrected_bytes:
            raise HTTPException(status_code=400, detail="補正画像のファイルが空です。")
        corrected_image = "data:image/png;base64," + base64.b64encode(corrected_bytes).decode("ascii")
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
    if not isinstance(regions, list) or not regions:
        raise HTTPException(status_code=400, detail="OCR対象の領域がありません。先に領域自動判定を実行してください。")

    updated_data = await run_in_threadpool(_recognize_regions_core, content, target_filename, page_number, regions, orientation, doc_type, severe_skew, corrected_image)
    return updated_data

@serialized
def _detect_regions_only_core(content: bytes, page_number: int, filename: str, line_boxes: Optional[List[Dict[str, Any]]] = None, orientation: str = "auto", deck_count: int = 0, processed_image: Optional[str] = None, corrected_image: Optional[str] = None) -> dict:
    """Detect coarse text/table regions from page pixels without invoking OCR."""
    try:
        doc = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"PDFファイルを開けません: {exc}")
    try:
        if len(doc) == 0:
            raise HTTPException(status_code=400, detail="PDF内にページがありません。")
        pno = max(0, min(int(page_number) - 1, len(doc) - 1))
        page = doc[pno]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        image = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY) if pix.n == 3 else cv2.cvtColor(pixels, cv2.COLOR_RGBA2GRAY)
        if corrected_image:
            try:
                image = cv2.cvtColor(decode_page_image(corrected_image, image.shape), cv2.COLOR_BGR2GRAY)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        if processed_image:
            try:
                encoded = processed_image.split(",", 1)[-1]
                cleaned_bytes = base64.b64decode(encoded, validate=True)
                erase_mask = cv2.imdecode(np.frombuffer(cleaned_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if erase_mask is not None:
                    if erase_mask.ndim == 3 and erase_mask.shape[2] == 4:
                        opaque = erase_mask[:, :, 3] > 0
                    elif erase_mask.ndim == 3:
                        opaque = cv2.cvtColor(erase_mask, cv2.COLOR_BGR2GRAY) < 250
                    else:
                        opaque = erase_mask < 250
                    if opaque.shape != image.shape:
                        opaque = cv2.resize(
                            opaque.astype(np.uint8), (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST
                        ) > 0
                    image[opaque] = 255
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail="画像処理済みデータを読み取れませんでした。画像処理をリセットして再実行してください。")
        height, width = image.shape[:2]
        # In automatic mode, treat a landscape scan as a two-page spread.
        # Explicit one- or three-column settings remain authoritative.
        two_page_mode = deck_count == 2 or (deck_count == 0 and width >= height * 1.05)
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        def find_center_gutter(mask):
            """Choose the least-ink X near the fold, tolerant of vertical noise."""
            # OCR-REPOS uses the summed darkness profile and smooths it before
            # selecting a split. Unlike a pure-white-gap threshold, this still
            # works when the scanned fold contains a dark vertical streak.
            gray = image.astype(np.float32)
            ink_density = np.maximum(0.0, 240.0 - gray).sum(axis=0)
            smooth_w = max(5, min(25, (width // 40) | 1))
            smooth = cv2.GaussianBlur(ink_density.reshape(1, -1), (smooth_w, 1), 5.0).ravel()
            search_left, search_right = int(width * 0.44), int(width * 0.56)
            center_band = smooth[search_left:search_right]
            if center_band.size == 0:
                split_x = width // 2
            else:
                split_x = search_left + int(np.argmin(center_band))
                # Keep the chosen minimum close to the expected center; wider
                # tolerance can snap to an interior page margin and merge one
                # scanned page into the other.
                if abs(split_x - width // 2) > int(width * 0.025):
                    split_x = width // 2
            half_gap = max(4, int(width * 0.006))
            return max(0, split_x - half_gap), min(width, split_x + half_gap)

        center_gutter = find_center_gutter(binary) if two_page_mode else None
        # The option is explicit: if a clean white gutter cannot be measured,
        # still split the two-page scan around the page center.
        split_gutter = center_gutter or (
            (max(0, width // 2 - max(4, int(width * 0.006))),
             min(width, width // 2 + max(4, int(width * 0.006))))
            if two_page_mode else None
        )

        # Long horizontal and vertical rules provide table boxes without text recognition.
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, width // 25), 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, height // 30)))
        h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
        v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
        rule_mask = cv2.add(h_lines, v_lines)
        contours = cv2.findContours(rule_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        table_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < width * 0.12 or h < height * 0.08 or w * h < width * height * 0.003:
                continue
            # Scan/page borders can create a huge rectangle that looks like a
            # table grid. Do not mask most of a page as a table when the box
            # touches the image edge; that can erase one half of a spread.
            touches_image_edge = (
                x <= width * 0.015 or y <= height * 0.015
                or x + w >= width * 0.985 or y + h >= height * 0.985
            )
            if touches_image_edge and (w >= width * 0.45 or h >= height * 0.75):
                continue
            # A rectangular border around prose (or a scan seam) is not a
            # table. Require a real grid with at least three distinct tracks
            # in each direction: outer borders plus internal separators.
            hr = np.count_nonzero(h_lines[y:y + h, x:x + w], axis=1) > 0
            vr = np.count_nonzero(v_lines[y:y + h, x:x + w], axis=0) > 0
            horizontal_tracks = int(np.count_nonzero(np.diff(np.pad(hr.astype(np.uint8), (1, 1))) == 1))
            vertical_tracks = int(np.count_nonzero(np.diff(np.pad(vr.astype(np.uint8), (1, 1))) == 1))
            if horizontal_tracks >= 3 and vertical_tracks >= 3:
                table_boxes.append((x, y, w, h))

        regions = []
        for idx, (x, y, w, h) in enumerate(sorted(table_boxes, key=lambda b: (b[1], b[0])), 1):
            regions.append({
                "id": f"detect-table-{idx}", "name": f"表候補 {idx}", "type": "table",
                "x": x, "y": y, "w": w, "h": h, "text": "", "reading_order": idx
            })

        # When OCR has already populated line coordinates, reuse the same geometry
        # cues as OCR layout (valid line bounds, table exclusion, and spread split).
        # Only boxes are accepted here; recognized strings are neither sent nor used.
        valid_lines = []
        geometry_exclusion_boxes = []
        geometry_lines = []
        for item in line_boxes or []:
            try:
                x, y, w, h = [int(item[key]) for key in ("x", "y", "w", "h")]
            except (KeyError, TypeError, ValueError):
                continue
            if w > 8 and h > 8 and x >= 0 and y >= 0 and x + w <= width and y + h <= height:
                text = str(item.get("text") or "").strip()
                geometry_lines.append({
                    "x": x, "y": y, "w": w, "h": h,
                    "is_vertical": bool(item.get("is_vertical")), "text": text
                })

        for line in geometry_lines:
            x, y, w, h = (line[k] for k in ("x", "y", "w", "h"))
            is_wide_horizontal_block = (
                orientation != "horizontal" and w >= width * 0.06 and w >= h * 2.0
            )
            is_very_small_block = w <= width * 0.08 and h <= height * 0.04
            if is_wide_horizontal_block:
                geometry_exclusion_boxes.append((x, y, w, h))
                continue
            if is_very_small_block:
                # A short fragment at the end of a vertical text column is
                # still body text. Exclude tiny blocks only when no nearby
                # block continues on the same x track.
                has_neighbor = False
                for other in geometry_lines:
                    if other is line:
                        continue
                    overlap_x = min(x + w, other["x"] + other["w"]) - max(x, other["x"])
                    if overlap_x < min(w, other["w"]) * 0.4:
                        continue
                    if other["y"] + other["h"] <= y:
                        gap = y - (other["y"] + other["h"])
                    elif y + h <= other["y"]:
                        gap = other["y"] - (y + h)
                    else:
                        gap = 0
                    if gap <= max(12, int(h * 1.6)):
                        has_neighbor = True
                        break
                if not has_neighbor:
                    geometry_exclusion_boxes.append((x, y, w, h))
                    continue
            valid_lines.append(line)

        # Cached OCR lines can be partial or stale. For a requested spread,
        # always use pixel geometry for explicit two-page scans. This keeps
        # page bounds independent of which page happened to have cached OCR.
        if two_page_mode:
            valid_lines = []

        if valid_lines:
            def outside_table(line):
                cx, cy = line["x"] + line["w"] / 2, line["y"] + line["h"] / 2
                return not any(tx - 8 <= cx <= tx + tw + 8 and ty - 8 <= cy <= ty + th + 8 for tx, ty, tw, th in table_boxes)

            non_table_lines = [line for line in valid_lines if outside_table(line)] or valid_lines
            min_x = min(line["x"] for line in valid_lines)
            max_x = max(line["x"] + line["w"] for line in valid_lines)
            min_y = min(line["y"] for line in valid_lines)
            max_y = max(line["y"] + line["h"] for line in valid_lines)
            total_w, total_h = max_x - min_x, max_y - min_y
            is_vertical = orientation == "vertical" or (
                orientation not in ("vertical", "horizontal")
                and sum(1 for line in valid_lines if line["is_vertical"]) > len(valid_lines) * 0.4
            )
            groups = []
            if split_gutter:
                gutter_left, gutter_right = split_gutter
                left = [line for line in valid_lines if line["x"] + line["w"] / 2 < gutter_left]
                right = [line for line in valid_lines if line["x"] + line["w"] / 2 > gutter_right]
                left_text = [line for line in non_table_lines if line["x"] + line["w"] / 2 < gutter_left]
                right_text = [line for line in non_table_lines if line["x"] + line["w"] / 2 > gutter_right]
                if left and right:
                    ordered = [("右ページ本文", right_text or right), ("左ページ本文", left_text or left)] if is_vertical else [("左ページ本文", left_text or left), ("右ページ本文", right_text or right)]
                    groups = [(name, lines) for name, lines in ordered if lines]
            if not groups:
                groups = [("本文候補", non_table_lines)]

            for idx, (name, lines) in enumerate(groups, len(regions) + 1):
                x0 = max(0, min(line["x"] for line in lines) - 8)
                y0 = max(0, min(line["y"] for line in lines) - 8)
                x1 = min(width, max(line["x"] + line["w"] for line in lines) + 8)
                y1 = min(height, max(line["y"] + line["h"] for line in lines) + 8)
                if x1 > x0 and y1 > y0:
                    regions.append({
                        "id": f"detect-body-{idx}", "name": name, "type": "body",
                        "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                        "text": "", "reading_order": len(regions) + 1
                    })
            heading_mask = binary.copy()
            for x, y, w, h in table_boxes + geometry_exclusion_boxes:
                heading_mask[y:y + h, x:x + w] = 0
            regions = split_vertical_heading_regions(heading_mask, regions, orientation)
            return {
                "success": True, "filename": filename, "current_page": pno + 1,
                "total_pages": len(doc), "image_width": width, "image_height": height,
                "regions": regions, "source": "cached_ocr_geometry",
                "heading_count": sum(r.get("type") == "heading" for r in regions),
                "requested_deck_count": deck_count,
                "spread_split": bool(split_gutter and len(groups) == 2),
                "gutter_source": "smoothed_ink_minimum" if center_gutter else ("center_fallback" if two_page_mode else "not_requested")
            }

        # Follow OCR layout's valid-line and table-exclusion rules using pixel
        # geometry as a proxy: ignore tiny noise, page-edge specks, oversized
        # illustration components, and anything inside a detected table.
        if two_page_mode and split_gutter:
            # Threshold each scanned page independently. Global Otsu can be
            # pulled by the dark center seam or a photo on the opposite page.
            body_mask = np.zeros_like(binary)
            gutter_left, gutter_right = split_gutter
            page_slices = [(0, gutter_left), (gutter_right, width)]
            for x0, x1 in page_slices:
                if x1 <= x0:
                    continue
                page_gray = image[:, x0:x1]
                page_binary = cv2.threshold(
                    page_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )[1]
                body_mask[:, x0:x1] = page_binary
        else:
            body_mask = binary.copy()
        for x, y, w, h in table_boxes:
            body_mask[y:y + h, x:x + w] = 0
        # Apply the same shape-only exclusions to image components when OCR
        # line geometry is available. Pixel-only glyphs are kept intact here;
        # classifying each glyph as a tiny block would erase ordinary text.
        for x, y, w, h in geometry_exclusion_boxes:
            body_mask[y:y + h, x:x + w] = 0
        count, _, stats, _ = cv2.connectedComponentsWithStats(body_mask, connectivity=8)
        boxes = []
        pixel_exclusion_count = 0
        max_glyph_dim = max(48, int(min(width, height) * 0.08))
        edge_x = int(width * 0.015)
        for idx in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if (
                area >= 12 and w >= 8 and h >= 8
                and w <= max_glyph_dim and h <= max_glyph_dim
                and x >= edge_x and x + w <= width - edge_x
            ):
                boxes.append((x, y, w, h))

        # OCR-free equivalent of the block-shape filters above. Join glyph
        # components into same-height rows, then exclude wide horizontal text
        # blocks and tiny isolated blocks by their geometry, never by page y.
        spread_pages_split = False
        if boxes:
            median_glyph_h = float(np.median([b[3] for b in boxes]))
            row_tolerance = max(3, int(median_glyph_h * 0.45))
            component_rows = []
            if two_page_mode and split_gutter:
                gutter_left, gutter_right = split_gutter
                page_box_groups = [
                    [b for b in boxes if b[0] + b[2] / 2 < gutter_left],
                    [b for b in boxes if b[0] + b[2] / 2 > gutter_right],
                ]
            else:
                page_box_groups = [boxes]
            for page_boxes in page_box_groups:
                page_rows = []
                for box in sorted(page_boxes, key=lambda b: b[1] + b[3] / 2):
                    cy = box[1] + box[3] / 2
                    if not page_rows:
                        page_rows.append([box])
                        continue
                    last_cy = float(np.median([b[1] + b[3] / 2 for b in page_rows[-1]]))
                    if abs(cy - last_cy) <= row_tolerance:
                        page_rows[-1].append(box)
                    else:
                        page_rows.append([box])
                # Same Y does not imply the same line block: a running
                # chapter name on the left and a page number on the right
                # must not become one page-wide row. Split at actual X gaps.
                for row in page_rows:
                    typical_width = float(np.median([b[2] for b in row]))
                    max_inline_gap = max(8, int(max(typical_width * 1.6, median_glyph_h * 0.9)))
                    segment = []
                    right_edge = None
                    for box in sorted(row, key=lambda b:b[0]):
                        if right_edge is not None and box[0] - right_edge > max_inline_gap:
                            component_rows.append(segment)
                            segment = []
                        segment.append(box)
                        right_edge = max(right_edge or 0, box[0] + box[2])
                    if segment:
                        component_rows.append(segment)

            blocks_to_remove = set()
            for row in component_rows:
                rx0 = min(b[0] for b in row)
                ry0 = min(b[1] for b in row)
                rx1 = max(b[0] + b[2] for b in row)
                ry1 = max(b[1] + b[3] for b in row)
                rw, rh = rx1 - rx0, ry1 - ry0
                wide_horizontal = (
                    orientation != "horizontal"
                    and rw >= width * 0.06 and rw <= width * (0.40 if two_page_mode else 0.70)
                    and rw >= rh * 2.0
                )
                very_small = rw <= width * 0.08 and rh <= height * 0.04 and len(row) <= 12
                if not (wide_horizontal or very_small):
                    continue

                # A short final row in a vertical text column can look tiny.
                # Keep shape candidates that continue vertically at the same
                # x; isolated horizontal headings/page-number blocks do not.
                has_vertical_neighbor = []
                for x, y, w, h in row:
                    sx0 = max(0, x + int(w * 0.25))
                    sx1 = min(width, max(sx0 + 1, x + int(w * 0.75)))
                    gap = max(1, int(h * 0.10))
                    reach = max(4, int(h * 1.6))
                    above_y0 = max(0, y - gap - reach)
                    above_y1 = max(0, y - gap)
                    below_y0 = min(height, y + h + gap)
                    below_y1 = min(height, below_y0 + reach)
                    above = body_mask[above_y0:above_y1, sx0:sx1]
                    below = body_mask[below_y0:below_y1, sx0:sx1]
                    has_vertical_neighbor.append(
                        bool(np.count_nonzero(above) or np.count_nonzero(below))
                    )
                isolated_ratio = 1.0 - sum(has_vertical_neighbor) / max(1, len(row))
                if isolated_ratio >= 0.70:
                    blocks_to_remove.update(row)

            if blocks_to_remove:
                pixel_exclusion_count = len(blocks_to_remove)
                boxes = [b for b in boxes if b not in blocks_to_remove]

        if boxes:
            # Build each page's body area from repeated glyph Y coordinates.
            # A scanned spread can contain an upper and a lower text block on
            # each page, so page-wide min/max bounds incorrectly bridge the
            # whitespace between those blocks (and can absorb a running head).
            page_box_groups = [("本文候補", boxes)]
            if split_gutter:
                gutter_left, gutter_right = split_gutter
                left_boxes = [b for b in boxes if b[0] + b[2] / 2 < gutter_left]
                right_boxes = [b for b in boxes if b[0] + b[2] / 2 > gutter_right]
                if left_boxes and right_boxes:
                    spread_pages_split = True
                    if orientation == "vertical":
                        page_box_groups = [("右ページ", right_boxes), ("左ページ", left_boxes)]
                    else:
                        page_box_groups = [("左ページ", left_boxes), ("右ページ", right_boxes)]

            box_groups = []
            typical_glyph_h = max(1, int(np.median([b[3] for b in boxes])))
            for page_name, page_boxes in page_box_groups:
                # Also look for a substantial X-coordinate valley inside each
                # page. A display title at the outer edge can otherwise stretch
                # the body rectangle and skew the spread split. Keep the title
                # as its own OCR candidate instead of merging it with body text.
                spatial_groups = [(page_name, page_boxes)]
                if two_page_mode and len(page_boxes) >= 12:
                    page_x0 = min(b[0] for b in page_boxes)
                    page_x1 = max(b[0] + b[2] for b in page_boxes)
                    page_span = page_x1 - page_x0
                    if page_span > 0:
                        x_counts = np.zeros(width, dtype=np.int32)
                        for x, _, w, _ in page_boxes:
                            x_counts[x:min(width, x + w)] += 1
                        positive = x_counts[page_x0:page_x1]
                        positive = positive[positive > 0]
                        typical_x_occupancy = float(np.median(positive)) if positive.size else 0.0
                        x_active = x_counts >= max(1.0, typical_x_occupancy * 0.10)
                        min_x_gap = max(
                            int(np.median([b[2] for b in page_boxes]) * 1.3),
                            int(width * 0.01), 8
                        )
                        min_cluster = max(8, int(len(page_boxes) * 0.08))
                        candidate_gaps = []
                        x = page_x0
                        while x < page_x1:
                            if x_active[x]:
                                x += 1
                                continue
                            gap_start = x
                            while x < page_x1 and not x_active[x]:
                                x += 1
                            if x - gap_start < min_x_gap:
                                continue
                            left_cluster = [b for b in page_boxes if b[0] + b[2] / 2 < gap_start]
                            right_cluster = [b for b in page_boxes if b[0] + b[2] / 2 >= x]
                            if min(len(left_cluster), len(right_cluster)) >= min_cluster:
                                candidate_gaps.append((x - gap_start, gap_start, x, left_cluster, right_cluster))
                        if candidate_gaps:
                            _, _, _, left_cluster, right_cluster = max(candidate_gaps, key=lambda item: item[0])
                            if len(left_cluster) <= len(right_cluster):
                                spatial_groups = [(f"{page_name}見出し候補", left_cluster), (f"{page_name}本文", right_cluster)]
                            else:
                                spatial_groups = [(f"{page_name}本文", left_cluster), (f"{page_name}見出し候補", right_cluster)]

                if orientation == "vertical":
                    # A large gutter can split prose before heading refinement.
                    # Keep the preceding right-hand columns before the heading.
                    spatial_groups.sort(key=lambda group: max(b[0] + b[2] for b in group[1]), reverse=True)

                for spatial_name, spatial_boxes in spatial_groups:
                    # Count boxes covering each Y coordinate. Repeated
                    # alignment marks each upper/lower text band independently.
                    occupancy = np.zeros(height, dtype=np.int32)
                    for _, y, _, h in spatial_boxes:
                        occupancy[max(0, y):min(height, y + h)] += 1
                    smooth_window = max(3, typical_glyph_h)
                    kernel = np.ones(smooth_window, dtype=np.float32) / smooth_window
                    smoothed = np.convolve(occupancy, kernel, mode="same")
                    active = smoothed >= max(1.0, len(spatial_boxes) * 0.0015)
                    active_y = np.flatnonzero(active)
                    y_ranges = []
                    if len(active_y):
                        run_start = previous = int(active_y[0])
                        max_gap = max(8, int(typical_glyph_h * 2.5))
                        for current in map(int, active_y[1:]):
                            if current - previous > max_gap:
                                y_ranges.append((run_start, previous + 1))
                                run_start = current
                            previous = current
                        y_ranges.append((run_start, previous + 1))

                    # Retain meaningful bands only; this avoids splitting at
                    # ordinary line gaps while keeping upper/lower blocks apart.
                    bands = []
                    for y0, y1 in y_ranges:
                        band_boxes = [b for b in spatial_boxes if b[1] + b[3] > y0 and b[1] < y1]
                        # A few disconnected scan-seam specks must not extend
                        # the body to a page edge. Demand repeated glyph support
                        # in each band, relative to this text group (bounded so
                        # genuine short text bands still survive).
                        min_band_support = max(3, min(12, int(len(spatial_boxes) * 0.01)))
                        if len(band_boxes) >= min_band_support:
                            bands.append((y0, y1, band_boxes))
                    if deck_count == 1 and bands:
                        # A fixed single-deck setting must not be overridden by
                        # automatically detected upper/lower ink bands. Keep
                        # substantial separated prose in the same body region.
                        primary = max(bands, key=lambda band: len(band[2]))
                        primary_boxes = primary[2]
                        primary_w = max(b[0] + b[2] for b in primary_boxes) - min(b[0] for b in primary_boxes)
                        body_boxes = []
                        for band in bands:
                            band_boxes = band[2]
                            band_w = max(b[0] + b[2] for b in band_boxes) - min(b[0] for b in band_boxes)
                            band_h = max(b[1] + b[3] for b in band_boxes) - min(b[1] for b in band_boxes)
                            # An isolated small horizontal running head/folio
                            # can contain several rows, so its own rows must
                            # not count as vertical prose continuation. Use
                            # its shape/support, never a fixed page margin.
                            small_horizontal_block = (
                                orientation != "horizontal" and band is not primary
                                and band_w >= band_h * 2.0
                                and band_h <= typical_glyph_h * 3
                                and band_w <= primary_w * 0.5
                                and len(band_boxes) <= len(primary_boxes) * 0.10
                            )
                            if small_horizontal_block:
                                pixel_exclusion_count += len(band_boxes)
                                continue
                            body_boxes.extend(band_boxes)
                        box_groups.append((spatial_name, body_boxes))
                    elif len(bands) >= 2:
                        for band_index, (_, _, band_boxes) in enumerate(bands, 1):
                            suffix = '上段' if band_index == 1 else '下段' if band_index == 2 else f'{band_index}段'
                            box_groups.append((f"{spatial_name}{suffix}", band_boxes))
                    elif bands:
                        # Use the one valid band, not all original components:
                        # the latter reintroduced the discarded edge noise.
                        box_groups.append((spatial_name, bands[0][2]))

            for name, group_boxes in box_groups:
                x0 = max(0, min(b[0] for b in group_boxes) - 8)
                y0 = max(0, min(b[1] for b in group_boxes) - 8)
                x1 = min(width, max(b[0] + b[2] for b in group_boxes) + 8)
                y1 = min(height, max(b[1] + b[3] for b in group_boxes) + 8)
                if x1 > x0 and y1 > y0:
                    regions.append({
                        "id": f"detect-body-{len(regions) + 1}", "name": name, "type": "body",
                        "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
                        "text": "", "reading_order": len(regions) + 1
                    })

        regions = split_vertical_heading_regions(body_mask, regions, orientation)
        return {
            "success": True, "filename": filename, "current_page": pno + 1,
            "total_pages": len(doc), "image_width": width, "image_height": height,
            "regions": regions, "requested_deck_count": deck_count,
            "heading_count": sum(r.get("type") == "heading" for r in regions),
            "two_page_mode": two_page_mode,
            "spread_split": spread_pages_split,
            "segmentation_source": "two_page_pixel_geometry" if two_page_mode else "pixel_geometry",
            "excluded_block_count": len(geometry_exclusion_boxes) + pixel_exclusion_count,
            "gutter_source": "smoothed_ink_minimum" if center_gutter else ("center_fallback" if two_page_mode else "not_requested"),
            "gutter_x": int((split_gutter[0] + split_gutter[1]) / 2) if split_gutter else None,
            "image_processing_applied": bool(processed_image or corrected_image),
            "corrected_image": corrected_image
        }
    finally:
        doc.close()

@app.post("/api/ocr/detect-regions-only")
async def detect_regions_only_endpoint(
    file: Optional[UploadFile] = File(None),
    filename: Optional[str] = Form(None),
    page_number: int = Form(1),
    line_boxes_json: Optional[str] = Form(None),
    orientation: str = Form("auto"),
    deck_count: int = Form(0),
    processed_image: Optional[str] = Form(None),
    corrected_image: Optional[str] = Form(None),
    corrected_image_file: Optional[UploadFile] = File(None)
):
    # A PNG file avoids multipart text-field size limits for full-page scans.
    if corrected_image_file is not None:
        corrected_bytes = await _read_file_safely(corrected_image_file)
        if not corrected_bytes:
            raise HTTPException(status_code=400, detail="補正画像のファイルが空です。")
        corrected_image = "data:image/png;base64," + base64.b64encode(corrected_bytes).decode("ascii")
    target_filename = file.filename if file and file.filename else filename
    if not target_filename or not target_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDFファイルを指定してください (.pdf)")
    if file and file.filename:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="PDFファイルの内容が空です")
        try:
            (PDF_CACHE_DIR / f"{Path(target_filename).stem}.pdf").write_bytes(content)
        except Exception:
            pass
    else:
        content = await run_in_threadpool(get_pdf_bytes_sync, target_filename)
        if not content:
            raise HTTPException(status_code=404, detail="PDFがサーバーのキャッシュにありません。PDFを開き直してください。")
    line_boxes = []
    if line_boxes_json:
        try:
            line_boxes = json.loads(line_boxes_json)
        except (TypeError, ValueError):
            line_boxes = []
    return await run_in_threadpool(_detect_regions_only_core, content, page_number, target_filename, line_boxes, orientation, deck_count, processed_image, corrected_image)

class ExtractTableGridRequest(BaseModel):
    filename: str
    page_number: int
    region: Dict[str, Any]
    doc_type: str = "japanese"
    orientation: str = "auto"
    corrected_image: Optional[str] = None
    processed_image: Optional[str] = None

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
    from app.ocr_pipeline.table_image_grid import render_table_image
    img_bgr = None
    highres = None
    corrected = req.corrected_image if req.corrected_image is not None else page_data.get("corrected_image")
    try:
        if corrected:
            img_bgr = decode_page_image(corrected)
        else:
            img_path = page_dir / f"page_{req.page_number:04d}.png"
            if img_path.exists():
                img_bgr = cv2.imdecode(np.fromfile(str(img_path), np.uint8), cv2.IMREAD_COLOR)
        pdf_candidates = [PDF_CACHE_DIR / f"{pdf_stem}.pdf", APP_PATHS.raw_pdfs / f"{pdf_stem}.pdf", DATA_DIR / f"{pdf_stem}.pdf"]
        pdf_path = next((p for p in pdf_candidates if p.is_file()), None)
        if pdf_path:
            with fitz.open(pdf_path) as doc:
                if not 1 <= req.page_number <= len(doc):
                    raise ValueError("PDFのページ番号を確認してください。")
                page = doc[req.page_number-1]
                if img_bgr is None:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False, colorspace=fitz.csRGB)
                    img_bgr = cv2.cvtColor(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3), cv2.COLOR_RGB2BGR)
                size = (page_data.get("image_width") or img_bgr.shape[1], page_data.get("image_height") or img_bgr.shape[0])
                if corrected and img_bgr.shape[:2] != (size[1], size[0]):
                    raise ValueError("補正画像の大きさが元のページと一致しません。")
                highres = render_table_image(page, reg, img_bgr if corrected else None, size)
        elif img_bgr is not None:
            crop = img_bgr[ty:ty+th, tx:tx+tw]
            if crop.size:
                highres = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        if img_bgr is None or highres is None:
            raise ValueError("表を認識するページ画像が見つかりません。PDFを開き直してください。")
        if req.processed_image:
            encoded = req.processed_image.split(",", 1)[-1]
            mask = cv2.imdecode(np.frombuffer(base64.b64decode(encoded, validate=True), np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None or mask.ndim != 3 or mask.shape[2] != 4 or mask.shape[:2] != img_bgr.shape[:2]:
                raise ValueError("画像処理のマスクを読み取れませんでした。")
            opaque = mask[:, :, 3] > 0
            img_bgr[opaque] = 255
            local_mask = opaque[ty:ty+th, tx:tx+tw].astype(np.uint8)
            highres[cv2.resize(local_mask, (highres.shape[1], highres.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 255
        if not rl:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            rl = detect_table_rule_lines(binary_img, tx, ty, tw, th)
        t_res = extract_table_cells_and_matrix(img_bgr, tx, ty, tw, th, existing_rule_lines=rl,
            doc_type=req.doc_type, ocr_lines=None, recognition_image=highres)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        "merges": t_res.get("merges", []),
        "text": t_res["text"]
    }
    return {"success": True, "table": table_entry}

class RecognizeSingleRegionRequest(BaseModel):
    filename: str
    page_number: int
    region: Dict[str, Any]
    doc_type: str = "japanese"
    default_orientation: str = "auto"
    severe_skew: bool = False

@app.post("/api/ocr/recognize-single-region")
@serialized
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
            APP_PATHS.raw_pdfs,
            DATA_DIR,
            APP_PATHS.ocr_results / pdf_stem
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
            crop_lines = run_ndlocr_on_image(crop, orientation=reg_orient, doc_type=req.doc_type,
                recover_small_text=req.severe_skew, recover_missing_columns=req.severe_skew)
            
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

@app.get("/ocr_footnotes.js")
def serve_ocr_footnotes_script():
    return FileResponse(BASE_DIR / "ocr_footnotes.js", media_type="application/javascript")

@app.get("/ocr_pdf_compression.js")
def serve_ocr_pdf_compression_script():
    return FileResponse(BASE_DIR / "ocr_pdf_compression.js", media_type="application/javascript")


@app.get("/ocr_pdf_library.js")
def serve_ocr_pdf_library_script():
    return FileResponse(BASE_DIR / "ocr_pdf_library.js", media_type="application/javascript")

@app.get("/backup_restore.js")
def serve_backup_restore_script():
    return FileResponse(BASE_DIR / "backup_restore.js", media_type="application/javascript")

@app.get("/storage_folders.js")
def serve_storage_folders_script():
    return FileResponse(BASE_DIR / "storage_folders.js", media_type="application/javascript")

@app.get("/api/ocr/footnote-catalog")
def ocr_footnote_catalog(filename: str):
    from app.ocr_footnotes import build_catalog
    return {"filename":filename,"pages":build_catalog(OCR_RESULTS_DIR / Path(filename).stem)}

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
                        # Recover figure image references from page-local PNGs when
                        # older saves kept the crop but lost its metadata.
                        page_figures = p_data.get("figures", [])
                        if isinstance(page_figures, list):
                            for fig_idx, fig in enumerate(page_figures):
                                if not isinstance(fig, dict) or fig.get("image_base64"):
                                    continue
                                file_name = fig.get("file_name") or f"figure_{fig_idx:02d}.png"
                                fig_path = p_dir / file_name
                                if not fig_path.exists() and not fig.get("file_name"):
                                    jpg_path = p_dir / f"figure_{fig_idx:02d}.jpg"
                                    if jpg_path.exists():
                                        fig_path = jpg_path
                                        file_name = jpg_path.name
                                if fig_path.exists():
                                    mime = "image/jpeg" if fig_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
                                    fig["image_base64"] = f"data:{mime};base64,{base64.b64encode(fig_path.read_bytes()).decode('ascii')}"
                                    fig["file_name"] = file_name
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
    archive_id: Optional[str] = None

@app.post("/api/batch/delete-disk-data")
async def delete_disk_data_api(
    filename: Optional[str] = Form(None),
    archive_id: Optional[str] = Form(None),
    body: Optional[DeleteDiskDataRequest] = None
):
    target = filename or (body.filename if body else None)
    archive = archive_id or (body.archive_id if body else None)
    if not target or not archive:
        raise HTTPException(status_code=400, detail="先にOCR付きPDFを保存・検証してください。生データは保持します。")
    try:
        return await run_in_threadpool(ocr_pdf_archive.delete_raw, target, archive)
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
# ユーザー補正辞書 & 縦中横（10〜99）REST API
# ==========================================
USER_DICT_PATHS = [
    APP_PATHS.user_dictionary
]
TCY_REGISTRY_PATHS = [
    APP_PATHS.tcy_registry
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



class ParagraphStyleRequest(BaseModel):
    style: Dict[str, Any]
    previous: Optional[Dict[str, Any]] = None
    scope: str = "database"
    document_id: Optional[str] = None

@app.get("/api/editor/paragraph-styles")
def get_paragraph_styles(document_id: Optional[str] = None):
    from app.editor_paragraph_styles import load_styles
    return {"styles": load_styles(db_service, document_id)}

@app.post("/api/editor/paragraph-styles")
def save_paragraph_style(req: ParagraphStyleRequest):
    from app.editor_paragraph_styles import update_style
    try:
        return update_style(db_service, req.style, req.previous, req.scope, req.document_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
