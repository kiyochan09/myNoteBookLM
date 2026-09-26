from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
import uuid
import hashlib
import base64

class BaseImporter(ABC):
    """
    MyNotebookLM インポーター共通基底クラス
    """
    def __init__(self, media_dir: Optional[Path] = None):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.media_dir = media_dir or (base_dir / "data" / "media")
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def parse_file(self, file_path: Path, original_filename: Optional[str] = None) -> Dict[str, Any]:
        """
        ファイルを解析し、DBService.save_document_bundle() に直接渡せる
        ドキュメントバンドル（document, sections, blocks, tables, figures, annotations）を返す
        """
        pass

    def save_image_bytes(self, img_bytes: bytes, original_ext: str = ".png", prefix: str = "img") -> Dict[str, Any]:
        """
        SHA-256ハッシュによる重複排除を行いながら画像を data/media/ に保存
        """
        sha256 = hashlib.sha256(img_bytes).hexdigest()
        ext = original_ext if original_ext.startswith(".") else f".{original_ext}"
        ext_lower = ext.lower()
        if ext_lower not in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"]:
            ext = ".png"
            ext_lower = ".png"
        
        file_name = f"{prefix}_{sha256[:16]}{ext}"
        out_path = self.media_dir / file_name
        if not out_path.exists():
            with open(out_path, "wb") as f:
                f.write(img_bytes)

        mime = "image/png"
        if ext_lower in [".jpg", ".jpeg"]:
            mime = "image/jpeg"
        elif ext_lower == ".gif":
            mime = "image/gif"
        elif ext_lower == ".webp":
            mime = "image/webp"
        elif ext_lower == ".svg":
            mime = "image/svg+xml"

        b64_str = f"data:{mime};base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        return {
            "file_name": file_name,
            "file_path": f"/media/{file_name}",
            "disk_path": str(out_path),
            "file_hash": sha256,
            "file_size_kb": round(len(img_bytes) / 1024.0, 1),
            "image_base64": b64_str
        }
