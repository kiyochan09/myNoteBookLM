import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path

def run_ocr_on_image(image_path: str, lang: str = "ja") -> list:
    ps_file = Path(__file__).parent / "run_win_ocr.ps1"
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out_json = tmp.name
    
    try:
        cmd = [
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", str(ps_file),
            "-ImgPath", os.path.abspath(image_path),
            "-OutJsonPath", os.path.abspath(out_json),
            "-Lang", lang
        ]
        res = subprocess.run(cmd, capture_output=True, timeout=30)
        if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
            try:
                with open(out_json, "r", encoding="utf-8-sig") as f:
                    content = f.read().strip()
                    if not content:
                        return []
                    data = json.loads(content)
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return [data]
            except Exception:
                return []
        return []
    except Exception as e:
        print(f"[OCR ERROR] {e}", file=sys.stderr)
        return []
    finally:
        if os.path.exists(out_json):
            try:
                os.remove(out_json)
            except Exception:
                pass
