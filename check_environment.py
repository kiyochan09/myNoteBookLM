"""Read-only migration diagnostics; never prints credentials or document text."""
import importlib.util
import json
from app.runtime_paths import PROJECT_ROOT, windows_font_dirs, windows_powershell, get_research_python, python_available
from app.ocr_pipeline.engine_paths import get_ndlocr_command
from app.research_embeddings import REVISION


def inspect_environment():
    modules=['fastapi','uvicorn','pydantic','fitz','cv2','PIL','numpy','docx','lxml','httpx','python_multipart']
    missing=[name for name in modules if importlib.util.find_spec(name) is None]
    research=get_research_python()
    model=PROJECT_ROOT/'models/research'/('multilingual-e5-small-'+REVISION[:12])
    command=get_ndlocr_command()
    return {'app_folder':str(PROJECT_ROOT),'missing_app_modules':missing,
            'ocr_runtime_available':command is not None,'ocr_python':command[0] if command else None,
            'research_python':str(research),'research_runtime_available':python_available(research,('numpy','tokenizers','onnxruntime')),
            'research_model_available':(model/'manifest.json').is_file(),
            'windows_powershell_available':windows_powershell() is not None,
            'font_folders':[str(p) for p in windows_font_dirs() if p.is_dir()],
            'data_folder':str(PROJECT_ROOT/'data')}


if __name__=='__main__':
    print(json.dumps(inspect_environment(),ensure_ascii=False,indent=2))
