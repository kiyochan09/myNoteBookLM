"""Find a working OCR runtime relative to this app, including after migration."""

from pathlib import Path
from app.runtime_paths import ocr_engine_dirs, ocr_python_candidates, python_available, venv_python


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANTIGRAVITY_SCRATCH = PROJECT_ROOT.parent
OCR_ENGINE_DIR = next((p for p in ocr_engine_dirs(PROJECT_ROOT) if p.is_dir()), ocr_engine_dirs(PROJECT_ROOT)[0])
OCR_ENGINE_PYTHON = venv_python(OCR_ENGINE_DIR / "venv")


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_ndlocr_command() -> list[str] | None:
    """Return the command for the local NDLOCR package, if it is installed."""
    for python in ocr_python_candidates(PROJECT_ROOT):
        # Run from the app root so the standalone source tree's ocr.py cannot shadow
        # the package installed in this local virtual environment.
        if python_available(python, ('ocr', 'deim', 'parseq')):
            return [str(python), "-m", "ocr"]
    return None
