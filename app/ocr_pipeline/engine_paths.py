"""Use the single, application-owned OCR source and runtime."""

from pathlib import Path
from app.runtime_paths import ocr_engine_dirs, ocr_python_candidates, python_available, venv_python


from app.paths import APP_PATHS

PROJECT_ROOT = APP_PATHS.root
OCR_ENGINE_DIR = ocr_engine_dirs(PROJECT_ROOT)[0]
OCR_SOURCE_DIR = OCR_ENGINE_DIR / 'src'
OCR_ENGINE_PYTHON = venv_python(OCR_ENGINE_DIR / "venv")


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_ndlocr_command() -> list[str] | None:
    """Return the bundled OCR command, without a fallback to an old installation."""
    if not (OCR_SOURCE_DIR / 'ocr.py').is_file():
        return None
    for python in ocr_python_candidates(PROJECT_ROOT):
        # Run from the app root so the standalone source tree's ocr.py cannot shadow
        # the package installed in this local virtual environment.
        if python_available(python, ('ocr', 'deim', 'parseq')):
            return [str(python), "-m", "ocr"]
    return None
