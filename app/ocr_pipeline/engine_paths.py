"""Paths for the OCR runtime stored inside the Antigravity workspace."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANTIGRAVITY_SCRATCH = PROJECT_ROOT.parent
OCR_ENGINE_DIR = ANTIGRAVITY_SCRATCH / "OCR-REPOS" / "ocr_engine"
OCR_ENGINE_PYTHON = OCR_ENGINE_DIR / "venv" / "Scripts" / "python.exe"


def get_project_root() -> Path:
    return PROJECT_ROOT


def get_ndlocr_command() -> list[str] | None:
    """Return the command for the local NDLOCR package, if it is installed."""
    module_file = OCR_ENGINE_PYTHON.parent.parent / "Lib" / "site-packages" / "ocr.py"
    if OCR_ENGINE_PYTHON.is_file() and module_file.is_file():
        # Run from the app root so the standalone source tree's ocr.py cannot shadow
        # the package installed in this local virtual environment.
        return [str(OCR_ENGINE_PYTHON), "-m", "ocr"]
    return None
