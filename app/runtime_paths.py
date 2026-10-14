"""Resolve optional runtimes from the current app location and current PC."""
from functools import lru_cache
import os
from pathlib import Path
import shutil
import subprocess
import sys

from app.paths import APP_PATHS, AppPaths

PROJECT_ROOT = APP_PATHS.root


def configured_path(name, root=PROJECT_ROOT):
    value = os.environ.get(name, '').strip()
    if not value:
        return None
    path = Path(os.path.expandvars(value)).expanduser()
    return path if path.is_absolute() else Path(root) / path


def venv_python(folder):
    return Path(folder) / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


@lru_cache(maxsize=64)
def _probe_python(path, modules, stamp):
    code = ('import importlib.util,sys;'
            'sys.exit(0 if all(importlib.util.find_spec(n) is not None for n in ' + repr(modules) + ') else 1)')
    try:
        result = subprocess.run([path, '-c', code], cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def python_available(path, modules=()):
    """A copied venv launcher may exist even when its old base Python is gone."""
    path = Path(path)
    try:
        stat = path.stat()
        config = path.parent.parent / 'pyvenv.cfg'
        config_stamp = config.stat().st_mtime_ns if config.is_file() else 0
    except OSError:
        return False
    return path.is_file() and _probe_python(str(path), tuple(modules), (stat.st_mtime_ns, config_stamp))


def ocr_engine_dirs(root=PROJECT_ROOT):
    """OCR has one application-owned location; never use a sibling app's venv."""
    return [AppPaths(root).ocr_runtime]


def ocr_python_candidates(root=PROJECT_ROOT):
    return [venv_python(ocr_engine_dirs(root)[0] / 'venv')]


def research_python_candidates(root=PROJECT_ROOT):
    root = Path(root)
    paths = [configured_path('MYNOTEBOOKLM_RESEARCH_PYTHON', root),
             venv_python(AppPaths(root).research_runtime), venv_python(root / 'runtimes/research')]
    return list(dict.fromkeys(p for p in paths if p is not None))


def get_research_python(root=PROJECT_ROOT):
    candidates = research_python_candidates(root)
    return next((p for p in candidates if python_available(p, ('numpy','tokenizers','onnxruntime'))), candidates[0])


def windows_font_dirs():
    windows = os.environ.get('WINDIR') or os.environ.get('SystemRoot')
    local = os.environ.get('LOCALAPPDATA')
    paths = [Path(windows) / 'Fonts'] if windows else []
    if local:
        paths.append(Path(local) / 'Microsoft/Windows/Fonts')
    return list(dict.fromkeys(paths))


def windows_powershell():
    windows = os.environ.get('SystemRoot') or os.environ.get('WINDIR')
    native = Path(windows) / 'System32/WindowsPowerShell/v1.0/powershell.exe' if windows else None
    if native is not None and native.is_file():
        return str(native)
    return shutil.which('powershell.exe') or shutil.which('powershell')
