"""Repair copied Python environments without removing their packages or models."""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import venv
from app.paths import APP_PATHS
from app.runtime_paths import PROJECT_ROOT, ocr_engine_dirs, python_available, venv_python


def ensure_venv(folder):
    folder = Path(folder).resolve()
    python = venv_python(folder)
    config = folder / 'pyvenv.cfg'
    if config.is_file():
        match = re.search(r'^version\s*=\s*(\d+)\.(\d+)', config.read_text(encoding='utf-8'), re.M)
        if match and tuple(map(int, match.groups())) != sys.version_info[:2]:
            version = '.'.join(match.groups())
            raise RuntimeError(f'既存の実行環境はPython {version}用です。Python {version}でこの修復処理を実行してください。')
    venv.EnvBuilder(with_pip=not python_available(python)).create(folder)
    result = subprocess.run([str(python), '-B', str(PROJECT_ROOT / 'app/venv_launchers.py'), str(folder)], cwd=PROJECT_ROOT)
    if result.returncode:
        raise RuntimeError('実行環境のランチャーを修復できませんでした。')
    # Do not reuse a negative probe cached before the repaired launcher existed.
    from app.runtime_paths import _probe_python
    _probe_python.cache_clear()
    if not python_available(python):
        raise RuntimeError('実行環境を修復できませんでした。Pythonのインストール状態を確認してください。')
    return python


def main():
    parser = argparse.ArgumentParser(description='コピーした実行環境のPython参照先を、このPCのPythonへ修復します。既存のパッケージ・モデルは削除しません。')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--research', action='store_true')
    group.add_argument('--ocr', action='store_true')
    group.add_argument('--folder', type=Path, help='修復する仮想環境フォルダ')
    args = parser.parse_args()
    if args.research:
        folder = APP_PATHS.research_runtime
    elif args.ocr:
        choices = [engine / name for engine in ocr_engine_dirs() for name in ('venv','.venv')]
        folder = next((p for p in choices if (p / 'pyvenv.cfg').is_file()), None)
        if folder is None:
            raise RuntimeError('本体内のruntimes/ocr/venvが見つかりません。アプリフォルダーを実行環境ごとコピーしてください。')
    else:
        folder = args.folder
    if args.ocr:
        config = folder / 'pyvenv.cfg'
        match = re.search(r'^version\s*=\s*(\d+)\.(\d+)', config.read_text(encoding='utf-8'), re.M)
        if match and tuple(map(int, match.groups())) != sys.version_info[:2]:
            raise RuntimeError('OCR実行環境と同じPython ' + '.'.join(match.groups()) + 'で修復してください。')
        # Recreate interpreter configuration/activation scripts in place, keeping
        # the installed dependency files and the separate managed OCR source.
        venv.EnvBuilder(with_pip=False).create(folder)
        python = venv_python(folder)
        helper = PROJECT_ROOT / 'runtimes/ocr/repair_launchers.py'
        result = subprocess.run([str(python), '-B', str(helper)], cwd=PROJECT_ROOT)
        if result.returncode:
            raise RuntimeError('OCRランチャーの修復に失敗しました。')
        from app.runtime_paths import _probe_python
        _probe_python.cache_clear()
        if not python_available(python, ('ocr', 'deim', 'parseq')):
            raise RuntimeError('修復後のOCR実行環境を確認できません。')
        print('OCR実行環境を修復しました:', python)
    else:
        print('実行環境を確認しました:', ensure_venv(folder))


if __name__ == '__main__':
    try:main()
    except (RuntimeError,OSError) as error:
        print(str(error),file=sys.stderr);sys.exit(1)
