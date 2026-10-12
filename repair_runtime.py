"""Repair copied Python environments without removing their packages or models."""
import argparse
from pathlib import Path
import re
import sys
import venv
from app.runtime_paths import PROJECT_ROOT, ocr_engine_dirs, python_available, venv_python


def ensure_venv(folder):
    folder = Path(folder).resolve()
    python = venv_python(folder)
    if python_available(python):
        return python
    config = folder / 'pyvenv.cfg'
    if config.is_file():
        match = re.search(r'^version\s*=\s*(\d+)\.(\d+)', config.read_text(encoding='utf-8'), re.M)
        if match and tuple(map(int, match.groups())) != sys.version_info[:2]:
            version = '.'.join(match.groups())
            raise RuntimeError(f'既存の実行環境はPython {version}用です。Python {version}でこの修復処理を実行してください。')
    venv.EnvBuilder(with_pip=True, upgrade=config.is_file()).create(folder)
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
        folder = PROJECT_ROOT / '.venv_research'
    elif args.ocr:
        choices = [engine / name for engine in ocr_engine_dirs() for name in ('venv','.venv')]
        folder = next((p for p in choices if (p / 'pyvenv.cfg').is_file()), None)
        if folder is None:
            raise RuntimeError('OCR実行環境が見つかりません。移行元のOCR-REPOSフォルダもコピーしてください。')
    else:
        folder = args.folder
    print('実行環境を確認しました:', ensure_venv(folder))


if __name__ == '__main__':
    try:main()
    except (RuntimeError,OSError) as error:
        print(str(error),file=sys.stderr);sys.exit(1)
