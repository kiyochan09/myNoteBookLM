"""Reconnect the preserved dependency environment to its managed OCR source."""
from pathlib import Path
import importlib.metadata
import json
import re
import sys
import zipfile


def repair():
    runtime = Path(__file__).resolve().parent
    environment = runtime / 'venv'
    if Path(sys.prefix).resolve() != environment.resolve():
        raise RuntimeError('Run this helper with runtimes/ocr/venv/Scripts/python.exe.')
    site = environment / ('Lib/site-packages' if sys.platform == 'win32' else f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
    source = runtime / 'src'
    if not (source / 'ocr.py').is_file():
        raise RuntimeError('Managed OCR source is missing.')
    # Relative to site-packages; no user name, drive, sibling project, or old venv.
    import os
    (site / 'mynotebooklm_ocr.pth').write_text(os.path.relpath(source, site).replace('\\', '/') + '\n', encoding='utf-8')
    from pip._vendor.distlib.scripts import ScriptMaker
    scripts = environment / ('Scripts' if sys.platform == 'win32' else 'bin')
    maker = ScriptMaker(None, str(scripts))
    maker.executable = str(scripts / ('python.exe' if sys.platform == 'win32' else 'python'))
    maker.clobber = True
    maker.variants = {''}
    count = 0
    for distribution in importlib.metadata.distributions():
        for entry in distribution.entry_points:
            if entry.group not in ('console_scripts', 'gui_scripts') or entry.name == 'ndlocr-lite':
                continue
            # Rebuild existing commands only; do not introduce optional applications.
            existing = scripts / (entry.name + ('.exe' if sys.platform == 'win32' else ''))
            if existing.is_file():
                maker.make(entry.name + ' = ' + entry.value, options={'gui': entry.group == 'gui_scripts'})
                count += 1
    # The locally modified engine has an __main__ CLI, not the upstream ocr:main.
    maker._write_script({'ndlocr-lite'}, maker._get_shebang('utf-8'),
                        b"import runpy\nrunpy.run_module('ocr', run_name='__main__')\n", [], 'py')
    # pip's versioned aliases and non-entry-point utilities can survive a move.
    # Reuse their existing script bodies with the current interpreter launcher.
    for path in scripts.iterdir():
        if not path.is_file() or path.name in ('python.exe', 'pythonw.exe'):
            continue
        raw = path.read_bytes()
        shebang = re.search(br'#![^\r\n]*python(?:w)?(?:\.exe)?[^\r\n]*\r?\n', raw)
        if not shebang or str(environment).encode('utf-8') in shebang.group():
            continue
        if path.suffix.lower() == '.exe':
            try:
                with zipfile.ZipFile(path) as packed:
                    body = packed.read('__main__.py')
                maker._write_script({path.stem}, maker._get_shebang('utf-8'), body, [], 'py')
            except (zipfile.BadZipFile, KeyError):
                raise RuntimeError('Cannot repair old launcher: ' + path.name)
        elif raw.startswith(b'#!'):
            path.write_bytes(maker._get_shebang('utf-8') + raw[shebang.end():])
    print(json.dumps({'launchers_rebuilt': count + 1, 'ocr_source': str(source)}, ensure_ascii=False))


if __name__ == '__main__':
    repair()
