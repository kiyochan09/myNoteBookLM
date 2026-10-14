"""Repair existing console launchers after moving a preserved environment."""
from pathlib import Path
import importlib.metadata
import re
import sys
import zipfile

def repair(environment):
    environment = Path(environment).resolve()
    if Path(sys.prefix).resolve() != environment:
        raise RuntimeError('Run with the interpreter of the environment to repair.')
    from pip._vendor.distlib.scripts import ScriptMaker
    scripts = environment / ('Scripts' if sys.platform == 'win32' else 'bin')
    maker = ScriptMaker(None, str(scripts))
    maker.executable = str(scripts / ('python.exe' if sys.platform == 'win32' else 'python'))
    maker.clobber = True
    maker.variants = {''}
    count = 0
    for distribution in importlib.metadata.distributions():
        for entry in distribution.entry_points:
            if entry.group not in ('console_scripts', 'gui_scripts'):
                continue
            existing = scripts / (entry.name + ('.exe' if sys.platform == 'win32' else ''))
            if existing.is_file():
                maker.make(entry.name + ' = ' + entry.value, options={'gui':entry.group == 'gui_scripts'})
                count += 1
    for path in scripts.iterdir():
        if not path.is_file() or path.name in ('python.exe', 'pythonw.exe'):
            continue
        raw = path.read_bytes()
        shebang = re.search(br'#![^\r\n]*python(?:w)?(?:\.exe)?[^\r\n]*\r?\n', raw)
        if not shebang or str(environment).encode('utf-8') in shebang.group():
            continue
        if path.suffix.lower() == '.exe':
            with zipfile.ZipFile(path) as packed:
                body = packed.read('__main__.py')
            maker._write_script({path.stem}, maker._get_shebang('utf-8'), body, [], 'py')
        elif raw.startswith(b'#!'):
            path.write_bytes(maker._get_shebang('utf-8') + raw[shebang.end():])
    print('Repaired environment launchers:', count)

if __name__ == '__main__':
    repair(sys.argv[1])
