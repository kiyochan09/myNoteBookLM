import sys
import base64
from pathlib import Path

def write_file(rel_path: str, b64_content: str):
    target = Path(rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(b64_content.strip()).decode('utf-8')
    target.write_text(raw, encoding='utf-8')
    print(f'Wrote {target} ({len(raw)} chars)')

if __name__ == '__main__':
    if len(sys.argv) == 3:
        write_file(sys.argv[1], sys.argv[2])
