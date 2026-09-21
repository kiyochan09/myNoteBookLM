import sys
import os
import socket
from pathlib import Path

# プロジェクトルートを sys.path の先頭に追加
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn

def is_port_in_use(port: int, host: str = '127.0.0.1') -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0

def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    for port in range(start_port, start_port + max_attempts):
        if not is_port_in_use(port):
            return port
    return start_port

if __name__ == "__main__":
    default_port = int(os.environ.get("PORT", 8000))
    port = default_port
    if is_port_in_use(port):
        alt_port = find_available_port(port + 1)
        print(f"[Notice] Port {port} is in use. Using port {alt_port} instead.")
        port = alt_port

    print(f"Starting server at http://127.0.0.1:{port} (app_dir: {BASE_DIR})")
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=port,
        reload=True,
        app_dir=str(BASE_DIR)
    )
