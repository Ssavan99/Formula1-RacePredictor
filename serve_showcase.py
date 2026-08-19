"""Preview server for the scroll-showcase experiment, on its own port.

Kept separate from serve_docs.py so the main site stays untouched while this
alternative is judged side by side.
"""
import functools, http.server, os, socketserver, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "showcase"
Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({".js": "application/javascript", ".json": "application/json",
                               ".css": "text/css", ".svg": "image/svg+xml"})
port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 4000))
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", port), functools.partial(Handler, directory=str(ROOT))) as httpd:
    print(f"showcase on http://localhost:{port}", flush=True)
    httpd.serve_forever()
