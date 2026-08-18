"""Local preview server for docs/ with correct JS MIME types.

Python's http.server reads MIME types from the Windows registry, where .js is
often registered as text/plain -- and browsers refuse to load text/plain as an
ES module, so the 3D car silently fails to import. GitHub Pages sends the
correct type, so this only matters for local preview.

Port comes from $PORT (or argv[1]), because nothing here needs a fixed port --
it serves static files, with no OAuth callback, webhook or CORS origin to match.

The served directory resolves relative to THIS FILE, not the working directory,
so it works whatever the launcher's cwd happens to be.
"""
import functools
import http.server
import os
import socketserver
import sys
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "docs"

Handler = http.server.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".json": "application/json",
    ".css": "text/css",
    ".svg": "image/svg+xml",
})

port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", 8765))
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("", port), functools.partial(Handler, directory=str(DOCS))) as httpd:
    print(f"serving {DOCS} on http://localhost:{port}", flush=True)
    httpd.serve_forever()
