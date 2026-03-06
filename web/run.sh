#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-8080}"
echo "=== VesselBoost Development Server ==="
echo "Serving at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
cd "$SCRIPT_DIR"

# Serve with COOP/COEP headers for SharedArrayBuffer (multi-threaded WASM)
python3 -c "
import http.server, functools

class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        super().end_headers()

http.server.HTTPServer(('', $PORT), CORSHandler).serve_forever()
"
