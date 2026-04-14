"""
server.py — tiny stdlib HTTP server that serves the D3 visualizer.

The graph JSON is injected directly into the HTML at startup — no API
endpoints, no static files on disk, no extra dependencies.
"""
from __future__ import annotations

import http.server
import importlib.resources
import json
import random
import socket
import threading
import webbrowser


def launch(
    graph_data: dict,
    port: int | None = None,
    initial_search: str | None = None,
) -> None:
    """Serve the visualizer and open the browser. Blocks until Ctrl+C."""
    port = port or _free_port()
    html = _build_html(graph_data, initial_search=initial_search or "")

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # silence request logs
            pass

    server = http.server.HTTPServer(("localhost", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"  pocket-mem visualizer -> {url}")
    print("  Press Ctrl+C to close")

    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
        print("\n  Visualizer closed.")


def _free_port() -> int:
    """Find an available port in the 8700–8799 range."""
    candidates = list(range(8700, 8800))
    random.shuffle(candidates)
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    # Fall back to OS-assigned port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _build_html(graph_data: dict, initial_search: str = "") -> str:
    """Load index.html template and inject the graph data."""
    try:
        # Python 3.9+ path
        ref = importlib.resources.files("memory_agent.visualizer.ui").joinpath("index.html")
        template = ref.read_text(encoding="utf-8")
    except Exception:
        # Fallback: read relative to this file
        import pathlib
        tpl = pathlib.Path(__file__).parent / "ui" / "index.html"
        template = tpl.read_text(encoding="utf-8")

    html = template.replace("__GRAPH_DATA__", json.dumps(graph_data))
    html = html.replace("__INITIAL_SEARCH__", json.dumps(initial_search))
    return html
