from __future__ import annotations

from pathlib import Path

from fastapi.responses import FileResponse, HTMLResponse, Response


def serve_frontend_asset(path: str = "") -> Response:
    dist_dir = frontend_dist_dir()
    if not dist_dir.is_dir():
        return HTMLResponse(_missing_build_html(), status_code=503)
    safe_path = path.strip("/")
    if safe_path:
        candidate = (dist_dir / safe_path).resolve()
        try:
            candidate.relative_to(dist_dir.resolve())
        except ValueError:
            return HTMLResponse("Not Found", status_code=404)
        if candidate.is_file():
            return FileResponse(candidate)
    index_path = dist_dir / "index.html"
    if not index_path.is_file():
        return HTMLResponse(_missing_build_html(), status_code=503)
    return FileResponse(index_path)


def frontend_dist_dir() -> Path:
    return Path(__file__).resolve().parent / "frontend" / "dist"


def _missing_build_html() -> str:
    return """
    <!doctype html>
    <html lang=\"en\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>PRISM Server Frontend Missing</title>
        <style>
          body { font-family: sans-serif; padding: 24px; max-width: 720px; margin: 0 auto; }
          code, pre { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }
        </style>
      </head>
      <body>
        <h1>Frontend build not found</h1>
        <p>The decoupled frontend has not been built yet.</p>
        <p>Run the following inside <code>src/prism/server/frontend</code>:</p>
        <pre>npm install\nnpm run build</pre>
      </body>
    </html>
    """


__all__ = ["frontend_dist_dir", "serve_frontend_asset"]
