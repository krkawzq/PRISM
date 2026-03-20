from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import ServerConfig
from .handlers import build_router
from .router import Request, Response
from .state import AppState


class ServerApp:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.state = AppState(config)
        self.router = build_router(self.state)

    def make_handler_class(self) -> type[BaseHTTPRequestHandler]:
        app = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                response = app.router.dispatch(Request.from_raw_path("GET", self.path))
                self._write_response(response)

            def log_message(self, format: str, *args: object) -> None:
                super().log_message(format, *args)

            def _write_response(self, response: Response) -> None:
                self.send_response(response.status)
                self.send_header("Content-Type", response.content_type)
                self.send_header("Content-Length", str(len(response.body)))
                for key, value in response.headers.items():
                    self.send_header(key, value)
                self.end_headers()
                if response.body:
                    self.wfile.write(response.body)

        return RequestHandler

    def serve_forever(self) -> None:
        handler_class = self.make_handler_class()
        server = ThreadingHTTPServer(
            (self.config.host, self.config.port), handler_class
        )
        try:
            print(
                f"[prism-server] serving http://{self.config.host}:{self.config.port}",
                flush=True,
            )
            server.serve_forever()
        finally:
            server.server_close()


def run_server(config: ServerConfig) -> None:
    ServerApp(config).serve_forever()
