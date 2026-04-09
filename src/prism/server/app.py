from __future__ import annotations

import uvicorn

from .api import create_api_app
from .config import ServerConfig
from .state import AppState


class ServerApp:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.state = AppState(config)
        self.app = create_api_app(config, state=self.state)

    def serve_forever(self) -> None:
        print(
            f"[prism-server] serving http://{self.config.host}:{self.config.port}",
            flush=True,
        )
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info",
        )


def run_server(config: ServerConfig) -> None:
    ServerApp(config).serve_forever()
