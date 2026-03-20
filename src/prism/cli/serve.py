from __future__ import annotations

import typer

from prism.server import ServerConfig, run_server


def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, min=1, max=65535, help="Bind port."),
) -> int:
    run_server(ServerConfig(host=host, port=port))
    return 0
