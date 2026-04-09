from __future__ import annotations

import typer

def serve_command(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Bind port."),
) -> None:
    from prism.server import ServerConfig, run_server

    run_server(ServerConfig(host=host, port=port))


__all__ = ["serve_command"]
