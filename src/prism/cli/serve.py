from __future__ import annotations

import typer

from prism.server import ServerConfig, run_server


def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, min=1, max=65535, help="Bind port."),
    pool_r: float = typer.Option(
        0.05, min=1e-12, help="Silent pool-scale r hyperparameter."
    ),
) -> int:
    run_server(ServerConfig(host=host, port=port, pool_r=pool_r))
    return 0
