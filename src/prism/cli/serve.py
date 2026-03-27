from __future__ import annotations

import typer

from prism.server import ServerConfig, run_server


def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, min=1, max=65535, help="Bind port."),
    browse_page_size: int = typer.Option(25, min=1, help="Gene browser page size."),
    top_gene_limit: int = typer.Option(32, min=1, help="Default search result limit."),
    inference_batch_size: int = typer.Option(
        128, min=1, help="Batch size for checkpoint-backed signal extraction."
    ),
) -> int:
    run_server(
        ServerConfig(
            host=host,
            port=port,
            browse_page_size=browse_page_size,
            top_gene_limit=top_gene_limit,
            inference_batch_size=inference_batch_size,
        )
    )
    return 0
