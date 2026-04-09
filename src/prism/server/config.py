from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    browse_page_size: int = 25
    top_gene_limit: int = 32
    inference_batch_size: int = 128
    posterior_gallery_cells: int = 12
    kbulk_default_max_classes: int = 6

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.browse_page_size < 1:
            raise ValueError("browse_page_size must be >= 1")
        if self.top_gene_limit < 1:
            raise ValueError("top_gene_limit must be >= 1")
        if self.inference_batch_size < 1:
            raise ValueError("inference_batch_size must be >= 1")
        if self.posterior_gallery_cells < 1:
            raise ValueError("posterior_gallery_cells must be >= 1")
        if self.kbulk_default_max_classes < 1:
            raise ValueError("kbulk_default_max_classes must be >= 1")
