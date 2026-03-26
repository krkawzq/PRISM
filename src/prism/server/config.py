from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    analysis_chunk_size: int = 2048
    top_gene_limit: int = 32
    gene_browser_page_size: int = 25
    plot_max_points: int = 2500
    pool_r: float = 0.05
    show_pool_fit_progress: bool = True

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.analysis_chunk_size < 1:
            raise ValueError(
                f"analysis_chunk_size must be >= 1, got {self.analysis_chunk_size}"
            )
        if self.top_gene_limit < 1:
            raise ValueError(f"top_gene_limit must be >= 1, got {self.top_gene_limit}")
        if self.gene_browser_page_size < 1:
            raise ValueError(
                "gene_browser_page_size must be >= 1, got "
                f"{self.gene_browser_page_size}"
            )
        if self.plot_max_points < 1:
            raise ValueError(
                f"plot_max_points must be >= 1, got {self.plot_max_points}"
            )
        if self.pool_r <= 0:
            raise ValueError(f"pool_r must be > 0, got {self.pool_r}")
