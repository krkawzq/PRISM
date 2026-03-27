from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    browse_page_size: int = 25
    top_gene_limit: int = 32
    plot_max_points: int = 2500
    inference_batch_size: int = 128
    global_eval_max_cells: int = 2000
    global_eval_max_genes: int = 256

    def __post_init__(self) -> None:
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"port must be in [1, 65535], got {self.port}")
        if self.browse_page_size < 1:
            raise ValueError(
                f"browse_page_size must be >= 1, got {self.browse_page_size}"
            )
        if self.top_gene_limit < 1:
            raise ValueError(f"top_gene_limit must be >= 1, got {self.top_gene_limit}")
        if self.plot_max_points < 1:
            raise ValueError(
                f"plot_max_points must be >= 1, got {self.plot_max_points}"
            )
        if self.inference_batch_size < 1:
            raise ValueError(
                f"inference_batch_size must be >= 1, got {self.inference_batch_size}"
            )
        if self.global_eval_max_cells < 1:
            raise ValueError(
                f"global_eval_max_cells must be >= 1, got {self.global_eval_max_cells}"
            )
        if self.global_eval_max_genes < 1:
            raise ValueError(
                f"global_eval_max_genes must be >= 1, got {self.global_eval_max_genes}"
            )
