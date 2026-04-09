from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import (
        CheckpointSummary,
        GeneAnalysis,
        GeneBrowsePage,
        GeneFitParams,
        GeneSummary,
        KBulkAnalysis,
        KBulkParams,
        browse_gene_candidates,
        build_checkpoint_summary,
        build_dataset_summary,
        build_gene_analysis,
        compute_kbulk_analysis,
        search_gene_candidates,
    )

__all__ = [
    "CheckpointSummary",
    "GeneAnalysis",
    "GeneBrowsePage",
    "GeneFitParams",
    "GeneSummary",
    "KBulkAnalysis",
    "KBulkParams",
    "browse_gene_candidates",
    "build_checkpoint_summary",
    "build_dataset_summary",
    "build_gene_analysis",
    "compute_kbulk_analysis",
    "search_gene_candidates",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        module = import_module(".analysis", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
