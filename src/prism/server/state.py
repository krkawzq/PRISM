from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import anndata as ad
import numpy as np

from prism.model import PoolEstimate, PriorEngine

from .config import ServerConfig
from .services.checkpoints import load_checkpoint, validate_checkpoint_against_dataset
from .services.datasets import (
    build_gene_to_idx,
    compute_detected_counts,
    compute_gene_totals,
    compute_totals,
    select_matrix,
)


@dataclass(slots=True)
class LoadedState:
    h5ad_path: Path
    ckpt_path: Path | None
    layer: str | None
    adata: ad.AnnData
    matrix: Any
    totals: np.ndarray
    gene_names: np.ndarray
    gene_to_idx: dict[str, int]
    gene_total_counts: np.ndarray
    gene_detected_counts: np.ndarray
    checkpoint: dict[str, Any] | None
    engine: PriorEngine | None
    checkpoint_s_hat: float | None
    fitted_gene_indices: np.ndarray
    fitted_gene_names: tuple[str, ...]

    @property
    def n_cells(self) -> int:
        return int(self.adata.n_obs)

    @property
    def n_genes(self) -> int:
        return int(self.adata.n_vars)


class AppState:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._loaded: LoadedState | None = None
        self._pool_estimate: PoolEstimate | None = None
        self._fit_cache: dict[str, dict[str, Any]] = {}

    @property
    def loaded(self) -> LoadedState | None:
        with self._lock:
            return self._loaded

    @property
    def pool_estimate(self) -> PoolEstimate | None:
        with self._lock:
            return self._pool_estimate

    def require_loaded(self) -> LoadedState:
        loaded = self.loaded
        if loaded is None:
            raise RuntimeError("dataset is not loaded")
        return loaded

    def set_pool_estimate(self, estimate: PoolEstimate) -> None:
        with self._lock:
            self._pool_estimate = estimate

    def get_cached_fit(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._fit_cache.get(key)

    def set_cached_fit(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._fit_cache[key] = value

    def load(
        self, *, h5ad_path: str, ckpt_path: str | None, layer: str | None
    ) -> LoadedState:
        resolved_h5ad_path = Path(h5ad_path).expanduser().resolve()
        resolved_ckpt_path = (
            Path(ckpt_path).expanduser().resolve()
            if ckpt_path and ckpt_path.strip()
            else None
        )

        adata = ad.read_h5ad(resolved_h5ad_path)
        matrix = select_matrix(adata, layer)
        totals = compute_totals(matrix)
        gene_names = np.asarray(adata.var_names.astype(str))
        gene_to_idx = build_gene_to_idx(gene_names)
        gene_total_counts = compute_gene_totals(matrix)
        gene_detected_counts = compute_detected_counts(matrix)

        checkpoint: dict[str, Any] | None = None
        engine: PriorEngine | None = None
        checkpoint_s_hat: float | None = None
        fitted_gene_names: tuple[str, ...] = ()
        fitted_gene_indices = np.asarray([], dtype=np.int64)

        if resolved_ckpt_path is not None:
            checkpoint = load_checkpoint(resolved_ckpt_path)
            engine = checkpoint.get("engine")
            if not isinstance(engine, PriorEngine):
                raise TypeError("checkpoint does not contain a valid PriorEngine")

            validate_checkpoint_against_dataset(checkpoint, gene_names)
            fitted_gene_names = tuple(
                name
                for name in engine.gene_names
                if name in gene_to_idx and engine.is_fitted(name)
            )
            fitted_gene_indices = np.asarray(
                [gene_to_idx[name] for name in fitted_gene_names],
                dtype=np.int64,
            )
            checkpoint_s_hat = float(checkpoint["s_hat"])

        loaded = LoadedState(
            h5ad_path=resolved_h5ad_path,
            ckpt_path=resolved_ckpt_path,
            layer=layer,
            adata=adata,
            matrix=matrix,
            totals=totals,
            gene_names=gene_names,
            gene_to_idx=gene_to_idx,
            gene_total_counts=gene_total_counts,
            gene_detected_counts=gene_detected_counts,
            checkpoint=checkpoint,
            engine=engine,
            checkpoint_s_hat=checkpoint_s_hat,
            fitted_gene_indices=fitted_gene_indices,
            fitted_gene_names=fitted_gene_names,
        )
        with self._lock:
            self._loaded = loaded
            self._pool_estimate = None
            self._fit_cache = {}
        return loaded
