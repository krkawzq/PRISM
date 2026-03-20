from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import RLock
from typing import Any

import anndata as ad
import numpy as np

from prism.model import PoolFitReport, PriorEngine, PriorEngineSetting

from .config import ServerConfig
from .services.checkpoints import load_checkpoint_bundle
from .services.datasets import (
    build_gene_to_idx,
    compute_cell_zero_fraction,
    compute_detected_counts,
    compute_gene_totals,
    compute_totals,
    select_matrix,
)


@dataclass(slots=True)
class DatasetState:
    h5ad_path: Path
    layer: str | None
    adata: ad.AnnData
    matrix: Any
    totals: np.ndarray
    gene_names: np.ndarray
    gene_to_idx: dict[str, int]
    gene_total_counts: np.ndarray
    gene_detected_counts: np.ndarray
    cell_zero_fraction: np.ndarray


@dataclass(slots=True)
class ModelState:
    source: str
    ckpt_path: Path | None
    checkpoint: dict[str, Any] | None
    engine: PriorEngine | None
    s_hat: float | None
    pool_report: PoolFitReport | None
    r_hint: float | None

    def engine_for_fit(
        self,
        *,
        setting: PriorEngineSetting,
        device: str,
        gene_names: np.ndarray,
    ) -> PriorEngine:
        if self.engine is not None and self.engine.setting == setting:
            return self.engine
        return PriorEngine(gene_names.tolist(), setting=setting, device=device)


@dataclass(slots=True)
class LoadedState:
    dataset: DatasetState
    model: ModelState
    fitted_gene_indices: np.ndarray
    fitted_gene_names: tuple[str, ...]

    @property
    def n_cells(self) -> int:
        return int(self.dataset.adata.n_obs)

    @property
    def n_genes(self) -> int:
        return int(self.dataset.adata.n_vars)


class AppState:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._loaded: LoadedState | None = None
        self._pool_report: PoolFitReport | None = None
        self._loaded_context_key: str | None = None
        self._load_cache: dict[str, LoadedState] = {}
        self._cache_store: dict[str, dict[str, Any]] = {
            "summary": {},
            "search": {},
            "top_genes": {},
            "analysis": {},
            "fit": {},
            "figures": {},
            "global_eval": {},
            "html": {},
        }

    @property
    def loaded(self) -> LoadedState | None:
        with self._lock:
            return self._loaded

    @property
    def pool_report(self) -> PoolFitReport | None:
        with self._lock:
            return self._pool_report

    def require_loaded(self) -> LoadedState:
        loaded = self.loaded
        if loaded is None:
            raise RuntimeError("dataset is not loaded")
        return loaded

    def current_context_key(self) -> str:
        with self._lock:
            if self._loaded_context_key is None:
                raise RuntimeError("dataset is not loaded")
            return self._loaded_context_key

    def set_pool_report(self, report: PoolFitReport) -> None:
        with self._lock:
            self._pool_report = report

    def get_cached_fit(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            return self._cache_store.setdefault("fit", {}).get(key)

    def set_cached_fit(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._cache_store.setdefault("fit", {})[key] = value

    def get_cached_global_eval(self, key: str) -> Any | None:
        with self._lock:
            return self._cache_store.setdefault("global_eval", {}).get(key)

    def set_cached_global_eval(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache_store.setdefault("global_eval", {})[key] = value

    def get_cache(self, namespace: str, key: str) -> Any | None:
        with self._lock:
            return self._cache_store.setdefault(namespace, {}).get(key)

    def set_cache(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            self._cache_store.setdefault(namespace, {})[key] = value

    def make_cache_key(self, namespace: str, *parts: object) -> str:
        context_key = self.current_context_key()
        payload = (namespace, context_key, *parts)
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def load(
        self, *, h5ad_path: str, ckpt_path: str | None, layer: str | None
    ) -> LoadedState:
        print(
            f"[prism-server] load request h5ad={h5ad_path} ckpt={ckpt_path or '-'} layer={layer or 'X'}",
            flush=True,
        )
        resolved_h5ad_path = Path(h5ad_path).expanduser().resolve()
        resolved_ckpt_path = (
            Path(ckpt_path).expanduser().resolve()
            if ckpt_path and ckpt_path.strip()
            else None
        )
        context_key = self._build_context_key(
            resolved_h5ad_path, resolved_ckpt_path, layer or ""
        )

        with self._lock:
            cached_loaded = self._load_cache.get(context_key)
        if cached_loaded is not None:
            with self._lock:
                self._loaded = cached_loaded
                self._pool_report = cached_loaded.model.pool_report
                self._loaded_context_key = context_key
            print(
                f"[prism-server] load cache hit context={context_key[:8]} cells={cached_loaded.n_cells} genes={cached_loaded.n_genes}",
                flush=True,
            )
            return cached_loaded

        print(f"[prism-server] reading h5ad {resolved_h5ad_path}", flush=True)
        adata = ad.read_h5ad(resolved_h5ad_path)
        matrix = select_matrix(adata, layer)
        totals = compute_totals(matrix)
        gene_names = np.asarray(adata.var_names.astype(str))
        gene_to_idx = build_gene_to_idx(gene_names)
        gene_total_counts = compute_gene_totals(matrix)
        gene_detected_counts = compute_detected_counts(matrix)
        cell_zero_fraction = compute_cell_zero_fraction(matrix)

        dataset = DatasetState(
            h5ad_path=resolved_h5ad_path,
            layer=layer,
            adata=adata,
            matrix=matrix,
            totals=totals,
            gene_names=gene_names,
            gene_to_idx=gene_to_idx,
            gene_total_counts=gene_total_counts,
            gene_detected_counts=gene_detected_counts,
            cell_zero_fraction=cell_zero_fraction,
        )

        if resolved_ckpt_path is None:
            print(
                "[prism-server] no checkpoint provided; server will use dataset-only mode",
                flush=True,
            )
            model = ModelState(
                source="dataset-only",
                ckpt_path=None,
                checkpoint=None,
                engine=None,
                s_hat=None,
                pool_report=None,
                r_hint=None,
            )
            fitted_gene_names: tuple[str, ...] = ()
            fitted_gene_indices = np.asarray([], dtype=np.int64)
        else:
            print(f"[prism-server] loading checkpoint {resolved_ckpt_path}", flush=True)
            bundle = load_checkpoint_bundle(resolved_ckpt_path, gene_names)
            model = ModelState(
                source="checkpoint",
                ckpt_path=resolved_ckpt_path,
                checkpoint=bundle.checkpoint,
                engine=bundle.engine,
                s_hat=bundle.s_hat,
                pool_report=bundle.pool_report,
                r_hint=bundle.r_hint,
            )
            fitted_gene_names = tuple(
                name
                for name in bundle.engine.gene_names
                if name in gene_to_idx and bundle.engine.is_fitted(name)
            )
            fitted_gene_indices = np.asarray(
                [gene_to_idx[name] for name in fitted_gene_names],
                dtype=np.int64,
            )
            print(
                f"[prism-server] checkpoint ready fitted_genes={len(fitted_gene_names)} s_hat={bundle.s_hat:.4f}",
                flush=True,
            )

        loaded = LoadedState(
            dataset=dataset,
            model=model,
            fitted_gene_indices=fitted_gene_indices,
            fitted_gene_names=fitted_gene_names,
        )
        with self._lock:
            self._loaded = loaded
            self._pool_report = model.pool_report
            self._loaded_context_key = context_key
            self._load_cache[context_key] = loaded
        print(
            f"[prism-server] dataset loaded cells={loaded.n_cells} genes={loaded.n_genes} source={loaded.model.source}",
            flush=True,
        )
        return loaded

    def _build_context_key(
        self, h5ad_path: Path, ckpt_path: Path | None, layer: str
    ) -> str:
        payload = (
            self._path_signature(h5ad_path),
            None if ckpt_path is None else self._path_signature(ckpt_path),
            layer,
        )
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def _path_signature(path: Path) -> tuple[str, int, int]:
        stat = path.stat()
        return (str(path), int(stat.st_mtime_ns), int(stat.st_size))
