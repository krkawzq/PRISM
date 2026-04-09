from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np

from .config import ServerConfig
from .services.checkpoints import CheckpointState, load_checkpoint_state
from .services.datasets import (
    build_gene_to_idx,
    compute_cell_zero_fraction,
    compute_detected_counts,
    compute_gene_totals,
    compute_totals,
    detect_label_columns,
    select_matrix,
)


@dataclass(slots=True)
class DatasetState:
    h5ad_path: Path
    layer: str | None
    adata: Any
    matrix: Any
    totals: np.ndarray
    gene_names: np.ndarray
    gene_names_lower: tuple[str, ...]
    gene_to_idx: dict[str, int]
    gene_lower_to_idx: dict[str, int]
    gene_total_counts: np.ndarray
    gene_detected_counts: np.ndarray
    cell_zero_fraction: np.ndarray
    ranked_gene_indices: np.ndarray
    label_values: dict[str, np.ndarray]


@dataclass(slots=True)
class LoadedState:
    context_key: str
    dataset: DatasetState
    checkpoint: CheckpointState | None

    @property
    def n_cells(self) -> int:
        return int(self.dataset.adata.n_obs)

    @property
    def n_genes(self) -> int:
        return int(self.dataset.adata.n_vars)

    @property
    def label_keys(self) -> tuple[str, ...]:
        return tuple(self.dataset.label_values)


class AppState:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._lock = RLock()
        self._loaded: LoadedState | None = None
        self._cache_store: dict[str, OrderedDict[str, Any]] = {
            "summary": OrderedDict(),
            "search": OrderedDict(),
            "analysis": OrderedDict(),
            "kbulk": OrderedDict(),
            "figures": OrderedDict(),
        }
        self._cache_limits = {
            "summary": 16,
            "search": 256,
            "analysis": 128,
            "kbulk": 64,
            "figures": 256,
        }

    @property
    def loaded(self) -> LoadedState | None:
        with self._lock:
            return self._loaded

    def require_loaded(self) -> LoadedState:
        loaded = self.loaded
        if loaded is None:
            raise RuntimeError("dataset is not loaded")
        return loaded

    def current_context_key(self) -> str:
        return self.require_loaded().context_key

    def get_cache(self, namespace: str, key: str) -> Any | None:
        with self._lock:
            store = self._cache_store.setdefault(namespace, OrderedDict())
            if key not in store:
                return None
            value = store.pop(key)
            store[key] = value
            return value

    def set_cache(self, namespace: str, key: str, value: Any) -> None:
        with self._lock:
            store = self._cache_store.setdefault(namespace, OrderedDict())
            if key in store:
                store.pop(key)
            store[key] = value
            limit = self._cache_limits.get(namespace, 64)
            while len(store) > limit:
                store.popitem(last=False)

    def get_or_create_cache(self, namespace: str, key: str, factory) -> Any:
        cached = self.get_cache(namespace, key)
        if cached is not None:
            return cached
        value = factory()
        self.set_cache(namespace, key, value)
        return value

    def make_cache_key(self, namespace: str, *parts: object) -> str:
        payload = (namespace, self.current_context_key(), *parts)
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def load(
        self, *, h5ad_path: str, ckpt_path: str | None, layer: str | None
    ) -> LoadedState:
        resolved_h5ad_path = Path(h5ad_path).expanduser().resolve()
        resolved_ckpt_path = (
            None
            if ckpt_path is None or not ckpt_path.strip()
            else Path(ckpt_path).expanduser().resolve()
        )
        loaded = self._load_uncached(resolved_h5ad_path, resolved_ckpt_path, layer)
        with self._lock:
            self._loaded = loaded
            for store in self._cache_store.values():
                store.clear()
        return loaded

    def _load_uncached(
        self, h5ad_path: Path, ckpt_path: Path | None, layer: str | None
    ) -> LoadedState:
        import anndata as ad

        adata = ad.read_h5ad(h5ad_path)
        matrix = select_matrix(adata, layer)
        totals = compute_totals(matrix)
        gene_names = np.asarray(adata.var_names.astype(str))
        gene_names_lower = tuple(str(name).lower() for name in gene_names.tolist())
        gene_to_idx = build_gene_to_idx(gene_names)
        gene_lower_to_idx: dict[str, int] = {}
        for idx, name in enumerate(gene_names_lower):
            gene_lower_to_idx.setdefault(name, idx)
        label_values = detect_label_columns(adata)
        gene_total_counts = compute_gene_totals(matrix)
        dataset = DatasetState(
            h5ad_path=h5ad_path,
            layer=layer,
            adata=adata,
            matrix=matrix,
            totals=totals,
            gene_names=gene_names,
            gene_names_lower=gene_names_lower,
            gene_to_idx=gene_to_idx,
            gene_lower_to_idx=gene_lower_to_idx,
            gene_total_counts=gene_total_counts,
            gene_detected_counts=compute_detected_counts(matrix),
            cell_zero_fraction=compute_cell_zero_fraction(matrix),
            ranked_gene_indices=np.argsort(gene_total_counts)[::-1].astype(np.int64),
            label_values=label_values,
        )
        checkpoint = (
            None
            if ckpt_path is None
            else load_checkpoint_state(
                ckpt_path,
                dataset_gene_names=gene_names.tolist(),
                gene_to_idx=gene_to_idx,
                available_label_keys=tuple(label_values),
            )
        )
        context_key = self._build_context_key(h5ad_path, ckpt_path, layer or "")
        return LoadedState(
            context_key=context_key,
            dataset=dataset,
            checkpoint=checkpoint,
        )

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
