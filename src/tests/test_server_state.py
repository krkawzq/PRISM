from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prism.server.config import ServerConfig
from prism.server.state import AppState


class _DummyVarNames:
    def __init__(self, values: list[str]) -> None:
        self._values = np.asarray(values, dtype=object)

    def astype(self, _dtype: type[str]) -> np.ndarray:
        return self._values.astype(str)


class _DummyAnnData:
    def __init__(self) -> None:
        self.X = np.asarray([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=np.float64)
        self.layers: dict[str, np.ndarray] = {}
        self.var_names = _DummyVarNames(["GeneA", "GeneB"])
        self.obs = pd.DataFrame(
            {"condition": ["a", "a", "b"]},
            index=["c0", "c1", "c2"],
        )
        self.n_obs = self.X.shape[0]
        self.n_vars = self.X.shape[1]


def test_app_state_cache_round_trip_requires_loaded_context() -> None:
    state = AppState(ServerConfig())
    with pytest.raises(RuntimeError, match="dataset is not loaded"):
        state.current_context_key()

    state.set_cache("summary", "x", 1)
    assert state.get_cache("summary", "x") == 1


def test_app_state_load_builds_dataset_and_clears_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    h5ad_path = tmp_path / "dataset.h5ad"
    h5ad_path.write_text("placeholder", encoding="utf-8")
    state = AppState(ServerConfig())
    state.set_cache("summary", "stale", {"x": 1})

    fake_module = SimpleNamespace(read_h5ad=lambda _path: _DummyAnnData())
    monkeypatch.setitem(__import__("sys").modules, "anndata", fake_module)
    monkeypatch.setattr("prism.server.state.load_checkpoint_state", lambda *args, **kwargs: None)

    loaded = state.load(h5ad_path=str(h5ad_path), ckpt_path=None, layer=None)

    assert loaded.n_cells == 3
    assert loaded.n_genes == 2
    assert loaded.dataset.gene_to_idx == {"GeneA": 0, "GeneB": 1}
    assert tuple(loaded.dataset.label_values) == ("condition",)
    assert state.get_cache("summary", "stale") is None
    assert len(loaded.context_key) == 40


def test_app_state_make_cache_key_includes_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    h5ad_path = tmp_path / "dataset.h5ad"
    h5ad_path.write_text("placeholder", encoding="utf-8")
    state = AppState(ServerConfig())
    fake_module = SimpleNamespace(read_h5ad=lambda _path: _DummyAnnData())
    monkeypatch.setitem(__import__("sys").modules, "anndata", fake_module)
    monkeypatch.setattr("prism.server.state.load_checkpoint_state", lambda *args, **kwargs: None)
    state.load(h5ad_path=str(h5ad_path), ckpt_path=None, layer=None)

    key1 = state.make_cache_key("summary", "dataset")
    key2 = state.make_cache_key("summary", "dataset")
    key3 = state.make_cache_key("summary", "other")

    assert key1 == key2
    assert key1 != key3
