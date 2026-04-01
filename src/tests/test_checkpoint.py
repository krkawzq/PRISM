from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np

from prism.model import (
    ModelCheckpoint,
    PriorGrid,
    ScaleMetadata,
    load_checkpoint,
    save_checkpoint,
)


def _make_prior_grid(gene_names: list[str]) -> PriorGrid:
    p_grid = np.linspace(0.05, 0.95, 8, dtype=np.float64)
    weights = np.ones((len(gene_names), 8), dtype=np.float64) / 8.0
    return PriorGrid(gene_names=gene_names, p_grid=p_grid, weights=weights, S=10.0)


class TestCheckpointSchema:
    def setup_method(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def teardown_method(self) -> None:
        self._tmpdir.cleanup()

    def test_v2_full_checkpoint_round_trip(self) -> None:
        path = self.root / "v2_full.pkl"
        priors = _make_prior_grid(["GeneA", "GeneB"])
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA", "GeneB"],
            priors=priors,
            scale=ScaleMetadata(S=10.0, mean_reference_count=10.0),
            fit_config={"grid_size": 512, "n_iter": 100},
            metadata={
                "schema_version": 2,
                "source": "test",
                "fit_distribution": "binomial",
                "posterior_distribution": "binomial",
                "grid_domain": "p",
            },
            label_priors={
                "ctrl": _make_prior_grid(["GeneA"]),
                "stim": _make_prior_grid(["GeneA"]),
            },
            label_scales={
                "ctrl": ScaleMetadata(S=10.0, mean_reference_count=10.0),
                "stim": ScaleMetadata(S=11.0, mean_reference_count=11.0),
            },
        )
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)
        assert loaded.gene_names == ["GeneA", "GeneB"]
        assert loaded.priors is not None
        assert loaded.priors.S == 10.0
        assert set(loaded.label_priors) == {"ctrl", "stim"}
        assert set(loaded.label_scales) == {"ctrl", "stim"}
        assert loaded.scale is not None
        assert loaded.scale.S == 10.0

    def test_v2_priors_none_round_trip(self) -> None:
        path = self.root / "v2_no_priors.pkl"
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA"],
            priors=None,
            scale=ScaleMetadata(S=10.0, mean_reference_count=10.0),
            fit_config={},
            metadata={
                "schema_version": 2,
                "fit_distribution": "binomial",
                "posterior_distribution": "binomial",
                "grid_domain": "p",
            },
            label_priors={},
            label_scales={},
        )
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)
        assert loaded.priors is None
        assert loaded.gene_names == ["GeneA"]

    def test_v2_scale_none_round_trip(self) -> None:
        path = self.root / "v2_no_scale.pkl"
        priors = _make_prior_grid(["GeneA"])
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA"],
            priors=priors,
            scale=None,
            fit_config={},
            metadata={
                "schema_version": 2,
                "fit_distribution": "binomial",
                "posterior_distribution": "binomial",
                "grid_domain": "p",
            },
            label_priors={},
            label_scales={},
        )
        save_checkpoint(checkpoint, path)
        loaded = load_checkpoint(path)
        assert loaded.scale is None
        assert loaded.priors is not None

    def test_v1_checkpoint_loads_correctly(self) -> None:
        path = self.root / "v1.pkl"
        p_grid = np.linspace(0.05, 0.95, 4, dtype=np.float64)
        weights = np.ones((2, 4), dtype=np.float64) / 4.0
        v1_payload = {
            "schema_version": 1,
            "gene_names": ["GeneA", "GeneB"],
            "priors": {
                "gene_names": ["GeneA", "GeneB"],
                "p_grid": p_grid,
                "weights": weights,
                "S": 10.0,
                "grid_domain": "p",
            },
            "scale": {"S": 10.0, "mean_reference_count": 10.0},
            "fit_config": {"grid_size": 256},
            "metadata": {"source": "legacy"},
        }
        with open(path, "wb") as fh:
            pickle.dump(v1_payload, fh)
        loaded = load_checkpoint(path)
        assert loaded.gene_names == ["GeneA", "GeneB"]
        assert loaded.priors is not None
        assert loaded.priors.S == 10.0
        assert loaded.priors.gene_names == ["GeneA", "GeneB"]
        assert loaded.scale is not None
        assert loaded.scale.S == 10.0
        assert loaded.fit_config == {"grid_size": 256}
        assert loaded.metadata["source"] == "legacy"
        assert loaded.metadata["fit_distribution"] == "binomial"
        assert loaded.metadata["posterior_distribution"] == "binomial"
        assert loaded.metadata["grid_domain"] == "p"
        assert loaded.label_priors == {}
        assert loaded.label_scales == {}

    def test_v1_implicit_schema_version_loads(self) -> None:
        path = self.root / "v1_no_version.pkl"
        p_grid = np.linspace(0.05, 0.95, 4, dtype=np.float64)
        weights = np.ones((2, 4), dtype=np.float64) / 4.0
        v1_payload = {
            "gene_names": ["GeneA", "GeneB"],
            "priors": {
                "gene_names": ["GeneA", "GeneB"],
                "p_grid": p_grid,
                "weights": weights,
                "S": 8.0,
                "grid_domain": "p",
            },
            "scale": {"S": 8.0, "mean_reference_count": 8.0},
        }
        with open(path, "wb") as fh:
            pickle.dump(v1_payload, fh)
        loaded = load_checkpoint(path)
        assert loaded.priors.S == 8.0
        assert loaded.scale.S == 8.0

    def test_save_then_load_idempotent(self) -> None:
        path = self.root / "idempotent.pkl"
        priors = _make_prior_grid(["GeneX", "GeneY"])
        checkpoint = ModelCheckpoint(
            gene_names=["GeneX", "GeneY"],
            priors=priors,
            scale=ScaleMetadata(S=5.0, mean_reference_count=5.0),
            fit_config={"n_iter": 50, "sigma_bins": 2.0},
            metadata={
                "schema_version": 2,
                "note": "idempotent test",
                "fit_distribution": "binomial",
                "posterior_distribution": "binomial",
                "grid_domain": "p",
            },
            label_priors={
                "treated": _make_prior_grid(["GeneX"]),
            },
            label_scales={
                "treated": ScaleMetadata(S=5.0, mean_reference_count=5.0),
            },
        )
        for _ in range(3):
            save_checkpoint(checkpoint, path)
            loaded = load_checkpoint(path)
            assert loaded.gene_names == ["GeneX", "GeneY"]
            assert loaded.priors.gene_names == ["GeneX", "GeneY"]
            assert loaded.scale.S == 5.0
            assert set(loaded.label_priors) == {"treated"}
            assert set(loaded.label_scales) == {"treated"}
            assert loaded.fit_config["n_iter"] == 50
