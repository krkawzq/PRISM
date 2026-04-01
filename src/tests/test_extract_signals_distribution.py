from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import anndata as ad
import numpy as np

from prism.cli.extract.signals import extract_signals_command
from prism.model import ModelCheckpoint, PriorGrid, save_checkpoint


def _prior_for_distribution(*, distribution: str, gene_names: list[str]) -> PriorGrid:
    if distribution == "poisson":
        return PriorGrid(
            gene_names=gene_names,
            p_grid=np.asarray([[1.0, 2.0, 4.0] for _ in gene_names], dtype=np.float64),
            weights=np.asarray([[0.2, 0.5, 0.3] for _ in gene_names], dtype=np.float64),
            S=4.0,
            grid_domain="rate",
            distribution="poisson",
        )
    return PriorGrid(
        gene_names=gene_names,
        p_grid=np.asarray([[0.2, 0.4, 0.8] for _ in gene_names], dtype=np.float64),
        weights=np.asarray([[0.2, 0.5, 0.3] for _ in gene_names], dtype=np.float64),
        S=4.0,
        grid_domain="p",
        distribution=distribution,
    )


class ExtractSignalsDistributionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_h5ad(self) -> Path:
        path = self.root / "toy.h5ad"
        adata = ad.AnnData(X=np.asarray([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32))
        adata.var_names = ["GeneA", "GeneB"]
        adata.obs_names = ["cell0", "cell1"]
        adata.write_h5ad(path)
        return path

    def _write_checkpoint(self, distribution: str) -> Path:
        checkpoint_path = self.root / f"{distribution}.pkl"
        priors = _prior_for_distribution(distribution=distribution, gene_names=["GeneA", "GeneB"])
        checkpoint = ModelCheckpoint(
            gene_names=["GeneA", "GeneB"],
            priors=priors,
            scale=None,
            fit_config={"likelihood": distribution, "nb_overdispersion": 0.1},
            metadata={
                "schema_version": 2,
                "fit_distribution": distribution,
                "posterior_distribution": distribution,
                "grid_domain": priors.grid_domain,
                "reference_gene_names": ["GeneB"],
            },
            label_priors={},
            label_scales={},
        )
        save_checkpoint(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_extract_signals_dry_run_accepts_all_supported_distributions(self) -> None:
        h5ad_path = self._write_h5ad()
        for distribution in ("binomial", "negative_binomial", "poisson"):
            checkpoint_path = self._write_checkpoint(distribution)
            result = extract_signals_command(
                checkpoint_path=checkpoint_path,
                h5ad_path=h5ad_path,
                output_path=self.root / f"{distribution}_signals.h5ad",
                layer=None,
                genes_path=None,
                output_mode="fitted-only",
                prior_source="global",
                label_key=None,
                batch_size=16,
                device="cpu",
                torch_dtype="float32",
                dtype="float32",
                channels=None,
                dry_run=True,
            )
            self.assertEqual(result, 0)

    def test_extract_signals_poisson_checkpoint_writes_map_rate_channel(self) -> None:
        h5ad_path = self._write_h5ad()
        checkpoint_path = self._write_checkpoint("poisson")
        output_path = self.root / "poisson_signals.h5ad"

        result = extract_signals_command(
            checkpoint_path=checkpoint_path,
            h5ad_path=h5ad_path,
            output_path=output_path,
            layer=None,
            genes_path=None,
            output_mode="fitted-only",
            prior_source="global",
            label_key=None,
            batch_size=16,
            device="cpu",
            torch_dtype="float32",
            dtype="float32",
            channels=["map_rate"],
            dry_run=False,
        )

        self.assertEqual(result, 0)
        self.assertTrue(output_path.exists())
        output = ad.read_h5ad(output_path)
        self.assertIn("map_rate", output.layers)
        self.assertEqual(output.layers["map_rate"].shape, (2, 2))
        self.assertTrue(np.isfinite(np.asarray(output.layers["map_rate"]).astype(np.float64)).all())


if __name__ == "__main__":
    unittest.main()
