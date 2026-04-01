from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import anndata as ad
import numpy as np

from prism.cli.data.downsample import downsample_command
from prism.cli.data.subset_genes import subset_genes_command


class DataCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_toy_h5ad(self) -> Path:
        path = self.root / "toy.h5ad"
        counts = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0, 9.0],
            ],
            dtype=np.float32,
        )
        adata = ad.AnnData(X=counts)
        adata.var_names = ["GeneA", "GeneB", "GeneC", "GeneD"]
        adata.obs_names = [f"cell{i}" for i in range(6)]
        adata.obs["treatment"] = ["ctrl", "ctrl", "ctrl", "stim", "stim", "stim"]
        adata.write_h5ad(path)
        return path

    def test_subset_genes_supports_structured_gene_list(self) -> None:
        h5ad_path = self._write_toy_h5ad()
        genes_json = self.root / "genes.json"
        output_path = self.root / "subset.h5ad"
        genes_json.write_text(
            (
                "{\n"
                '  "gene_names": ["GeneC", "GeneA", "MissingGene"],\n'
                '  "method": "signal-variance",\n'
                '  "metadata": {"score_order": "descending"}\n'
                "}\n"
            ),
            encoding="utf-8",
        )

        subset_genes_command(
            input_path=h5ad_path,
            genes_path=genes_json,
            output_path=output_path,
            allow_missing=True,
        )

        subset = ad.read_h5ad(output_path)
        self.assertEqual(subset.var_names.tolist(), ["GeneC", "GeneA"])
        metadata = subset.uns["gene_subset"]
        self.assertEqual(metadata["gene_list_method"], "signal-variance")
        self.assertEqual(metadata["n_requested_genes"], 3)
        self.assertEqual(metadata["n_selected_genes"], 2)
        self.assertEqual(metadata["n_missing_genes"], 1)

    def test_downsample_preserves_per_class_minimum(self) -> None:
        h5ad_path = self._write_toy_h5ad()
        output_path = self.root / "downsampled.h5ad"

        downsample_command(
            input_path=h5ad_path,
            output_path=output_path,
            label_key="treatment",
            fraction=0.34,
            seed=7,
            per_class_min=1,
        )

        sampled = ad.read_h5ad(output_path)
        counts = sampled.obs["treatment"].astype("category").value_counts().to_dict()
        self.assertEqual(int(sampled.n_vars), 4)
        self.assertEqual(int(sampled.n_obs), 2)
        self.assertEqual(int(counts["ctrl"]), 1)
        self.assertEqual(int(counts["stim"]), 1)
        metadata = sampled.uns["sampling"]
        self.assertEqual(metadata["method"], "stratified_fraction")
        self.assertEqual(metadata["label_key"], "treatment")
        self.assertEqual(metadata["n_obs_before"], 6)
        self.assertEqual(metadata["n_obs_after"], 2)


if __name__ == "__main__":
    unittest.main()
