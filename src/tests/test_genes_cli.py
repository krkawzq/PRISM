from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np

from prism.cli.genes.common import compute_hvg_ranking_from_adata
from prism.cli.genes.filter import filter_genes_command
from prism.cli.genes.merge import merge_genes_command
from prism.cli.genes.rank import rank_genes_command
from prism.io import read_gene_list_spec


class GenesCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_toy_h5ad(self) -> Path:
        path = self.root / "toy.h5ad"
        counts = np.array(
            [
                [10.0, 5.0, 1.0, 0.0],
                [12.0, 4.0, 0.0, 1.0],
                [11.0, 6.0, 3.0, 2.0],
            ],
            dtype=np.float32,
        )
        signal = np.array(
            [
                [9.0, 0.2, 0.1, 0.0],
                [8.0, 0.1, 1.5, 0.2],
                [8.5, 0.3, 1.0, 2.0],
            ],
            dtype=np.float32,
        )
        adata = ad.AnnData(X=counts)
        adata.var_names = ["MT-CO1", "RPL3", "GeneA", "GeneB"]
        adata.obs_names = ["c1", "c2", "c3"]
        adata.layers["signal"] = signal
        adata.write_h5ad(path)
        return path

    def test_rank_merge_filter_chain(self) -> None:
        h5ad_path = self._write_toy_h5ad()
        rank_json = self.root / "rank.json"
        rank_txt = self.root / "rank.txt"
        merged_json = self.root / "merged.json"
        merged_txt = self.root / "merged.txt"
        filtered_json = self.root / "filtered.json"
        filtered_txt = self.root / "filtered.txt"
        removed_json = self.root / "removed.json"
        removed_txt = self.root / "removed.txt"
        restrict_txt = self.root / "restrict.txt"
        restrict_txt.write_text(
            "MT-CO1\nGeneA\nGeneB\nMissingGene\n",
            encoding="utf-8",
        )

        rank_genes_command(
            input_path=h5ad_path,
            method="signal-variance",
            output_ranked_genes=rank_txt,
            output_json=rank_json,
            top_k=3,
            restrict_genes_path=restrict_txt,
            max_cells=None,
            random_seed=0,
            hvg_flavor="seurat_v3",
            prior_source="global",
            label=None,
        )

        ranked = read_gene_list_spec(rank_json)
        self.assertEqual(ranked.method, "signal-variance")
        self.assertEqual(ranked.metadata["score_order"], "descending")
        self.assertEqual(ranked.metadata["preview_top_k"], 3)
        self.assertEqual(ranked.metadata["n_requested_restrict_genes"], 4)
        self.assertEqual(ranked.metadata["n_missing_restrict_genes"], 1)
        self.assertEqual(ranked.gene_names, ["GeneB", "GeneA", "MT-CO1"])

        merge_genes_command(
            input_paths=[rank_json, rank_json],
            output_ranked_genes=merged_txt,
            output_json=merged_json,
            method="rank-sum",
            gene_set_mode="exact",
        )
        merged = read_gene_list_spec(merged_json)
        self.assertEqual(merged.method, "merge:rank-sum")
        self.assertEqual(merged.metadata["score_order"], "ascending")
        self.assertEqual(merged.metadata["merged_source_method"], "signal-variance")
        self.assertEqual(merged.gene_names, ranked.gene_names)

        filter_genes_command(
            input_genes=rank_json,
            output_genes=filtered_txt,
            output_json=filtered_json,
            removed_genes=removed_txt,
            removed_json=removed_json,
            species="human",
            config_path=None,
            config_only=False,
            dry_run=False,
        )
        filtered = read_gene_list_spec(filtered_json)
        removed = read_gene_list_spec(removed_json)
        self.assertEqual(filtered.gene_names, ["GeneB", "GeneA"])
        self.assertEqual(removed.gene_names, ["MT-CO1"])

    def test_merge_accepts_text_and_json_inputs(self) -> None:
        json_path = self.root / "rank.json"
        text_path = self.root / "rank.txt"
        merged_json = self.root / "merged.json"
        merged_txt = self.root / "merged.txt"
        json_path.write_text(
            (
                "{\n"
                '  "gene_names": ["GeneA", "GeneB", "GeneC"],\n'
                '  "scores": [3.0, 2.0, 1.0],\n'
                '  "method": "signal-variance"\n'
                "}\n"
            ),
            encoding="utf-8",
        )
        text_path.write_text("GeneA\nGeneB\nGeneC\n", encoding="utf-8")

        merge_genes_command(
            input_paths=[json_path, text_path],
            output_ranked_genes=merged_txt,
            output_json=merged_json,
            method="rank-sum",
            gene_set_mode="exact",
        )
        merged = read_gene_list_spec(merged_json)
        self.assertEqual(merged.gene_names, ["GeneA", "GeneB", "GeneC"])
        self.assertEqual(merged.metadata["input_declared_methods"], ["signal-variance"])

    def test_compute_hvg_ranking_accepts_variances_norm(self) -> None:
        adata = ad.AnnData(X=np.ones((3, 2), dtype=np.float32))
        adata.var_names = ["GeneA", "GeneB"]

        def fake_highly_variable_genes(
            adata_obj: ad.AnnData,
            *,
            flavor: str,
            inplace: bool,
        ) -> None:
            del flavor, inplace
            adata_obj.var["variances_norm"] = np.asarray([0.2, 0.8], dtype=np.float64)

        with patch("scanpy.pp.highly_variable_genes", new=fake_highly_variable_genes):
            gene_names, scores = compute_hvg_ranking_from_adata(
                adata,
                flavor="seurat_v3",
            )

        self.assertEqual(list(gene_names), ["GeneA", "GeneB"])
        self.assertTrue(np.allclose(scores, [0.2, 0.8]))


if __name__ == "__main__":
    unittest.main()
