#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Subset an AnnData object to genes listed in a text file or gene-list JSON."
    )
    parser.add_argument("input_h5ad", type=Path, help="Input AnnData file.")
    parser.add_argument(
        "gene_list_path", type=Path, help="Gene-list JSON or plain text file."
    )
    parser.add_argument("output_h5ad", type=Path, help="Output subset AnnData file.")
    return parser.parse_args()


def load_gene_list(path: Path) -> tuple[list[str], dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        payload = json.loads(text)
        gene_names = payload.get("gene_names")
        if not isinstance(gene_names, list) or not all(
            isinstance(name, str) for name in gene_names
        ):
            raise TypeError("gene-list JSON is missing a valid gene_names field")
        return gene_names, payload
    gene_names = [line.strip() for line in text.splitlines() if line.strip()]
    if not gene_names:
        raise ValueError("gene list text file is empty")
    return gene_names, {"method": "text-file", "top_k": len(gene_names)}


def resolve_gene_indices(adata: ad.AnnData, gene_names: list[str]) -> np.ndarray:
    name_to_idx = {str(name): idx for idx, name in enumerate(adata.var_names.tolist())}
    missing = [name for name in gene_names if name not in name_to_idx]
    if missing:
        raise ValueError(
            f"有 {len(missing)} 个基因不在输入 AnnData 中，例如 {missing[:5]}"
        )
    return np.asarray([name_to_idx[name] for name in gene_names], dtype=np.int64)


def main() -> None:
    args = parse_args()
    input_path = args.input_h5ad.expanduser().resolve()
    gene_list_path = args.gene_list_path.expanduser().resolve()
    output_path = args.output_h5ad.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(input_path)
    gene_names, payload = load_gene_list(gene_list_path)
    indices = resolve_gene_indices(adata, gene_names)

    subset = adata[:, indices].copy()
    top_k_value = payload.get("top_k", len(gene_names))
    subset.uns["gene_subset"] = {
        "source_h5ad": str(input_path),
        "gene_list_path": str(gene_list_path),
        "method": str(payload.get("method", "")),
        "top_k": int(top_k_value)
        if isinstance(top_k_value, (int, float))
        else int(len(gene_names)),
        "n_obs": int(subset.n_obs),
        "n_vars_before": int(adata.n_vars),
        "n_vars_after": int(subset.n_vars),
    }
    subset.write_h5ad(output_path)

    print(f"saved {output_path}")
    print(f"cells: {adata.n_obs} -> {subset.n_obs}")
    print(f"genes: {adata.n_vars} -> {subset.n_vars}")
    print(f"method: {payload.get('method', '')}")
    print(f"first5: {gene_names[:5]}")


if __name__ == "__main__":
    main()
