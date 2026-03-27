#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json

import anndata as ad
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.traceback import install as install_rich_traceback

console = Console()

install_rich_traceback(show_locals=False)


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

    job = Table(show_header=False, box=None)
    job.add_row("Input", str(input_path))
    job.add_row("Gene list", str(gene_list_path))
    job.add_row("Output", str(output_path))
    console.print(Panel(job, title="Gene Subset AnnData", border_style="cyan"))

    with console.status("Loading AnnData and gene list..."):
        adata = ad.read_h5ad(input_path)
        gene_names, payload = load_gene_list(gene_list_path)
        indices = resolve_gene_indices(adata, gene_names)

    with console.status("Subsetting genes and writing output..."):
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

    summary = Table(title="Subset Summary")
    summary.add_column("Field")
    summary.add_column("Value", overflow="fold")
    summary.add_row("Cells", f"{adata.n_obs} -> {subset.n_obs}")
    summary.add_row("Genes", f"{adata.n_vars} -> {subset.n_vars}")
    summary.add_row("Method", str(payload.get("method", "")))
    summary.add_row("First genes", ", ".join(gene_names[:5]) if gene_names else "None")
    summary.add_row("Output", str(output_path))
    console.print(summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(
            Panel(str(exc), title="gene_subset_anndata failed", border_style="red")
        )
        raise SystemExit(1) from exc
