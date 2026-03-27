#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute the overlapping gene list between two h5ad datasets."
    )
    parser.add_argument("input_a", type=Path)
    parser.add_argument("input_b", type=Path)
    parser.add_argument(
        "--output-genes",
        type=Path,
        required=True,
        help="Text file with one overlapping gene per line.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--sort",
        choices=("first", "second", "alpha"),
        default="first",
        help="Ordering of overlapping genes.",
    )
    return parser.parse_args()


def load_var_names(path: Path) -> list[str]:
    adata = ad.read_h5ad(path, backed="r")
    try:
        return [str(name) for name in adata.var_names.tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()


def main() -> None:
    args = parse_args()
    input_a = args.input_a.expanduser().resolve()
    input_b = args.input_b.expanduser().resolve()
    output_genes = args.output_genes.expanduser().resolve()
    output_json = (
        None if args.output_json is None else args.output_json.expanduser().resolve()
    )

    genes_a = load_var_names(input_a)
    genes_b = load_var_names(input_b)
    set_a = set(genes_a)
    set_b = set(genes_b)
    overlap = set_a & set_b

    if args.sort == "first":
        ordered = [gene for gene in genes_a if gene in overlap]
    elif args.sort == "second":
        ordered = [gene for gene in genes_b if gene in overlap]
    else:
        ordered = sorted(overlap)

    output_genes.parent.mkdir(parents=True, exist_ok=True)
    output_genes.write_text("\n".join(ordered) + "\n", encoding="utf-8")

    payload = {
        "input_a": str(input_a),
        "input_b": str(input_b),
        "n_genes_a": len(genes_a),
        "n_genes_b": len(genes_b),
        "n_overlap": len(ordered),
        "sort": args.sort,
        "output_genes": str(output_genes),
        "first10": ordered[:10],
    }
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"input_a genes : {len(genes_a)}")
    print(f"input_b genes : {len(genes_b)}")
    print(f"overlap       : {len(ordered)}")
    print(f"saved genes   : {output_genes}")
    if output_json is not None:
        print(f"saved json    : {output_json}")
    print(f"first10       : {ordered[:10]}")


if __name__ == "__main__":
    main()
