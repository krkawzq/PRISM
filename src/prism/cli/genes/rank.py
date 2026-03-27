from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import typer

from .common import (
    checkpoint_prior_entropy_scores,
    compute_hvg_ranking_from_adata,
    compute_lognorm_ranking,
    console,
    write_json,
)


def rank_genes_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad or checkpoint path."
    ),
    method: str = typer.Option(..., "--method", help="Ranking method."),
    output_ranked_genes: Path = typer.Option(
        ..., "--output-ranked-genes", help="Output text file with ranked genes."
    ),
    output_json: Path | None = typer.Option(
        None, "--output-json", help="Optional JSON summary path."
    ),
    top_k: int | None = typer.Option(
        None, min=1, help="Optional top-k summary for the JSON payload."
    ),
    hvg_flavor: str = typer.Option(
        "seurat_v3", help="Scanpy HVG flavor for HVG-based methods."
    ),
    prior_source: str = typer.Option(
        "global", help="Checkpoint prior source used by prior-entropy methods."
    ),
    label: str | None = typer.Option(None, help="Label used when prior_source=label."),
) -> int:
    input_path = input_path.expanduser().resolve()
    descending = True
    metadata: dict[str, object] = {}
    if method == "prior-entropy":
        gene_names, scores = checkpoint_prior_entropy_scores(
            input_path, prior_source=prior_source, label=label
        )
        metadata.update(
            {
                "score_definition": "prior entropy of F_g",
                "prior_source": prior_source,
                "label": label,
            }
        )
    elif method == "prior-entropy-reverse":
        gene_names, scores = checkpoint_prior_entropy_scores(
            input_path, prior_source=prior_source, label=label
        )
        metadata.update(
            {
                "score_definition": "prior entropy of F_g",
                "prior_source": prior_source,
                "label": label,
            }
        )
        descending = False
    else:
        adata = ad.read_h5ad(input_path)
        if method == "hvg":
            gene_names, scores = compute_hvg_ranking_from_adata(
                adata, flavor=hvg_flavor
            )
            metadata["hvg_flavor"] = hvg_flavor
        elif method == "lognorm-variance":
            gene_names, scores = compute_lognorm_ranking(adata, dispersion=False)
            metadata["score_definition"] = "variance(log1p(normalize_total(X)))"
        elif method == "lognorm-dispersion":
            gene_names, scores = compute_lognorm_ranking(adata, dispersion=True)
            metadata["score_definition"] = (
                "variance(log1p(normalize_total(X))) / mean(log1p(normalize_total(X)))"
            )
        else:
            raise ValueError(f"unsupported method: {method}")
    order = np.argsort(scores)
    if descending:
        order = order[::-1]
    ranked = [str(gene_names[idx]) for idx in order.tolist()]
    output_ranked_genes = output_ranked_genes.expanduser().resolve()
    output_ranked_genes.parent.mkdir(parents=True, exist_ok=True)
    output_ranked_genes.write_text("\n".join(ranked) + "\n", encoding="utf-8")
    if output_json is not None:
        output_json = output_json.expanduser().resolve()
        preview_order = (
            order[:top_k] if top_k is not None else order[: min(50, len(order))]
        )
        write_json(
            output_json,
            {
                "source_path": str(input_path),
                "method": method,
                "top_k": None if top_k is None else int(top_k),
                "gene_names": [str(gene_names[idx]) for idx in preview_order.tolist()],
                "scores": [float(scores[idx]) for idx in preview_order.tolist()],
                "metadata": metadata,
            },
        )
    console.print(f"[bold green]Saved[/bold green] {output_ranked_genes}")
    return 0


__all__ = ["rank_genes_command"]
