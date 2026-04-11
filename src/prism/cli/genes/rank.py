from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import typer

from prism.cli.common import (
    print_saved_path,
    resolve_int,
    resolve_optional_int,
    resolve_optional_path,
    resolve_optional_str,
    resolve_str,
    unwrap_typer_value,
)
from prism.io import read_string_list

from .common import (
    SUPPORTED_HVG_FLAVORS,
    SUPPORTED_PRIOR_SOURCES,
    SUPPORTED_RANK_METHODS,
    compute_ranking,
    console,
    filter_gene_scores,
    normalize_hvg_flavor,
    normalize_prior_source,
    normalize_rank_method,
    print_gene_summary,
    rank_gene_scores,
    read_gene_list,
    write_gene_list,
)


def _resolve_obs_values(
    values: list[str] | None | object,
    values_path: Path | None,
) -> list[str] | None:
    option_values = unwrap_typer_value(values)
    resolved: list[str] = []
    seen: set[str] = set()
    if option_values is not None:
        for item in option_values:
            for part in str(item).split(","):
                label = part.strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                resolved.append(label)
    if values_path is not None:
        for item in read_string_list(values_path):
            label = str(item).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            resolved.append(label)
    return resolved or None


def _dedupe(values: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        resolved.append(item)
    return resolved


def _map_output_gene_names(
    input_path: Path,
    *,
    gene_names: np.ndarray,
    output_gene_column: str | None,
) -> list[str]:
    resolved_gene_names = [str(gene) for gene in gene_names.tolist()]
    if output_gene_column is None:
        return resolved_gene_names
    if input_path.suffix.lower() != ".h5ad":
        raise ValueError("--output-gene-column is only supported for h5ad inputs")
    adata = ad.read_h5ad(input_path, backed="r")
    try:
        if output_gene_column not in adata.var.columns:
            raise KeyError(f"var column {output_gene_column!r} does not exist")
        index_lookup = {
            str(gene_name): idx
            for idx, gene_name in enumerate(adata.var_names.tolist())
        }
        output_values = np.asarray(adata.var[output_gene_column], dtype=object).reshape(
            -1
        )
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    mapped: list[str] = []
    missing: list[str] = []
    for gene_name in resolved_gene_names:
        idx = index_lookup.get(gene_name)
        if idx is None:
            missing.append(gene_name)
            continue
        mapped_value = str(output_values[idx]).strip()
        if not mapped_value:
            missing.append(gene_name)
            continue
        mapped.append(mapped_value)
    if missing:
        raise ValueError(
            f"could not map {len(missing)} genes through var column {output_gene_column!r}; "
            f"examples: {missing[:5]}"
        )
    return _dedupe(mapped)


def rank_genes_command(
    input_path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Input h5ad or checkpoint path."
    ),
    method: str = typer.Option(..., "--method", help="Ranking method."),
    output_path: Path = typer.Option(
        ..., "--output", "-o", help="Output text file with ranked genes."
    ),
    restrict_genes_path: Path | None = typer.Option(
        None,
        "--restrict-genes",
        exists=True,
        dir_okay=False,
        help="Optional text gene list restricting the ranked output.",
    ),
    max_cells: int | None = typer.Option(
        None, min=1, help="Maximum number of cells to use for h5ad-based methods."
    ),
    top_n: int | None = typer.Option(
        None, "--top-n", min=1, help="Optional number of top-ranked genes to write."
    ),
    random_seed: int = typer.Option(
        0, "--seed", min=0, help="Random seed used for cell subsampling."
    ),
    hvg_flavor: str = typer.Option("seurat_v3", help="Scanpy HVG flavor."),
    prior_source: str = typer.Option(
        "global", help="Checkpoint prior source for prior-entropy methods."
    ),
    label: str | None = typer.Option(None, help="Label used when prior_source=label."),
    obs_key: str | None = typer.Option(
        None,
        "--obs-key",
        help="Optional obs column used to restrict h5ad-based ranking to selected cells.",
    ),
    obs_values: list[str] | None = typer.Option(
        None,
        "--obs-value",
        help="Repeatable obs values used with --obs-key. Comma-separated values are also accepted.",
    ),
    obs_values_path: Path | None = typer.Option(
        None,
        "--obs-values",
        exists=True,
        dir_okay=False,
        help="Optional file listing obs values to keep when --obs-key is set.",
    ),
    output_gene_column: str | None = typer.Option(
        None,
        "--output-gene-column",
        help="Optional h5ad var column used to write alternate gene identifiers instead of var_names.",
    ),
) -> int:
    input_path = input_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    restrict_genes_path = resolve_optional_path(restrict_genes_path)
    obs_values_path = resolve_optional_path(obs_values_path)
    method_resolved = normalize_rank_method(resolve_str(method))
    max_cells = resolve_optional_int(max_cells)
    top_n = resolve_optional_int(top_n)
    random_seed = resolve_int(random_seed)
    hvg_flavor_resolved = normalize_hvg_flavor(resolve_str(hvg_flavor))
    prior_source_resolved = normalize_prior_source(resolve_str(prior_source))
    label = resolve_optional_str(label)
    obs_key = resolve_optional_str(obs_key)
    output_gene_column = resolve_optional_str(output_gene_column)
    resolved_obs_values = _resolve_obs_values(obs_values, obs_values_path)

    restrict_genes = (
        None if restrict_genes_path is None else read_gene_list(restrict_genes_path)
    )
    result = compute_ranking(
        input_path,
        method=method_resolved,
        hvg_flavor=hvg_flavor_resolved,
        prior_source=prior_source_resolved,
        label=label,
        obs_key=obs_key,
        obs_values=resolved_obs_values,
        max_cells=max_cells,
        random_seed=random_seed,
    )
    gene_names, scores, filter_metadata = filter_gene_scores(
        result.gene_names, result.scores, restrict_genes=restrict_genes
    )
    _, ranked_gene_names, _ = rank_gene_scores(
        gene_names, scores, descending=result.descending
    )
    if top_n is not None:
        ranked_gene_names = ranked_gene_names[:top_n]
    output_gene_names = _map_output_gene_names(
        input_path,
        gene_names=ranked_gene_names,
        output_gene_column=output_gene_column,
    )
    write_gene_list(output_path, output_gene_names)

    print_gene_summary(
        "Rank Genes",
        Input=input_path,
        Method=method_resolved,
        GenesOut=len(output_gene_names),
        OutputGeneColumn=output_gene_column,
        ObsKey=obs_key,
        ObsValues=None if resolved_obs_values is None else len(resolved_obs_values),
        TopN=top_n,
        MissingRestrictGenes=filter_metadata["n_missing_restrict_genes"],
        Output=output_path,
    )
    print_saved_path(console, output_path)
    return 0


__all__ = [
    "SUPPORTED_HVG_FLAVORS",
    "SUPPORTED_PRIOR_SOURCES",
    "SUPPORTED_RANK_METHODS",
    "rank_genes_command",
]
