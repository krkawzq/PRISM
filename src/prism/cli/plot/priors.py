from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import cast

import anndata as ad
import typer

from prism.cli.common import (
    console,
    print_elapsed,
    print_key_value_table,
    print_saved_path,
    resolve_bool,
    resolve_float,
    resolve_optional_float,
    resolve_optional_int,
    resolve_optional_path,
    resolve_optional_str,
    resolve_str,
    unwrap_typer_value,
)
from prism.cli.genes.common import (
    PRIOR_ENTROPY_METHODS,
    compute_ranking,
    normalize_rank_method,
)
from prism.io import read_string_list
from prism.model import ModelCheckpoint, load_checkpoint
from prism.plotting import (
    SUPPORTED_CURVE_MODES,
    SUPPORTED_STAT_FIELDS,
    SUPPORTED_Y_SCALES,
    curve_sets_summary_dataframe,
    curve_sets_to_dataframe,
    default_checkpoint_name,
    load_annotation_tables,
    plot_prior_facet_figure,
    plot_prior_overlay_figure,
    plt,
    resolve_curve_mode,
    resolve_multi_checkpoint_prior_curve_sets,
    resolve_stat_fields,
    resolve_x_axis,
    resolve_y_scale,
)

from .common import (
    normalize_layout,
    resolve_gene_names,
    resolve_label_names,
    resolve_optional_list,
)

SUPPORTED_H5AD_RANK_METHODS = tuple(
    method
    for method in (
        "hvg",
        "signal-hvg",
        "lognorm-variance",
        "lognorm-dispersion",
        "signal-variance",
        "signal-dispersion",
    )
)


def _resolve_annotation_names(
    value: list[str] | None | object,
) -> list[str] | None:
    resolved = resolve_optional_list(value)
    if resolved is None:
        return None
    return list(resolved)


def _resolve_checkpoint_name_values(
    value: list[str] | None | object,
) -> list[str] | None:
    resolved = unwrap_typer_value(value)
    if resolved is None:
        return None
    names: list[str] = []
    for item in cast(list[str], resolved):
        parts = [part.strip() for part in str(item).split(",")]
        names.extend(part for part in parts if part)
    return names


def _resolve_repeated_values(
    value: list[str] | None | object,
    *,
    values_path: Path | None = None,
) -> list[str] | None:
    resolved = unwrap_typer_value(value)
    values: list[str] = []
    seen: set[str] = set()
    if resolved is not None:
        for item in cast(list[str], resolved):
            for part in str(item).split(","):
                label = part.strip()
                if not label or label in seen:
                    continue
                seen.add(label)
                values.append(label)
    if values_path is not None:
        for item in read_string_list(values_path):
            label = str(item).strip()
            if not label or label in seen:
                continue
            seen.add(label)
            values.append(label)
    return values or None


def _resolve_shared_source_h5ad(
    checkpoint_specs: list[tuple[str, Path, ModelCheckpoint]],
) -> Path | None:
    candidates: list[Path] = []
    for _, _, checkpoint in checkpoint_specs:
        source_path = checkpoint.metadata.get("source_h5ad_path")
        if source_path is None:
            return None
        candidate = Path(str(source_path)).expanduser().resolve()
        if not candidate.exists():
            return None
        candidates.append(candidate)
    if not candidates:
        return None
    first = candidates[0]
    return first if all(path == first for path in candidates[1:]) else None


def _normalize_rank_method_for_h5ad(value: str) -> str:
    resolved = normalize_rank_method(value)
    if resolved in PRIOR_ENTROPY_METHODS:
        raise ValueError(
            "--rank-method must be an h5ad-based ranking method when used with --rank-h5ad"
        )
    return resolved


def _load_var_column_mapping(
    h5ad_path: Path,
    *,
    gene_id_column: str,
) -> tuple[dict[str, str], set[str], dict[str, str]]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if gene_id_column not in adata.var.columns:
            raise KeyError(
                f"var column {gene_id_column!r} does not exist in {h5ad_path}"
            )
        gene_ids = [str(name) for name in adata.var_names.tolist()]
        aliases = [str(value).strip() for value in adata.var[gene_id_column].tolist()]
    finally:
        if getattr(adata, "isbacked", False):
            adata.file.close()
    alias_to_gene: dict[str, str] = {}
    ambiguous: set[str] = set()
    gene_to_alias: dict[str, str] = {}
    for gene_id, alias in zip(gene_ids, aliases, strict=True):
        if alias:
            existing = alias_to_gene.get(alias)
            if existing is not None and existing != gene_id:
                ambiguous.add(alias)
            else:
                alias_to_gene[alias] = gene_id
        gene_to_alias[gene_id] = alias or gene_id
    return alias_to_gene, ambiguous, gene_to_alias


def _resolve_gene_annotation_h5ad(
    *,
    explicit_h5ad_path: Path | None,
    checkpoint_specs: list[tuple[str, Path, ModelCheckpoint]],
    fallback_h5ad_path: Path | None = None,
) -> Path | None:
    if explicit_h5ad_path is not None:
        return explicit_h5ad_path
    if fallback_h5ad_path is not None:
        return fallback_h5ad_path
    return _resolve_shared_source_h5ad(checkpoint_specs)


def _resolve_ranked_gene_names(
    *,
    h5ad_path: Path,
    rank_method: str,
    rank_obs_key: str | None,
    rank_obs_values: list[str] | None,
    rank_top_n: int,
    rank_max_cells: int | None,
    rank_seed: int,
) -> tuple[list[str], dict[str, object]]:
    result = compute_ranking(
        h5ad_path,
        method=rank_method,
        hvg_flavor="seurat_v3",
        prior_source="global",
        label=None,
        obs_key=rank_obs_key,
        obs_values=rank_obs_values,
        max_cells=rank_max_cells,
        random_seed=rank_seed,
    )
    order = (
        result.scores.argsort()[::-1] if result.descending else result.scores.argsort()
    )
    selected = [str(gene) for gene in result.gene_names[order][:rank_top_n].tolist()]
    return selected, {
        "Rank source": h5ad_path,
        "Rank method": rank_method,
        "Rank genes": len(selected),
        "Rank obs key": "-" if rank_obs_key is None else rank_obs_key,
        "Rank obs values": "-"
        if not rank_obs_values
        else ", ".join(rank_obs_values[:6])
        + (", ..." if len(rank_obs_values) > 6 else ""),
    }


def _map_requested_gene_names(
    requested_gene_names: list[str],
    *,
    annotation_h5ad_path: Path | None,
    gene_id_column: str | None,
    available_gene_names: set[str],
) -> tuple[list[str], dict[str, str] | None]:
    if gene_id_column is None:
        return requested_gene_names, None
    if annotation_h5ad_path is None:
        raise ValueError(
            "--gene-id-column requires --gene-annotations-h5ad or checkpoint metadata with a shared source_h5ad_path"
        )
    alias_to_gene, ambiguous_aliases, gene_to_alias = _load_var_column_mapping(
        annotation_h5ad_path, gene_id_column=gene_id_column
    )
    resolved: list[str] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for gene_name in requested_gene_names:
        if gene_name in available_gene_names:
            resolved.append(gene_name)
            continue
        if gene_name in ambiguous_aliases:
            ambiguous.append(gene_name)
            continue
        mapped = alias_to_gene.get(gene_name)
        if mapped is None:
            missing.append(gene_name)
            continue
        resolved.append(mapped)
    if ambiguous:
        raise ValueError(
            f"gene labels are ambiguous in var column {gene_id_column!r}: {ambiguous[:10]}"
        )
    if missing:
        raise ValueError(
            f"gene labels were not found directly or in var column {gene_id_column!r}: {missing[:10]}"
        )
    return resolved, gene_to_alias


def _merge_annotation_tables(
    annotation_tables: dict[str, dict[tuple[str, str], str]] | None,
    curve_sets: dict[str, list[object]],
    *,
    gene_to_alias: dict[str, str] | None,
    annotation_name: str | None,
) -> dict[str, dict[tuple[str, str], str]] | None:
    if gene_to_alias is None or annotation_name is None:
        return annotation_tables
    merged = {} if annotation_tables is None else dict(annotation_tables)
    if annotation_name in merged:
        raise ValueError(
            f"annotation table name {annotation_name!r} is already used; choose a different --annot-name or omit --gene-id-column"
        )
    table: dict[tuple[str, str], str] = {}
    for gene_name, curves in curve_sets.items():
        label = gene_to_alias.get(gene_name, gene_name)
        for curve in curves:
            source = str(getattr(curve, "source"))
            table[(gene_name, source)] = label
    merged[annotation_name] = table
    return merged


def _normalize_missing_policy(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"error", "drop"}:
        raise ValueError("missing_policy must be one of: error, drop")
    return resolved


def _resolve_checkpoint_specs(
    checkpoint_paths: list[Path],
    checkpoint_name_values: list[str] | None,
) -> list[tuple[str, Path, ModelCheckpoint]]:
    if checkpoint_name_values is not None and len(checkpoint_name_values) != len(
        checkpoint_paths
    ):
        raise ValueError("--checkpoint-name count must match checkpoint paths")
    resolved_paths = [path.expanduser().resolve() for path in checkpoint_paths]
    checkpoints = [(path, load_checkpoint(path)) for path in resolved_paths]
    explicit_names = checkpoint_name_values is not None
    names = (
        checkpoint_name_values
        if checkpoint_name_values is not None
        else [
            default_checkpoint_name(checkpoint, checkpoint_path=path)
            for path, checkpoint in checkpoints
        ]
    )
    if any(not name for name in names):
        raise ValueError("checkpoint names cannot be blank")
    if explicit_names and len(names) != len(set(names)):
        raise ValueError("--checkpoint-name values must be unique")
    if not explicit_names:
        seen: dict[str, int] = {}
        deduped_names: list[str] = []
        for name in names:
            next_index = seen.get(name, 0) + 1
            seen[name] = next_index
            deduped_names.append(name if next_index == 1 else f"{name}#{next_index}")
        names = deduped_names
    return [
        (name, path, checkpoint)
        for name, (path, checkpoint) in zip(names, checkpoints, strict=True)
    ]


def _resolve_x_axis_auto(curve_sets: dict[str, list[object]], *, x_axis: str) -> str:
    if x_axis.strip().lower() != "auto":
        return resolve_x_axis(x_axis)
    support_domains = {
        str(getattr(curve, "support_domain"))
        for curves in curve_sets.values()
        for curve in curves
    }
    if len(support_domains) != 1:
        raise ValueError(
            "x_axis='auto' requires all selected curves to share the same support domain"
        )
    support_domain = next(iter(support_domains))
    return "rate" if support_domain == "rate" else "scaled"


def plot_priors_command(
    checkpoint_paths: list[Path] = typer.Argument(
        ..., exists=True, dir_okay=False, help="One or more checkpoint paths."
    ),
    checkpoint_names: list[str] | None = typer.Option(
        None,
        "--checkpoint-name",
        help="Optional repeatable display names aligned with checkpoint paths.",
    ),
    gene_names: list[str] | None = typer.Option(
        None, "--gene", help="Repeatable gene name to plot."
    ),
    genes_path: Path | None = typer.Option(
        None,
        "--genes",
        "--gene-list",
        exists=True,
        dir_okay=False,
        help="Optional gene list file. Uses the first --top-n genes when provided.",
    ),
    top_n: int | None = typer.Option(
        None,
        "--top-n",
        min=1,
        help="Number of leading genes to use with --genes, or the default count for --rank-h5ad gene selection.",
    ),
    rank_h5ad_path: Path | None = typer.Option(
        None,
        "--rank-h5ad",
        exists=True,
        dir_okay=False,
        help="Optional h5ad used to rank genes when --gene/--genes is omitted. Defaults to the shared checkpoint source_h5ad_path when available.",
    ),
    rank_method: str = typer.Option(
        "lognorm-variance",
        "--rank-method",
        help="Gene ranking method for --rank-h5ad: "
        + ", ".join(SUPPORTED_H5AD_RANK_METHODS)
        + ".",
    ),
    rank_obs_key: str | None = typer.Option(
        None,
        "--rank-obs-key",
        help="Optional obs column used to filter cells before ranking.",
    ),
    rank_obs_values: list[str] | None = typer.Option(
        None,
        "--rank-obs-value",
        help="Repeatable obs values used with --rank-obs-key. Comma-separated values are also accepted.",
    ),
    rank_obs_values_path: Path | None = typer.Option(
        None,
        "--rank-obs-values",
        exists=True,
        dir_okay=False,
        help="Optional file listing obs values used with --rank-obs-key.",
    ),
    rank_top_n: int | None = typer.Option(
        None,
        "--rank-top-n",
        min=1,
        help="Number of top-ranked genes to plot when --rank-h5ad is used. Defaults to --top-n when provided, otherwise 20.",
    ),
    rank_max_cells: int | None = typer.Option(
        None,
        "--rank-max-cells",
        min=1,
        help="Optional maximum number of cells to use while ranking genes from --rank-h5ad.",
    ),
    rank_seed: int = typer.Option(
        0,
        "--rank-seed",
        min=0,
        help="Random seed used for ranking-time cell subsampling.",
    ),
    gene_annotations_h5ad: Path | None = typer.Option(
        None,
        "--gene-annotations-h5ad",
        exists=True,
        dir_okay=False,
        help="Optional h5ad used to map requested gene labels through a var column.",
    ),
    gene_id_column: str | None = typer.Option(
        None,
        "--gene-id-column",
        help="Optional h5ad var column used to map requested gene symbols to checkpoint gene ids and annotate facet plots.",
    ),
    output_path: Path = typer.Option(..., "--output", "-o", help="Output figure path."),
    output_csv_path: Path | None = typer.Option(
        None,
        "--output-csv",
        help="Optional CSV path for exported curve coordinates.",
    ),
    summary_csv_path: Path | None = typer.Option(
        None,
        "--summary-csv",
        help="Optional CSV path for per-curve summary statistics.",
    ),
    annot_csv_paths: list[Path] | None = typer.Option(
        None,
        "--annot-csv",
        exists=True,
        dir_okay=False,
        help="Optional repeatable CSV files with columns gene,label,... for facet annotations.",
    ),
    annot_names: list[str] | None = typer.Option(
        None,
        "--annot-name",
        help="Optional repeatable names for --annot-csv, aligned by position.",
    ),
    labels: list[str] | None = typer.Option(
        None,
        "--label",
        help="Optional repeatable label priors to include. Defaults to all label priors.",
    ),
    labels_path: Path | None = typer.Option(
        None,
        "--labels",
        "--label-list",
        exists=True,
        dir_okay=False,
        help="Optional file listing labels to include.",
    ),
    x_axis: str = typer.Option(
        "auto",
        help="x axis: auto, scaled, support, or rate. auto chooses rate for rate-domain priors, otherwise scaled.",
    ),
    curve_mode: str = typer.Option(
        "density",
        help="Curve mode: " + ", ".join(SUPPORTED_CURVE_MODES) + ".",
    ),
    y_scale: str = typer.Option(
        "linear",
        help="y axis scale: " + ", ".join(SUPPORTED_Y_SCALES) + ".",
    ),
    mass_quantile: float = typer.Option(
        0.995,
        min=0.5,
        max=1.0,
        help="Upper cumulative mass used to truncate the displayed axis.",
    ),
    include_global: bool = typer.Option(
        True,
        "--include-global/--no-include-global",
        help="Include the global prior when present.",
    ),
    layout: str = typer.Option(
        "overlay",
        help="Plot layout: overlay or facet.",
    ),
    show_subplot_labels: bool = typer.Option(
        False,
        "--show-subplot-labels/--no-show-subplot-labels",
        help="Render gene/source labels inside every facet subplot.",
    ),
    show_legend: bool = typer.Option(
        True,
        "--show-legend/--no-show-legend",
        help="Show legend in overlay layout.",
    ),
    missing_policy: str = typer.Option(
        "error",
        help="How to handle missing genes or labels across checkpoints: error or drop.",
    ),
    stat_fields: list[str] | None = typer.Option(
        None,
        "--stat",
        help="Repeatable per-curve stats to annotate: "
        + ", ".join(SUPPORTED_STAT_FIELDS)
        + ".",
    ),
    panel_width: float | None = typer.Option(
        None, min=0.0, help="Optional panel width override."
    ),
    panel_height: float | None = typer.Option(
        None, min=0.0, help="Optional panel height override."
    ),
) -> int:
    start_time = perf_counter()
    genes_path = resolve_optional_path(genes_path)
    top_n = resolve_optional_int(top_n)
    rank_h5ad_path = resolve_optional_path(rank_h5ad_path)
    rank_obs_values_path = resolve_optional_path(rank_obs_values_path)
    rank_top_n = resolve_optional_int(rank_top_n)
    rank_max_cells = resolve_optional_int(rank_max_cells)
    gene_annotations_h5ad = resolve_optional_path(gene_annotations_h5ad)
    output_path = output_path.expanduser().resolve()
    output_csv_path = resolve_optional_path(output_csv_path)
    summary_csv_path = resolve_optional_path(summary_csv_path)
    labels_path = resolve_optional_path(labels_path)
    x_axis = resolve_str(x_axis)
    curve_mode = resolve_str(curve_mode)
    y_scale = resolve_str(y_scale)
    layout = resolve_str(layout)
    rank_method = _normalize_rank_method_for_h5ad(resolve_str(rank_method))
    rank_obs_key = resolve_optional_str(rank_obs_key)
    rank_seed = int(rank_seed)
    gene_id_column = resolve_optional_str(gene_id_column)
    annot_csv_resolved = (
        None
        if unwrap_typer_value(annot_csv_paths) is None
        else [
            Path(path).expanduser().resolve()
            for path in unwrap_typer_value(annot_csv_paths)
        ]
    )
    annot_names_resolved = _resolve_annotation_names(annot_names)
    mass_quantile = resolve_float(mass_quantile)
    include_global = resolve_bool(include_global)
    show_subplot_labels = resolve_bool(show_subplot_labels)
    show_legend = resolve_bool(show_legend)
    missing_policy = _normalize_missing_policy(resolve_str(missing_policy))
    panel_width = resolve_optional_float(panel_width)
    panel_height = resolve_optional_float(panel_height)
    resolved_rank_obs_values = _resolve_repeated_values(
        rank_obs_values, values_path=rank_obs_values_path
    )

    resolved_layout = normalize_layout(layout)
    resolved_labels = resolve_label_names(
        labels=resolve_optional_list(labels),
        labels_path=labels_path,
    )
    resolved_curve_mode = resolve_curve_mode(curve_mode)
    resolved_y_scale = resolve_y_scale(y_scale)
    resolved_stat_fields = resolve_stat_fields(resolve_optional_list(stat_fields))

    checkpoint_specs = _resolve_checkpoint_specs(
        checkpoint_paths,
        _resolve_checkpoint_name_values(checkpoint_names),
    )
    explicit_gene_selection = resolve_optional_list(gene_names) is not None or genes_path is not None
    if explicit_gene_selection and rank_h5ad_path is not None:
        raise ValueError("--rank-h5ad cannot be combined with explicit --gene/--genes selection")
    shared_source_h5ad = _resolve_shared_source_h5ad(checkpoint_specs)
    effective_rank_h5ad = rank_h5ad_path
    rank_summary: dict[str, object] | None = None
    if explicit_gene_selection:
        requested_gene_names = resolve_gene_names(
            gene_names=resolve_optional_list(gene_names),
            genes_path=genes_path,
            top_n=top_n,
        )
    else:
        if effective_rank_h5ad is None:
            effective_rank_h5ad = shared_source_h5ad
            if effective_rank_h5ad is not None:
                console.print(
                    f"[cyan]Using checkpoint source_h5ad_path for ranking:[/cyan] {effective_rank_h5ad}"
                )
        if effective_rank_h5ad is None:
            raise ValueError(
                "provide --gene/--genes or --rank-h5ad; no shared source_h5ad_path metadata was found"
            )
        effective_rank_top_n = rank_top_n if rank_top_n is not None else (top_n or 20)
        requested_gene_names, rank_summary = _resolve_ranked_gene_names(
            h5ad_path=effective_rank_h5ad,
            rank_method=rank_method,
            rank_obs_key=rank_obs_key,
            rank_obs_values=resolved_rank_obs_values,
            rank_top_n=effective_rank_top_n,
            rank_max_cells=rank_max_cells,
            rank_seed=rank_seed,
        )
    available_gene_names = {
        gene_name
        for _, _, checkpoint in checkpoint_specs
        for gene_name in checkpoint.gene_names
    }
    annotation_h5ad_path = _resolve_gene_annotation_h5ad(
        explicit_h5ad_path=gene_annotations_h5ad,
        checkpoint_specs=checkpoint_specs,
        fallback_h5ad_path=effective_rank_h5ad if gene_id_column is not None else None,
    )
    resolved_genes, gene_to_alias = _map_requested_gene_names(
        requested_gene_names,
        annotation_h5ad_path=annotation_h5ad_path,
        gene_id_column=gene_id_column,
        available_gene_names=available_gene_names,
    )
    if rank_summary is not None:
        print_key_value_table(console, title="Ranked Gene Selection", values=rank_summary)
    if annot_csv_resolved and resolved_layout != "facet":
        console.print("[yellow]Using facet layout because --annot-csv requires facets[/yellow]")
        resolved_layout = "facet"
    if resolved_layout == "overlay" and len(resolved_genes) > 1:
        console.print(
            "[yellow]Overlaying multiple genes in a single figure; use --layout facet for per-gene panels[/yellow]"
        )
    annotation_tables = None
    if annot_csv_resolved:
        annotation_tables = load_annotation_tables(annot_csv_resolved, annot_names_resolved)
    curve_sets = resolve_multi_checkpoint_prior_curve_sets(
        checkpoint_specs,
        gene_names=resolved_genes,
        labels=resolved_labels,
        include_global=include_global,
        missing_policy=missing_policy,
    )
    if resolved_layout == "facet":
        annotation_tables = _merge_annotation_tables(
            annotation_tables,
            curve_sets,
            gene_to_alias=gene_to_alias,
            annotation_name=gene_id_column,
        )
    resolved_x_axis = _resolve_x_axis_auto(curve_sets, x_axis=x_axis)
    overlay_panel_width = (
        8.2 if panel_width is None or panel_width <= 0 else panel_width
    )
    overlay_panel_height = (
        4.8 if panel_height is None or panel_height <= 0 else panel_height
    )
    facet_panel_width = 4.6 if panel_width is None or panel_width <= 0 else panel_width
    facet_panel_height = (
        3.2 if panel_height is None or panel_height <= 0 else panel_height
    )

    fig = (
        plot_prior_facet_figure(
            curve_sets,
            x_axis=resolved_x_axis,
            mass_quantile=mass_quantile,
            show_subplot_labels=show_subplot_labels,
            annotation_tables=annotation_tables,
            curve_mode=resolved_curve_mode,
            y_scale=resolved_y_scale,
            stats_fields=resolved_stat_fields,
            panel_width=facet_panel_width,
            panel_height=facet_panel_height,
        )
        if resolved_layout == "facet"
        else plot_prior_overlay_figure(
            curve_sets,
            x_axis=resolved_x_axis,
            mass_quantile=mass_quantile,
            curve_mode=resolved_curve_mode,
            y_scale=resolved_y_scale,
            show_legend=show_legend,
            stats_fields=resolved_stat_fields,
            panel_width=overlay_panel_width,
            panel_height=overlay_panel_height,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print_saved_path(console, output_path)

    if output_csv_path is not None:
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        curve_sets_to_dataframe(curve_sets, x_axis=resolved_x_axis).to_csv(
            output_csv_path, index=False
        )
        print_saved_path(console, output_csv_path)
    if summary_csv_path is not None:
        summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
        curve_sets_summary_dataframe(curve_sets).to_csv(summary_csv_path, index=False)
        print_saved_path(console, summary_csv_path)
    print_elapsed(console, perf_counter() - start_time)
    return 0


__all__ = ["plot_priors_command"]
