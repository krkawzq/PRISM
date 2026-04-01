from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from prism.io import read_string_list
from prism.model import ModelCheckpoint, PriorGrid

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_CYCLE = ["#1d4ed8", "#0f766e", "#c2410c", "#7c3aed", "#be123c", "#0891b2"]
SUPPORTED_CURVE_MODES = ("density", "cdf")
SUPPORTED_Y_SCALES = ("linear", "log")
SUPPORTED_STAT_FIELDS = ("mean_p", "mean_mu", "map_p", "map_mu", "entropy", "S")
SUPPORTED_OVERLAP_METRICS = ("overlap", "jsd", "wasserstein", "best_scale")


@dataclass(frozen=True, slots=True)
class PriorCurve:
    source: str
    label_name: str | None
    grid_domain: str
    p_values: np.ndarray
    mu_values: np.ndarray
    weights: np.ndarray
    S: float


@dataclass(frozen=True, slots=True)
class LabelGridEntry:
    label: str
    batch: str
    perturbation: str


def load_string_list_file(path: Path) -> list[str]:
    return read_string_list(path)


def _format_annotation_value(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value_f = float(value)
        if np.isnan(value_f):
            return "NA"
        return f"{value_f:.2f}"
    text = str(value)
    try:
        return f"{float(text):.2f}"
    except ValueError:
        return text


def load_annotation_tables(
    csv_paths: list[Path], annot_names: list[str] | None
) -> dict[str, dict[tuple[str, str], str]]:
    if annot_names and len(annot_names) != len(csv_paths):
        raise ValueError("--annot-name count must match --annot-csv count")
    tables: dict[str, dict[tuple[str, str], str]] = {}
    for idx, path in enumerate(csv_paths):
        name = path.stem if not annot_names else annot_names[idx]
        df = pd.read_csv(path)
        if "gene" not in df.columns:
            raise ValueError(f"annotation CSV missing required 'gene' column: {path}")
        label_col = "label" if "label" in df.columns else None
        value_cols = [col for col in df.columns if col not in {"gene", "label"}]
        if not value_cols:
            raise ValueError(f"annotation CSV has no value columns: {path}")
        mapping: dict[tuple[str, str], str] = {}
        for _, row in df.iterrows():
            gene = str(row["gene"])
            raw_label = None if label_col is None else row[label_col]
            label_text = "" if raw_label is None else str(raw_label).strip()
            label_is_na = False
            if raw_label is not None:
                na_value = pd.isna(raw_label)
                label_is_na = (
                    bool(na_value.item())
                    if hasattr(na_value, "item")
                    else bool(na_value)
                )
            if (
                raw_label is None
                or label_text in {"", "NA", "NaN", "nan"}
                or label_is_na
            ):
                source = "global"
            else:
                source = f"label:{label_text}"
            text = ", ".join(
                f"{col}={_format_annotation_value(row[col])}" for col in value_cols
            )
            mapping[(gene, source)] = text
        tables[name] = mapping
    return tables


def resolve_x_axis(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in {"mu", "p", "rate"}:
        raise ValueError("x_axis must be one of: mu, p, rate")
    return resolved


def resolve_curve_mode(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in SUPPORTED_CURVE_MODES:
        raise ValueError(
            "curve_mode must be one of: " + ", ".join(SUPPORTED_CURVE_MODES)
        )
    return resolved


def resolve_y_scale(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in SUPPORTED_Y_SCALES:
        raise ValueError("y_scale must be one of: " + ", ".join(SUPPORTED_Y_SCALES))
    return resolved


def resolve_stat_fields(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    resolved = _dedupe_names(values)
    unknown = [value for value in resolved if value not in SUPPORTED_STAT_FIELDS]
    if unknown:
        raise ValueError(
            "unknown stats fields: "
            + ", ".join(unknown)
            + "; supported: "
            + ", ".join(SUPPORTED_STAT_FIELDS)
        )
    return tuple(resolved)


def resolve_overlap_metric(value: str) -> str:
    resolved = value.strip().lower()
    if resolved not in SUPPORTED_OVERLAP_METRICS:
        raise ValueError(
            "metric must be one of: " + ", ".join(SUPPORTED_OVERLAP_METRICS)
        )
    return resolved


def _dedupe_names(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        stripped = value.strip()
        if not stripped or stripped in seen:
            continue
        seen.add(stripped)
        ordered.append(stripped)
    return ordered


def _curve_from_prior(priors: PriorGrid, gene_name: str, *, source: str) -> PriorCurve:
    prior = priors.subset(gene_name)
    return PriorCurve(
        source=source,
        label_name=None if source == "global" else source.removeprefix("label:"),
        grid_domain=prior.grid_domain,
        p_values=(
            np.asarray(prior.p_grid, dtype=np.float64).reshape(-1)
            if prior.grid_domain == "p"
            else np.full(
                np.asarray(prior.p_grid).reshape(-1).shape, np.nan, dtype=np.float64
            )
        ),
        mu_values=np.asarray(prior.mu_grid, dtype=np.float64).reshape(-1),
        weights=np.asarray(prior.weights, dtype=np.float64).reshape(-1),
        S=float(prior.S),
    )


def resolve_prior_curve_sets(
    checkpoint: ModelCheckpoint,
    *,
    gene_names: list[str],
    labels: list[str] | None,
    include_global: bool,
) -> dict[str, list[PriorCurve]]:
    requested = _dedupe_names(list(gene_names))
    if not requested:
        raise ValueError("at least one gene is required")
    selected_labels = (
        sorted(checkpoint.label_priors) if not labels else _dedupe_names(list(labels))
    )
    unknown = [
        label for label in selected_labels if label not in checkpoint.label_priors
    ]
    if unknown:
        available = sorted(checkpoint.label_priors)
        raise ValueError(
            f"unknown label priors: {unknown}; available examples: {available[:10]}"
        )
    curve_sets: dict[str, list[PriorCurve]] = {}
    for gene_name in requested:
        curves: list[PriorCurve] = []
        if (
            include_global
            and checkpoint.priors is not None
            and gene_name in checkpoint.priors.gene_names
        ):
            curves.append(
                _curve_from_prior(checkpoint.priors, gene_name, source="global")
            )
        for label in selected_labels:
            priors = checkpoint.label_priors[label]
            if gene_name not in priors.gene_names:
                continue
            curves.append(_curve_from_prior(priors, gene_name, source=f"label:{label}"))
        if not curves:
            raise ValueError(
                f"gene {gene_name!r} is not present in the selected prior sources"
            )
        curve_sets[gene_name] = curves
    return curve_sets


def parse_batch_grid_entries(labels: list[str]) -> list[LabelGridEntry]:
    entries: list[LabelGridEntry] = []
    for label in _dedupe_names(labels):
        if "_" not in label:
            raise ValueError(
                f"could not infer batch/perturbation from label {label!r}; "
                "provide --label-grid-csv with columns label,batch,perturbation"
            )
        batch, perturbation = label.split("_", 1)
        if not batch or not perturbation:
            raise ValueError(f"invalid batch-grid label: {label!r}")
        entries.append(
            LabelGridEntry(label=label, batch=batch, perturbation=perturbation)
        )
    if not entries:
        raise ValueError("no batch-grid labels were resolved")
    return entries


def load_label_grid_entries(path: Path) -> list[LabelGridEntry]:
    df = pd.read_csv(path)
    required = {"label", "batch", "perturbation"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"label-grid CSV is missing required columns {missing}: {path}"
        )
    entries: list[LabelGridEntry] = []
    for _, row in df.iterrows():
        entries.append(
            LabelGridEntry(
                label=str(row["label"]).strip(),
                batch=str(row["batch"]).strip(),
                perturbation=str(row["perturbation"]).strip(),
            )
        )
    if not entries:
        raise ValueError(f"label-grid CSV is empty: {path}")
    return entries


def resolve_batch_grid_curve_sets(
    checkpoint: ModelCheckpoint,
    *,
    gene_names: list[str],
    entries: list[LabelGridEntry],
) -> tuple[dict[str, dict[tuple[str, str], PriorCurve]], list[str], list[str]]:
    if not checkpoint.label_priors:
        raise ValueError("checkpoint has no label priors")
    requested_genes = _dedupe_names(list(gene_names))
    if not requested_genes:
        raise ValueError("at least one gene is required")
    if not entries:
        raise ValueError("at least one label-grid entry is required")
    unknown = [
        entry.label for entry in entries if entry.label not in checkpoint.label_priors
    ]
    if unknown:
        raise ValueError(f"unknown label priors: {sorted(set(unknown))[:10]}")
    batches: list[str] = []
    perturbations: list[str] = []
    for entry in entries:
        if entry.batch not in batches:
            batches.append(entry.batch)
        if entry.perturbation not in perturbations:
            perturbations.append(entry.perturbation)
    curve_sets: dict[str, dict[tuple[str, str], PriorCurve]] = {}
    for gene_name in requested_genes:
        curve_map: dict[tuple[str, str], PriorCurve] = {}
        for entry in entries:
            priors = checkpoint.label_priors[entry.label]
            if gene_name not in priors.gene_names:
                continue
            curve_map[(entry.batch, entry.perturbation)] = _curve_from_prior(
                priors,
                gene_name,
                source=f"label:{entry.label}",
            )
        curve_sets[gene_name] = curve_map
    return curve_sets, batches, perturbations


def resolve_x_values(curve: PriorCurve, *, x_axis: str) -> np.ndarray:
    if x_axis == "p":
        if curve.grid_domain != "p":
            raise ValueError("x_axis='p' is only valid for p-grid priors")
        return curve.p_values
    return curve.mu_values


def resolve_curve_y_values(curve: PriorCurve, *, curve_mode: str) -> np.ndarray:
    if curve_mode == "density":
        return np.asarray(curve.weights, dtype=np.float64)
    if curve_mode == "cdf":
        weights = np.asarray(curve.weights, dtype=np.float64)
        total = max(float(np.sum(weights)), 1e-12)
        return np.cumsum(weights / total)
    raise ValueError(f"unsupported curve_mode: {curve_mode}")


def summarize_prior_curve(curve: PriorCurve) -> dict[str, float]:
    p_values = np.asarray(curve.p_values, dtype=np.float64)
    mu_values = np.asarray(curve.mu_values, dtype=np.float64)
    weights = np.asarray(curve.weights, dtype=np.float64)
    total = max(float(np.sum(weights)), 1e-12)
    normalized = weights / total
    map_idx = int(np.argmax(normalized))
    entropy = float(-np.sum(normalized * np.log(np.clip(normalized, 1e-12, None))))
    return {
        "mean_p": float(np.sum(p_values * normalized))
        if curve.grid_domain == "p"
        else float("nan"),
        "mean_mu": float(np.sum(mu_values * normalized)),
        "map_p": float(p_values[map_idx]) if curve.grid_domain == "p" else float("nan"),
        "map_mu": float(mu_values[map_idx]),
        "entropy": entropy,
        "S": float(curve.S),
    }


def format_curve_stats(curve: PriorCurve, *, fields: tuple[str, ...]) -> str:
    if not fields:
        return ""
    summary = summarize_prior_curve(curve)
    return ", ".join(f"{field}={summary[field]:.3g}" for field in fields)


def _display_cutoff(grid: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    grid_np = np.asarray(grid, dtype=np.float64).reshape(-1)
    weight_np = np.asarray(weights, dtype=np.float64).reshape(-1)
    if grid_np.size == 0:
        return 1.0
    mass = max(float(np.sum(weight_np)), 1e-12)
    cdf = np.cumsum(weight_np / mass)
    idx = int(np.searchsorted(cdf, quantile, side="left"))
    idx = min(max(idx, 0), grid_np.size - 1)
    return float(grid_np[idx])


def _curve_y_for_plot(
    curve: PriorCurve, *, curve_mode: str, y_scale: str
) -> np.ndarray:
    values = resolve_curve_y_values(curve, curve_mode=curve_mode)
    if y_scale == "log":
        return np.clip(values, 1e-12, None)
    return values


def plot_prior_overlay_figure(
    curve_sets: dict[str, list[PriorCurve]],
    *,
    x_axis: str,
    mass_quantile: float,
    curve_mode: str = "density",
    y_scale: str = "linear",
    show_legend: bool = True,
    stats_fields: tuple[str, ...] = (),
    panel_width: float = 8.2,
    panel_height: float = 4.8,
):
    genes = list(curve_sets)
    n_genes = len(genes)
    cols = min(2, n_genes)
    rows = (n_genes + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(panel_width * cols, panel_height * rows), squeeze=False
    )
    for ax in axes.ravel():
        ax.set_visible(False)
    for panel_idx, gene_name in enumerate(genes):
        ax = axes.ravel()[panel_idx]
        ax.set_visible(True)
        curves = curve_sets[gene_name]
        display_max = max(
            _display_cutoff(
                resolve_x_values(curve, x_axis=x_axis),
                curve.weights,
                mass_quantile,
            )
            for curve in curves
        )
        display_max = max(display_max, 1e-12)
        for curve_idx, curve in enumerate(curves):
            x_values = resolve_x_values(curve, x_axis=x_axis)
            y_values = _curve_y_for_plot(
                curve,
                curve_mode=curve_mode,
                y_scale=y_scale,
            )
            mask = x_values <= display_max + 1e-12
            if not np.any(mask):
                mask = np.ones_like(x_values, dtype=bool)
            ax.plot(
                x_values[mask],
                y_values[mask],
                lw=2.0,
                color=COLOR_CYCLE[curve_idx % len(COLOR_CYCLE)],
                label=curve.source,
            )
        ax.set_title(gene_name)
        ax.set_xlabel(x_axis)
        ax.set_ylabel("prior mass" if curve_mode == "density" else "prior CDF")
        ax.set_yscale(y_scale)
        if show_legend:
            ax.legend(frameon=False)
        if stats_fields:
            lines = [
                f"{curve.source}: {format_curve_stats(curve, fields=stats_fields)}"
                for curve in curves
            ]
            ax.text(
                0.02,
                0.98,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.8,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
            )
        ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def plot_prior_facet_figure(
    curve_sets: dict[str, list[PriorCurve]],
    *,
    x_axis: str,
    mass_quantile: float,
    show_subplot_labels: bool = False,
    annotation_tables: dict[str, dict[tuple[str, str], str]] | None = None,
    curve_mode: str = "density",
    y_scale: str = "linear",
    stats_fields: tuple[str, ...] = (),
    panel_width: float = 4.6,
    panel_height: float = 3.2,
):
    genes = list(curve_sets)
    if not genes:
        raise ValueError("no genes available for plotting")
    column_labels: list[str] = []
    seen: set[str] = set()
    for curves in curve_sets.values():
        for curve in curves:
            if curve.source not in seen:
                seen.add(curve.source)
                column_labels.append(curve.source)
    n_rows = len(genes)
    n_cols = len(column_labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(panel_width * n_cols, panel_height * n_rows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    for row_idx, gene_name in enumerate(genes):
        curve_map = {curve.source: curve for curve in curve_sets[gene_name]}
        row_display_max = max(
            _display_cutoff(
                resolve_x_values(curve, x_axis=x_axis),
                curve.weights,
                mass_quantile,
            )
            for curve in curve_map.values()
        )
        row_display_max = max(row_display_max, 1e-12)
        for col_idx, column_label in enumerate(column_labels):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(column_label)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(f"{gene_name}\nprior mass" if col_idx == 0 else "prior mass")
            curve = curve_map.get(column_label)
            if curve is None:
                ax.set_xlim(0.0, row_display_max)
                ax.set_visible(False)
                continue
            x_values = resolve_x_values(curve, x_axis=x_axis)
            y_values = _curve_y_for_plot(
                curve,
                curve_mode=curve_mode,
                y_scale=y_scale,
            )
            mask = x_values <= row_display_max + 1e-12
            if not np.any(mask):
                mask = np.ones_like(x_values, dtype=bool)
            ax.plot(
                x_values[mask],
                y_values[mask],
                lw=2.0,
                color=COLOR_CYCLE[col_idx % len(COLOR_CYCLE)],
            )
            ax.set_xlim(0.0, row_display_max)
            ax.set_yscale(y_scale)
            ax.grid(alpha=0.18)
            if show_subplot_labels:
                ax.text(
                    0.98,
                    0.98,
                    f"{gene_name}\n{column_label}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.75,
                        "edgecolor": "none",
                        "pad": 1.5,
                    },
                )
            if stats_fields:
                ax.text(
                    0.02,
                    0.02,
                    format_curve_stats(curve, fields=stats_fields),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "none",
                        "pad": 1.0,
                    },
                )
            if annotation_tables:
                lines = [
                    f"{table_name}: {mapping[(gene_name, column_label)]}"
                    for table_name, mapping in annotation_tables.items()
                    if (gene_name, column_label) in mapping
                ]
                if lines:
                    ax.text(
                        0.5,
                        -0.24,
                        "\n".join(lines),
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        fontsize=7,
                    )
    fig.tight_layout()
    return fig


def plot_batch_grid_figure(
    gene_name: str,
    curve_map: dict[tuple[str, str], PriorCurve],
    *,
    batches: list[str],
    perturbations: list[str],
    x_axis: str,
    mass_quantile: float,
    curve_mode: str = "density",
    y_scale: str = "linear",
    stats_fields: tuple[str, ...] = (),
    hide_empty: bool = False,
    show_axis_ticks: bool = True,
    panel_width: float = 2.2,
    panel_height: float = 1.9,
):
    if not curve_map:
        raise ValueError("curve_map cannot be empty")
    row_display_max = max(
        _display_cutoff(
            resolve_x_values(curve, x_axis=x_axis),
            curve.weights,
            mass_quantile,
        )
        for curve in curve_map.values()
    )
    row_display_max = max(row_display_max, 1e-12)
    fig, axes = plt.subplots(
        len(batches),
        len(perturbations),
        figsize=(panel_width * len(perturbations), panel_height * len(batches)),
        squeeze=False,
    )
    for row_idx, batch in enumerate(batches):
        for col_idx, perturbation in enumerate(perturbations):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(perturbation, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(batch, fontsize=8)
            key = (batch, perturbation)
            curve = curve_map.get(key)
            if curve is None:
                ax.set_xlim(0.0, row_display_max)
                if hide_empty:
                    ax.set_visible(False)
                    continue
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            x_values = resolve_x_values(curve, x_axis=x_axis)
            y_values = _curve_y_for_plot(
                curve,
                curve_mode=curve_mode,
                y_scale=y_scale,
            )
            mask = x_values <= row_display_max + 1e-12
            if not np.any(mask):
                mask = np.ones_like(x_values, dtype=bool)
            ax.plot(x_values[mask], y_values[mask], lw=1.3, color="#1d4ed8")
            ax.set_xlim(0.0, row_display_max)
            ax.set_yscale(y_scale)
            if show_axis_ticks:
                ax.tick_params(axis="both", labelsize=6)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if stats_fields:
                ax.text(
                    0.02,
                    0.02,
                    format_curve_stats(curve, fields=stats_fields),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=6,
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.8,
                        "edgecolor": "none",
                        "pad": 0.8,
                    },
                )
            ax.grid(alpha=0.16)
    fig.suptitle(gene_name, fontsize=12)
    fig.tight_layout()
    return fig


def curve_sets_to_dataframe(
    curve_sets: dict[str, list[PriorCurve]],
    *,
    x_axis: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curves in curve_sets.items():
        for curve in curves:
            x_values = resolve_x_values(curve, x_axis=x_axis)
            for idx, (p_value, mu_value, x_value, weight_value) in enumerate(
                zip(
                    curve.p_values,
                    curve.mu_values,
                    x_values,
                    curve.weights,
                    strict=True,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "gene": gene_name,
                        "source": curve.source,
                        "label": curve.label_name,
                        "point_index": idx,
                        "p": float(p_value),
                        "mu": float(mu_value),
                        "x": float(x_value),
                        "weight": float(weight_value),
                        "x_axis": x_axis,
                        "S": float(curve.S),
                    }
                )
    return pd.DataFrame(rows)


def curve_sets_summary_dataframe(
    curve_sets: dict[str, list[PriorCurve]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curves in curve_sets.items():
        for curve in curves:
            rows.append(
                {
                    "gene": gene_name,
                    "source": curve.source,
                    "label": curve.label_name,
                    **summarize_prior_curve(curve),
                }
            )
    return pd.DataFrame(rows)


def batch_grid_curve_sets_to_dataframe(
    curve_sets: dict[str, dict[tuple[str, str], PriorCurve]],
    *,
    x_axis: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curve_map in curve_sets.items():
        for (batch, perturbation), curve in curve_map.items():
            x_values = resolve_x_values(curve, x_axis=x_axis)
            for idx, (p_value, mu_value, x_value, weight_value) in enumerate(
                zip(
                    curve.p_values,
                    curve.mu_values,
                    x_values,
                    curve.weights,
                    strict=True,
                ),
                start=1,
            ):
                rows.append(
                    {
                        "gene": gene_name,
                        "source": curve.source,
                        "label": curve.label_name,
                        "batch": batch,
                        "perturbation": perturbation,
                        "point_index": idx,
                        "p": float(p_value),
                        "mu": float(mu_value),
                        "x": float(x_value),
                        "weight": float(weight_value),
                        "x_axis": x_axis,
                        "S": float(curve.S),
                    }
                )
    return pd.DataFrame(rows)


def batch_grid_summary_dataframe(
    curve_sets: dict[str, dict[tuple[str, str], PriorCurve]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for gene_name, curve_map in curve_sets.items():
        for (batch, perturbation), curve in curve_map.items():
            rows.append(
                {
                    "gene": gene_name,
                    "source": curve.source,
                    "label": curve.label_name,
                    "batch": batch,
                    "perturbation": perturbation,
                    **summarize_prior_curve(curve),
                }
            )
    return pd.DataFrame(rows)


def _prepare_density(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid_min: float,
    grid_max: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    support = np.linspace(grid_min, grid_max, n_points, dtype=np.float64)
    values = np.interp(support, x, y, left=0.0, right=0.0)
    values = np.clip(values, 0.0, None)
    area = float(np.trapezoid(values, support))
    if area <= 0:
        raise ValueError("density area must be positive")
    values /= area
    return support, values


def _distribution_metrics(
    support: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> tuple[float, float, float]:
    dx = float(support[1] - support[0]) if support.size > 1 else 1.0
    overlap = float(np.sum(np.minimum(p, q)) * dx)
    overlap = min(max(overlap, 0.0), 1.0)
    p_mass = p / max(float(np.sum(p)), 1e-12)
    q_mass = q / max(float(np.sum(q)), 1e-12)
    m_mass = 0.5 * (p_mass + q_mass)
    eps = 1e-12
    jsd = 0.5 * float(np.sum(p_mass * np.log((p_mass + eps) / (m_mass + eps))))
    jsd += 0.5 * float(np.sum(q_mass * np.log((q_mass + eps) / (m_mass + eps))))
    cdf_diff = np.cumsum(p_mass - q_mass)
    wasserstein = float(np.sum(np.abs(cdf_diff)) * dx)
    return overlap, jsd, wasserstein


def _best_scaled_metrics(
    ctrl_x: np.ndarray,
    ctrl_y: np.ndarray,
    pert_x: np.ndarray,
    pert_y: np.ndarray,
    *,
    scale_min: float,
    scale_max: float,
    scale_grid_size: int,
    interp_points: int,
) -> tuple[float, float, float, float]:
    best_scale = 1.0
    best_overlap = -1.0
    best_jsd = float("inf")
    best_wasserstein = float("inf")
    scale_grid = np.exp(
        np.linspace(
            np.log(scale_min), np.log(scale_max), scale_grid_size, dtype=np.float64
        )
    )
    for scale in scale_grid:
        scaled_x = pert_x * scale
        grid_min = float(min(np.min(ctrl_x), np.min(scaled_x)))
        grid_max = float(max(np.max(ctrl_x), np.max(scaled_x)))
        support, ctrl_density = _prepare_density(
            ctrl_x,
            ctrl_y,
            grid_min=grid_min,
            grid_max=grid_max,
            n_points=interp_points,
        )
        _, pert_density = _prepare_density(
            scaled_x,
            pert_y,
            grid_min=grid_min,
            grid_max=grid_max,
            n_points=interp_points,
        )
        overlap, jsd, wasserstein = _distribution_metrics(
            support,
            ctrl_density,
            pert_density,
        )
        if overlap > best_overlap:
            best_scale = float(scale)
            best_overlap = overlap
            best_jsd = jsd
            best_wasserstein = wasserstein
    return best_scale, best_overlap, best_jsd, best_wasserstein


def compute_overlap_dataframe(
    checkpoint: ModelCheckpoint,
    *,
    control_label: str,
    gene_names: list[str],
    labels: list[str],
    scale_min: float,
    scale_max: float,
    scale_grid_size: int,
    interp_points: int,
) -> pd.DataFrame:
    if scale_min > scale_max:
        raise ValueError("--scale-min must be <= --scale-max")
    if not checkpoint.label_priors:
        raise ValueError("checkpoint has no label priors")
    if control_label not in checkpoint.label_priors:
        raise ValueError(
            f"control label {control_label!r} not found in checkpoint label priors"
        )
    control_priors = checkpoint.label_priors[control_label]
    rows: list[dict[str, object]] = []
    for label in labels:
        if label == control_label:
            continue
        if label not in checkpoint.label_priors:
            raise ValueError(f"unknown label prior: {label!r}")
        priors = checkpoint.label_priors[label]
        common_genes = [
            gene
            for gene in gene_names
            if gene in control_priors.gene_names and gene in priors.gene_names
        ]
        for gene in common_genes:
            ctrl_prior = control_priors.subset(gene)
            pert_prior = priors.subset(gene)
            if ctrl_prior.grid_domain != pert_prior.grid_domain:
                raise ValueError(
                    f"cannot compare priors with different grid_domain for gene {gene!r}: "
                    f"{ctrl_prior.grid_domain!r} vs {pert_prior.grid_domain!r}"
                )
            ctrl_x = np.asarray(ctrl_prior.mu_grid, dtype=np.float64).reshape(-1)
            ctrl_y = np.asarray(ctrl_prior.weights, dtype=np.float64).reshape(-1)
            pert_x = np.asarray(pert_prior.mu_grid, dtype=np.float64).reshape(-1)
            pert_y = np.asarray(pert_prior.weights, dtype=np.float64).reshape(-1)
            best_scale, overlap, jsd, wasserstein = _best_scaled_metrics(
                ctrl_x,
                ctrl_y,
                pert_x,
                pert_y,
                scale_min=scale_min,
                scale_max=scale_max,
                scale_grid_size=scale_grid_size,
                interp_points=interp_points,
            )
            rows.append(
                {
                    "gene": gene,
                    "label": label,
                    "overlap": overlap,
                    "jsd": jsd,
                    "wasserstein": wasserstein,
                    "best_scale": best_scale,
                }
            )
    if not rows:
        raise ValueError("no overlap rows were produced")
    return pd.DataFrame(rows).sort_values(["label", "gene"]).reset_index(drop=True)


def _default_overlap_cmap(metric: str) -> str:
    if metric == "overlap":
        return "viridis"
    if metric == "best_scale":
        return "coolwarm"
    if metric == "jsd":
        return "viridis_r"
    return "magma_r"


def plot_overlap_heatmap_figure(
    df: pd.DataFrame,
    *,
    metric: str,
    gene_order: list[str],
    label_order: list[str],
    annotate_cells: bool = False,
    panel_width: float = 0.45,
    panel_height: float = 0.35,
    cmap: str | None = None,
):
    pivot = df.pivot(index="gene", columns="label", values=metric)
    matrix = pivot.reindex(index=gene_order, columns=label_order).to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(
            max(4.0, panel_width * max(len(label_order), 1)),
            max(3.2, panel_height * max(len(gene_order), 1)),
        )
    )
    norm = None
    if metric == "best_scale":
        finite_values = matrix[np.isfinite(matrix)]
        if finite_values.size > 0:
            vmin = float(np.min(finite_values))
            vmax = float(np.max(finite_values))
            if vmin < 1.0 < vmax and vmin < vmax:
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=_default_overlap_cmap(metric) if cmap is None else cmap,
        norm=norm,
    )
    ax.set_xticks(np.arange(len(label_order)))
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(gene_order)))
    ax.set_yticklabels(gene_order)
    ax.set_xlabel("label")
    ax.set_ylabel("gene")
    ax.set_title(metric)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label(metric)
    if annotate_cells:
        for row_idx, gene in enumerate(gene_order):
            for col_idx, label in enumerate(label_order):
                value = pivot.get(label, pd.Series(dtype=float)).get(gene)
                if pd.isna(value):
                    continue
                ax.text(
                    col_idx,
                    row_idx,
                    f"{float(value):.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if metric != "overlap" else "black",
                )
    fig.tight_layout()
    return fig
