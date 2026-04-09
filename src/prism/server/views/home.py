from __future__ import annotations

from html import escape
from typing import cast
from urllib.parse import quote_plus

from prism.server.services.analysis import CheckpointSummary, GeneBrowsePage

from .components import render_chip_row, render_detail_grid, render_section_header
from .layout import render_loader, render_message, render_nav, render_page, stat_card


def render_home_page(
    *,
    dataset_summary: dict[str, object] | None,
    checkpoint_summary: CheckpointSummary | None,
    gene_browser: GeneBrowsePage | None,
    search_query: str = "",
    h5ad_path: str = "",
    ckpt_path: str = "",
    layer: str = "",
    error_message: str | None = None,
) -> str:
    body_parts = [
        render_nav(current_query=search_query),
        render_loader(h5ad_path=h5ad_path, ckpt_path=ckpt_path, layer=layer),
    ]
    if error_message:
        body_parts.append(render_message(error_message, level="error"))
    if dataset_summary is None:
        body_parts.append(_render_welcome_panel())
        return render_page(title="PRISM Server", body="".join(body_parts))

    body_parts.append(_render_dataset_snapshot(dataset_summary))
    body_parts.append(_render_checkpoint_panel(checkpoint_summary))
    if gene_browser is not None:
        body_parts.append(_render_gene_browser(gene_browser))
    return render_page(title="PRISM Server", body="".join(body_parts))


def _render_welcome_panel() -> str:
    section_header = render_section_header(
        "Welcome",
        "Use this local analysis workspace to load a dataset, inspect checkpoint coverage, and move directly into gene-level diagnostics.",
        eyebrow="Overview",
    )
    return f"""
    <section class="panel">
      {section_header}
      <div class="feature-grid">
        <article class="feature"><h3>Load Once</h3><p>Bring a dataset and optional checkpoint into one shared analysis context.</p></article>
        <article class="feature"><h3>Browse Fast</h3><p>Filter genes by name, sort by expression, and jump straight into a target gene page.</p></article>
        <article class="feature"><h3>Inspect Deeply</h3><p>Compare raw counts, posterior summaries, on-demand fits, and kBulk group behavior in one place.</p></article>
      </div>
    </section>
    """


def _render_dataset_snapshot(dataset_summary: dict[str, object]) -> str:
    label_keys = cast_tuple_str(dataset_summary.get("label_keys"))
    n_cells = _coerce_int(dataset_summary.get("n_cells"))
    n_genes = _coerce_int(dataset_summary.get("n_genes"))
    total_count_mean = _coerce_float(dataset_summary.get("total_count_mean"))
    total_count_median = _coerce_float(dataset_summary.get("total_count_median"))
    total_count_p99 = _coerce_float(dataset_summary.get("total_count_p99"))
    stats = [
        stat_card("Cells", f"{n_cells:,}"),
        stat_card("Genes", f"{n_genes:,}"),
        stat_card("Layer", str(dataset_summary["layer"])),
        stat_card("Mean total count", f"{total_count_mean:.3f}"),
        stat_card("Median total count", f"{total_count_median:.3f}"),
        stat_card("P99 total count", f"{total_count_p99:.3f}"),
        stat_card("Label keys", ", ".join(label_keys) if label_keys else "-"),
    ]
    chips = render_chip_row(
        [(key, "info") for key in label_keys[:6]]
        or [("No label keys detected", "neutral")]
    )
    detail_grid = render_detail_grid(
        [
            ("Dataset path", str(dataset_summary["h5ad_path"])),
            ("Active layer", str(dataset_summary["layer"])),
            ("Label key count", str(len(label_keys))),
        ]
    )
    section_header = render_section_header(
        "Dataset Snapshot",
        "Current dataset context, expression depth summary, and detected label coverage.",
        eyebrow="Dataset",
    )
    return f"""
    <section class="panel">
      {section_header}
      {chips}
      <div class="stat-grid">{"".join(stats)}</div>
      {detail_grid}
    </section>
    """


def _render_checkpoint_panel(checkpoint_summary: CheckpointSummary | None) -> str:
    section_header = render_section_header(
        "Checkpoint Summary" if checkpoint_summary is not None else "Checkpoint",
        "Checkpoint-backed analysis metadata and coverage for the active dataset."
        if checkpoint_summary is not None
        else "No checkpoint is loaded. Raw gene inspection and on-demand fit still work.",
        eyebrow="Model",
    )
    if checkpoint_summary is None:
        return f'<section class="panel">{section_header}</section>'
    preview = (
        ", ".join(checkpoint_summary.label_preview)
        if checkpoint_summary.label_preview
        else "-"
    )
    chips = render_chip_row(
        [
            (
                "Global prior ready"
                if checkpoint_summary.has_global_prior
                else "Global prior missing",
                "success" if checkpoint_summary.has_global_prior else "warning",
            ),
            (f"{checkpoint_summary.n_label_priors:,} label priors", "info"),
            (checkpoint_summary.distribution, "neutral"),
        ]
    )
    stats = [
        stat_card("Genes", f"{checkpoint_summary.gene_count:,}"),
        stat_card(
            "Global prior", "yes" if checkpoint_summary.has_global_prior else "no"
        ),
        stat_card("Label priors", f"{checkpoint_summary.n_label_priors:,}"),
        stat_card("Distribution", checkpoint_summary.distribution),
        stat_card("Support", checkpoint_summary.support_domain or "-"),
        stat_card(
            "Scale",
            "-"
            if checkpoint_summary.scale is None
            else f"{checkpoint_summary.scale:.3f}",
        ),
        stat_card(
            "Mean reference",
            "-"
            if checkpoint_summary.mean_reference_count is None
            else f"{checkpoint_summary.mean_reference_count:.3f}",
        ),
        stat_card(
            "Reference overlap",
            f"{checkpoint_summary.n_overlap_reference_genes:,} / {checkpoint_summary.n_reference_genes:,}",
        ),
        stat_card("Suggested label key", checkpoint_summary.suggested_label_key or "-"),
    ]
    details = render_detail_grid(
        [
            ("Checkpoint path", checkpoint_summary.ckpt_path),
            ("Label preview", preview),
        ]
    )
    return f"""
    <section class="panel">
      {section_header}
      {chips}
      <div class="stat-grid">{"".join(stats)}</div>
      {details}
    </section>
    """


def _render_gene_browser(page: GeneBrowsePage) -> str:
    scope_options = {
        "auto": "Auto",
        "fitted": "Checkpoint genes",
        "all": "All genes",
    }
    sort_options = {
        "total_count": "Total count",
        "detected_cells": "Detected cells",
        "detected_fraction": "Detected fraction",
        "gene_name": "Gene name",
        "gene_index": "Gene index",
    }
    dir_value = "desc" if page.descending else "asc"
    rows = "".join(
        f'<tr><td><a class="table-link" href="/gene?q={quote_plus(item.gene_name)}">{escape(item.gene_name)}</a></td><td>{item.gene_index}</td><td>{item.total_count:,}</td><td>{item.detected_cells:,}</td><td>{item.detected_fraction:.3f}</td></tr>'
        for item in page.items
    )
    if not rows:
        rows = '<tr><td colspan="5" class="muted">No genes matched the current filter.</td></tr>'
    section_header = render_section_header(
        "Gene Browser",
        "Filter and rank genes by abundance, detection, or identity before opening a gene workspace.",
        eyebrow="Explore",
    )
    return f"""
    <section class="panel">
      {section_header}
      <form class="toolbar browser" action="/" method="get">
        <label class="field field-grow"><span>Substring search</span><input type="text" name="browse_q" value="{escape(page.query)}" placeholder="substring search"></label>
        <label class="field"><span>Scope</span><select name="browse_scope">
          {"".join(f'<option value="{value}"{" selected" if page.scope == value else ""}>{label}</option>' for value, label in scope_options.items())}
        </select></label>
        <label class="field"><span>Sort by</span><select name="browse_sort">
          {"".join(f'<option value="{value}"{" selected" if page.sort_by == value else ""}>{label}</option>' for value, label in sort_options.items())}
        </select></label>
        <label class="field"><span>Direction</span><select name="browse_dir">
          <option value="desc"{" selected" if dir_value == "desc" else ""}>Descending</option>
          <option value="asc"{" selected" if dir_value == "asc" else ""}>Ascending</option>
        </select></label>
        <label class="field"><span>Page</span><input type="number" name="browse_page" min="1" value="{page.page}"></label>
        <div class="form-actions form-actions-inline"><button type="submit">Apply</button></div>
      </form>
      <p class="muted">Showing page {page.page} / {page.total_pages}, {page.total_items:,} genes total.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Gene</th><th>Index</th><th>Total</th><th>Detected</th><th>Detected frac</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </section>
    """


def cast_tuple_str(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        return ()
    return tuple(str(item) for item in value)


def _coerce_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)):
        return int(cast(int | float | str, value))
    return 0


def _coerce_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, str)):
        return float(cast(int | float | str, value))
    return 0.0
