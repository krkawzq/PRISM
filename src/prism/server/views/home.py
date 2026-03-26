from __future__ import annotations

from html import escape
from typing import cast
from urllib.parse import quote_plus

from prism.server.services.analysis import GeneBrowsePage
from prism.server.services.datasets import GeneCandidate
from prism.server.services.global_eval import GlobalEvalParams, GlobalEvaluationResult

from .layout import render_loader, render_nav, render_page, stat_card


def render_home_page(
    *,
    dataset_summary: dict[str, object] | None,
    gene_browser: GeneBrowsePage | None,
    search_query: str = "",
    h5ad_path: str = "",
    ckpt_path: str = "",
    layer: str = "",
    error_message: str | None = None,
    global_eval: GlobalEvaluationResult | None = None,
    global_eval_params: GlobalEvalParams | None = None,
    global_eval_figures: dict[str, str] | None = None,
) -> str:
    loader = render_loader(h5ad_path=h5ad_path, ckpt_path=ckpt_path, layer=layer)
    if dataset_summary is None:
        message = (
            f'<section class="panel"><h2>Load error</h2><p>{escape(error_message)}</p></section>'
            if error_message
            else '<section class="panel"><h2>Welcome</h2><p>Load an h5ad file and optionally a PRISM checkpoint to browse genes, reuse fitted priors, and compare against baselines.</p></section>'
        )
        body = f"{render_nav(current_query=search_query)}{loader}{message}"
        return render_page(title="PRISM Analysis Server", body=body)

    n_cells = int(cast(int, dataset_summary["n_cells"]))
    n_genes = int(cast(int, dataset_summary["n_genes"]))
    fitted_genes = int(cast(int, dataset_summary["fitted_genes"]))
    s_hat = float(cast(float, dataset_summary["s_hat"]))
    median_total = float(cast(float, dataset_summary["median_total"]))
    stats = "".join(
        [
            stat_card("Cells", f"{n_cells:,}"),
            stat_card("Genes", f"{n_genes:,}"),
            stat_card("Fitted genes", f"{fitted_genes:,}"),
            stat_card("Layer", str(dataset_summary["layer"])),
            stat_card("Model source", str(dataset_summary["model_source"])),
            stat_card("s_hat", f"{s_hat:.3f}"),
            stat_card("Median total", f"{median_total:.1f}"),
        ]
    )
    browser_block = ""
    if gene_browser is not None:
        browser_block = _render_gene_browser_block(gene_browser)
    global_eval_block = _render_global_eval_block(
        global_eval,
        has_checkpoint=bool(dataset_summary["ckpt_path"]),
        params=global_eval_params or GlobalEvalParams(),
        figures=global_eval_figures or {},
    )
    body = f"""
      {render_nav(current_query=search_query)}
      {loader}
      <section class="panel">
        <h2>Dataset snapshot</h2>
        <div class="stat-grid">{stats}</div>
        <div class="meta-block">
          <div><strong>h5ad</strong><span>{escape(str(dataset_summary["h5ad_path"]))}</span></div>
          <div><strong>checkpoint</strong><span>{escape(str(dataset_summary["ckpt_path"]))}</span></div>
        </div>
      </section>
      {global_eval_block}
      {browser_block}
    """
    return render_page(title="PRISM Analysis Server", body=body)


def _render_gene_browser_block(page: GeneBrowsePage) -> str:
    scope_options = [
        ("auto", "Auto scope"),
        ("fitted", "Fitted only"),
        ("all", "All genes"),
    ]
    sort_options = [
        ("total_umi", "Total UMI"),
        ("detected_cells", "Detected cells"),
        ("detected_fraction", "Detected fraction"),
        ("gene_name", "Gene name"),
        ("gene_index", "Gene index"),
    ]
    dir_options = [("desc", "Descending"), ("asc", "Ascending")]
    scope_html = "".join(
        f'<option value="{value}"{" selected" if value == page.scope else ""}>{label}</option>'
        for value, label in scope_options
    )
    sort_html = "".join(
        f'<option value="{value}"{" selected" if value == page.sort_by else ""}>{label}</option>'
        for value, label in sort_options
    )
    dir_html = "".join(
        f'<option value="{value}"{" selected" if ((page.descending and value == "desc") or (not page.descending and value == "asc")) else ""}>{label}</option>'
        for value, label in dir_options
    )
    rows = "".join(_candidate_row(candidate) for candidate in page.items)
    if not rows:
        rows = '<tr><td colspan="5" class="muted">No genes match the current filter.</td></tr>'
    page_start = 0 if page.total_items == 0 else (page.page - 1) * page.page_size + 1
    page_end = min(page.page * page.page_size, page.total_items)
    prev_page = max(page.page - 1, 1)
    next_page = min(page.page + 1, page.total_pages)
    pager_links = "".join(
        _pager_button(
            target, page.page, page.query, page.sort_by, page.descending, page.scope
        )
        for target in _pager_window(page.page, page.total_pages)
    )
    return f"""
      <section class="panel">
        <div class="browser-header">
          <div>
            <h2>Gene browser</h2>
            <p class="muted">Server-rendered gene table with reliable pagination, sorting, filtering, and direct page jump.</p>
          </div>
          <div class="browser-meta">
            <span class="fit-badge">scope: {escape(page.scope)}</span>
            <span class="fit-badge">page size: {page.page_size}</span>
            <span class="fit-badge">genes: {page.total_items}</span>
          </div>
        </div>
        <form class="browser-toolbar" action="/" method="get">
          <input type="text" name="browse_q" value="{escape(page.query)}" placeholder="Filter genes by substring">
          <select name="browse_scope">{scope_html}</select>
          <select name="browse_sort">{sort_html}</select>
          <select name="browse_dir">{dir_html}</select>
          <input type="number" name="browse_page" min="1" max="{page.total_pages}" step="1" value="{page.page}">
          <button type="submit">Apply</button>
        </form>
        <div class="browser-summary">
          <span class="browser-page-label">Showing {page_start}-{page_end} / {page.total_items}</span>
          <span class="browser-page-label">Page {page.page} / {page.total_pages}</span>
        </div>
        <div class="table-wrap browser-table-wrap"><table class="browser-table">
          <thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th><th>Detected cells</th><th>Detected frac</th></tr></thead>
          <tbody>{rows}</tbody>
        </table></div>
        <div class="browser-pager-row">
          <form class="browser-inline-pager" action="/" method="get">
            <input type="hidden" name="browse_q" value="{escape(page.query)}">
            <input type="hidden" name="browse_scope" value="{escape(page.scope)}">
            <input type="hidden" name="browse_sort" value="{escape(page.sort_by)}">
            <input type="hidden" name="browse_dir" value="{"desc" if page.descending else "asc"}">
            <button type="submit" name="browse_page" value="{prev_page}"{" disabled" if page.page <= 1 else ""}>Previous</button>
            {pager_links}
            <button type="submit" name="browse_page" value="{next_page}"{" disabled" if page.page >= page.total_pages else ""}>Next</button>
          </form>
          <form class="browser-jump-form" action="/" method="get">
            <input type="hidden" name="browse_q" value="{escape(page.query)}">
            <input type="hidden" name="browse_scope" value="{escape(page.scope)}">
            <input type="hidden" name="browse_sort" value="{escape(page.sort_by)}">
            <input type="hidden" name="browse_dir" value="{"desc" if page.descending else "asc"}">
            <label>Go to page <input type="number" name="browse_page" min="1" max="{page.total_pages}" step="1" value="{page.page}"></label>
            <button type="submit">Jump</button>
          </form>
        </div>
      </section>
    """


def _pager_window(current: int, total: int, radius: int = 2) -> list[int]:
    start = max(1, current - radius)
    end = min(total, current + radius)
    if start == 1:
        end = min(total, max(end, 1 + radius * 2))
    if end == total:
        start = max(1, min(start, total - radius * 2))
    return list(range(start, end + 1))


def _pager_button(
    target: int,
    current: int,
    query: str,
    sort_by: str,
    descending: bool,
    scope: str,
) -> str:
    active = " browser-page-current" if target == current else ""
    return (
        '<a class="browser-page-btn'
        + active
        + f'" href="/?browse_q={quote_plus(query)}&browse_scope={quote_plus(scope)}&browse_sort={quote_plus(sort_by)}&browse_dir={"desc" if descending else "asc"}&browse_page={target}">{target}</a>'
    )


def _render_global_eval_block(
    result: GlobalEvaluationResult | None,
    *,
    has_checkpoint: bool,
    params: GlobalEvalParams,
    figures: dict[str, str],
) -> str:
    if not has_checkpoint:
        return '<section class="panel"><h2>Global evaluation</h2><p>Load a checkpoint to enable on-demand global metrics.</p></section>'

    controls = f"""
    <form class="toolbar" action="/" method="get">
      <input type="hidden" name="global_eval" value="1">
      <input type="number" name="ge_max_cells" min="0" step="1" value="{params.max_cells}" placeholder="max cells (0=all)">
      <input type="number" name="ge_max_genes" min="16" step="1" value="{params.max_genes}" placeholder="max genes">
      <input type="number" name="ge_batch" min="8" step="1" value="{params.gene_batch_size}" placeholder="gene batch">
      <input type="number" name="ge_seed" min="0" step="1" value="{params.random_seed}" placeholder="seed">
      <button type="submit">Compute global metrics</button>
    </form>
    """
    if result is None:
        return f'<section class="panel"><h2>Global evaluation</h2><p>Compute checkpoint-backed global metrics on demand for X, NormalizeTotalX, Log1pNormalizeTotalX, and signal. You can subsample cells here to keep runtime under control.</p>{controls}</section>'

    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics.silhouette:.4f}</td><td>{metrics.ari:.4f}</td><td>{metrics.nmi:.4f}</td><td>{metrics.pca_var_ratio:.4f}</td><td>{metrics.neighborhood_consistency:.4f}</td><td>{'-' if metrics.mean_treatment_cv is None else f'{metrics.mean_treatment_cv:.4f}'}</td></tr>"
        for name, metrics in result.representation_metrics.items()
    )
    meta = f'<p class="muted">Label source: <strong>{escape(result.label_key)}</strong> | Cells: {result.n_cells:,} | Genes: {result.n_genes:,} | Labels: {result.n_labels}</p>'
    table = f'<div class="table-wrap"><table class="compact-table"><thead><tr><th>Representation</th><th>Silhouette</th><th>ARI</th><th>NMI</th><th>PCA var@10</th><th>Neighbor consistency</th><th>Mean treatment CV</th></tr></thead><tbody>{rows}</tbody></table></div>'
    fg = result.fg_summary
    overlap_rows = "".join(
        f"<tr><td>{k}</td><td>{vals['trad_entropy']:.3f}</td><td>{vals['trad_structure']:.3f}</td></tr>"
        for k, vals in sorted(result.hvg_overlap.items())
    )
    distribution_rows = "".join(
        [
            f"<tr><th>Entropy mean</th><td>{fg.entropy_mean:.4f}</td><th>Entropy p95</th><td>{fg.entropy_p95:.4f}</td></tr>",
            f"<tr><th>Peak count mean</th><td>{fg.peak_count_mean:.3f}</td><th>Inflection mean</th><td>{fg.inflection_count_mean:.3f}</td></tr>",
            f"<tr><th>Sharpness mean</th><td>{fg.sharpness_mean:.4f}</td><th>Low/High expr entropy</th><td>{fg.low_expression_entropy_mean:.4f} / {fg.high_expression_entropy_mean:.4f}</td></tr>",
        ]
    )
    relation_rows = "".join(
        [
            f"<tr><th>Entropy~Expr rho</th><td>{fg.entropy_expression_spearman:.4f}</td><th>Entropy~Sharp rho</th><td>{fg.entropy_sharpness_spearman:.4f}</td></tr>",
            f"<tr><th>Entropy~Peak rho</th><td>{fg.entropy_peak_spearman:.4f}</td><th>Entropy~Infl rho</th><td>{fg.entropy_inflection_spearman:.4f}</td></tr>",
        ]
    )
    consistency_rows = "".join(
        [
            f"<tr><th>Trad vs Entropy rho</th><td>{result.hvg_spearman['trad_vs_entropy']:.4f}</td><th>Trad vs Structure rho</th><td>{result.hvg_spearman['trad_vs_structure']:.4f}</td></tr>",
            f"<tr><th>Entropy vs HVG rho</th><td>{fg.hvg_spearman:.4f}</td><th>Structure vs HVG rho</th><td>{fg.structure_hvg_spearman:.4f}</td></tr>",
            f"<tr><th>Overlap@100 / 500</th><td>{fg.overlap_at_100:.3f} / {fg.overlap_at_500:.3f}</td><th>Struct@100 / 500</th><td>{fg.structure_overlap_at_100:.3f} / {fg.structure_overlap_at_500:.3f}</td></tr>",
        ]
    )
    overlap_table = f'<div class="table-wrap"><table><thead><tr><th>Top-k</th><th>Trad vs Entropy</th><th>Trad vs Structure</th></tr></thead><tbody>{overlap_rows}</tbody></table></div>'
    plot_block = ""
    if figures.get("overlap"):
        plot_block = f'<section class="panel figure-wide"><h3>Overlap curve</h3><div class="figure figure-hero"><img src="{figures["overlap"]}"></div></section>'
    fg_block = f"""<section class="panel"><h3>F_g analysis</h3>
      <div class="fg-section-grid">
        <div class="table-wrap summary-table"><table><thead><tr><th colspan="4">Distribution</th></tr></thead><tbody>{distribution_rows}</tbody></table></div>
        <div class="table-wrap summary-table"><table><thead><tr><th colspan="4">Relations</th></tr></thead><tbody>{relation_rows}</tbody></table></div>
        <div class="table-wrap summary-table"><table><thead><tr><th colspan="4">HVG consistency</th></tr></thead><tbody>{consistency_rows}</tbody></table></div>
      </div>
      {overlap_table}
    </section>{plot_block}"""
    return f'<section class="panel"><h2>Global evaluation</h2>{meta}{controls}{table}</section>{fg_block}'


def _candidate_row(candidate: GeneCandidate) -> str:
    href = f"/gene?q={quote_plus(candidate.gene_name)}"
    return (
        f'<tr><td><a href="{href}">{escape(candidate.gene_name)}</a></td>'
        f"<td>{candidate.gene_index}</td>"
        f"<td>{candidate.total_umi:,}</td>"
        f"<td>{candidate.detected_cells:,}</td>"
        f"<td>{candidate.detected_fraction:.3f}</td></tr>"
    )
