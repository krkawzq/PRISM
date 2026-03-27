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
            else '<section class="panel"><h2>Welcome</h2><p>Load an h5ad file and optionally a PRISM checkpoint to browse genes, inspect posteriors, and run quick on-demand fits.</p></section>'
        )
        return render_page(
            title="PRISM Analysis Server",
            body=f"{render_nav(current_query=search_query)}{loader}{message}",
        )

    n_cells = int(cast(int, dataset_summary["n_cells"]))
    n_genes = int(cast(int, dataset_summary["n_genes"]))
    fitted_genes = int(cast(int, dataset_summary["fitted_genes"]))
    s_value = dataset_summary["S"]
    s_source = dataset_summary["S_source"]
    mean_reference_count = dataset_summary["mean_reference_count"]
    stats = [
        stat_card("Cells", f"{n_cells:,}"),
        stat_card("Genes", f"{n_genes:,}"),
        stat_card("Fitted genes", f"{fitted_genes:,}"),
        stat_card("Layer", str(dataset_summary["layer"])),
        stat_card("Model source", str(dataset_summary["model_source"])),
        stat_card(
            "S", "-" if s_value is None else f"{float(cast(float, s_value)):.3f}"
        ),
        stat_card("S source", "-" if s_source is None else str(s_source)),
        stat_card(
            "Mean reference count",
            "-"
            if mean_reference_count is None
            else f"{float(cast(float, mean_reference_count)):.3f}",
        ),
        stat_card("Reference genes", str(dataset_summary["reference_genes"])),
        stat_card(
            "Label key",
            "-"
            if dataset_summary["label_key"] is None
            else str(dataset_summary["label_key"]),
        ),
    ]
    browser_block = (
        "" if gene_browser is None else _render_gene_browser_block(gene_browser)
    )
    global_block = _render_global_eval_block(
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
        <div class="stat-grid">{"".join(stats)}</div>
        <div class="meta-block">
          <div><strong>h5ad</strong><span>{escape(str(dataset_summary["h5ad_path"]))}</span></div>
          <div><strong>checkpoint</strong><span>{escape(str(dataset_summary["ckpt_path"]))}</span></div>
        </div>
      </section>
      {global_block}
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
        f'<option value="{value}"{" selected" if ((page.descending and value == "desc") or ((not page.descending) and value == "asc")) else ""}>{label}</option>'
        for value, label in dir_options
    )
    rows = (
        "".join(_candidate_row(candidate) for candidate in page.items)
        or '<tr><td colspan="5" class="muted">No genes match the current filter.</td></tr>'
    )
    page_start = 0 if page.total_items == 0 else (page.page - 1) * page.page_size + 1
    page_end = min(page.page * page.page_size, page.total_items)
    return f"""
      <section class="panel">
        <h2>Gene browser</h2>
        <form class="browser-toolbar" action="/" method="get">
          <input type="text" name="browse_q" value="{escape(page.query)}" placeholder="Filter genes by substring">
          <select name="browse_scope">{scope_html}</select>
          <select name="browse_sort">{sort_html}</select>
          <select name="browse_dir">{dir_html}</select>
          <input type="number" name="browse_page" min="1" max="{page.total_pages}" step="1" value="{page.page}">
          <button type="submit">Apply</button>
        </form>
        <div class="browser-summary"><span class="browser-page-label">Showing {page_start}-{page_end} / {page.total_items}</span><span class="browser-page-label">Page {page.page} / {page.total_pages}</span></div>
        <div class="table-wrap browser-table-wrap"><table class="browser-table"><thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th><th>Detected cells</th><th>Detected frac</th></tr></thead><tbody>{rows}</tbody></table></div>
      </section>
    """


def _render_global_eval_block(
    result: GlobalEvaluationResult | None,
    *,
    has_checkpoint: bool,
    params: GlobalEvalParams,
    figures: dict[str, str],
) -> str:
    if not has_checkpoint:
        return '<section class="panel"><h2>Global evaluation</h2><p>Load a checkpoint to enable checkpoint-backed global metrics.</p></section>'
    controls = f"""
    <form class="toolbar" action="/" method="get">
      <input type="hidden" name="global_eval" value="1">
      <input type="number" name="ge_max_cells" min="64" step="1" value="{params.max_cells}" placeholder="max cells">
      <input type="number" name="ge_max_genes" min="8" step="1" value="{params.max_genes}" placeholder="max genes">
      <input type="number" name="ge_batch" min="1" step="1" value="{params.gene_batch_size}" placeholder="gene batch">
      <input type="number" name="ge_seed" min="0" step="1" value="{params.random_seed}" placeholder="seed">
      <button type="submit">Compute global metrics</button>
    </form>
    """
    if result is None:
        return f'<section class="panel"><h2>Global evaluation</h2><p>Compare raw counts, normalized counts, log-normalized counts, and checkpoint-backed signal on a sampled subset.</p>{controls}</section>'
    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics.silhouette:.4f}</td><td>{metrics.ari:.4f}</td><td>{metrics.nmi:.4f}</td><td>{metrics.pca_var_ratio:.4f}</td><td>{metrics.neighborhood_consistency:.4f}</td></tr>"
        for name, metrics in result.representation_metrics.items()
    )
    entropy_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{score:.4f}</td></tr>"
        for name, score in result.top_entropy_genes
    )
    figure_block = (
        ""
        if "global_overview" not in figures
        else f'<section class="panel figure-wide"><h3>Metric overview</h3><div class="figure"><img src="{figures["global_overview"]}"></div></section>'
    )
    return f'<section class="panel"><h2>Global evaluation</h2><p class="muted">Label source: <strong>{escape(result.label_key)}</strong> | Cells: {result.n_cells:,} | Genes: {result.n_genes:,}</p>{controls}<div class="table-wrap"><table class="compact-table"><thead><tr><th>Representation</th><th>Silhouette</th><th>ARI</th><th>NMI</th><th>PCA var@10</th><th>Neighbor consistency</th></tr></thead><tbody>{rows}</tbody></table></div><h3>Top prior-entropy genes</h3><div class="table-wrap"><table><thead><tr><th>Gene</th><th>Entropy</th></tr></thead><tbody>{entropy_rows}</tbody></table></div></section>{figure_block}'


def _candidate_row(candidate: GeneCandidate) -> str:
    href = f"/gene?q={quote_plus(candidate.gene_name)}"
    return f'<tr><td><a href="{href}">{escape(candidate.gene_name)}</a></td><td>{candidate.gene_index}</td><td>{candidate.total_umi:,}</td><td>{candidate.detected_cells:,}</td><td>{candidate.detected_fraction:.3f}</td></tr>'
