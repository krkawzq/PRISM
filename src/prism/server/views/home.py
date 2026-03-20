from __future__ import annotations

from html import escape
from typing import cast
from urllib.parse import quote_plus

from prism.server.services.datasets import GeneCandidate
from prism.server.services.global_eval import GlobalEvaluationResult

from .layout import render_loader, render_nav, render_page, stat_card


def render_home_page(
    *,
    dataset_summary: dict[str, object] | None,
    top_genes: list[GeneCandidate],
    search_query: str = "",
    search_results: list[GeneCandidate] | None = None,
    h5ad_path: str = "",
    ckpt_path: str = "",
    layer: str = "",
    error_message: str | None = None,
    global_eval: GlobalEvaluationResult | None = None,
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

    search_block = ""
    if search_results is not None:
        rows = "".join(_candidate_row(candidate) for candidate in search_results)
        search_block = f"""
        <section class="panel">
          <h2>Search results</h2>
          <div class="table-wrap"><table>
            <thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th><th>Detected cells</th><th>Detected frac</th></tr></thead>
            <tbody>{rows}</tbody>
          </table></div>
        </section>
        """

    rows = "".join(_candidate_row(candidate) for candidate in top_genes)
    global_eval_block = _render_global_eval_block(
        global_eval, has_checkpoint=bool(dataset_summary["ckpt_path"])
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
      {search_block}
      <section class="panel">
        <h2>Top genes</h2>
        <div class="table-wrap"><table>
          <thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th><th>Detected cells</th><th>Detected frac</th></tr></thead>
          <tbody>{rows}</tbody>
        </table></div>
      </section>
    """
    return render_page(title="PRISM Analysis Server", body=body)


def _render_global_eval_block(
    result: GlobalEvaluationResult | None, *, has_checkpoint: bool
) -> str:
    if not has_checkpoint:
        return '<section class="panel"><h2>Global evaluation</h2><p>Load a checkpoint to enable on-demand global metrics.</p></section>'

    controls = """
    <form class="toolbar" action="/" method="get">
      <input type="hidden" name="global_eval" value="1">
      <button type="submit">Compute global metrics</button>
    </form>
    """
    if result is None:
        return f'<section class="panel"><h2>Global evaluation</h2><p>Compute checkpoint-backed global metrics on demand for X, NormalizeTotalX, Log1pNormalizeTotalX, and signal.</p>{controls}</section>'

    rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics.silhouette:.4f}</td><td>{metrics.ari:.4f}</td><td>{metrics.nmi:.4f}</td><td>{metrics.pca_var_ratio:.4f}</td><td>{metrics.neighborhood_consistency:.4f}</td><td>{'-' if metrics.mean_treatment_cv is None else f'{metrics.mean_treatment_cv:.4f}'}</td></tr>"
        for name, metrics in result.representation_metrics.items()
    )
    meta = f'<p class="muted">Label source: <strong>{escape(result.label_key)}</strong> | Cells: {result.n_cells:,} | Genes: {result.n_genes:,} | Labels: {result.n_labels}</p>'
    table = f'<div class="table-wrap"><table class="compact-table"><thead><tr><th>Representation</th><th>Silhouette</th><th>ARI</th><th>NMI</th><th>PCA var@10</th><th>Neighbor consistency</th><th>Mean treatment CV</th></tr></thead><tbody>{rows}</tbody></table></div>'
    fg = result.fg_summary
    fg_stats = "".join(
        [
            stat_card("Entropy mean", f"{fg.entropy_mean:.4f}"),
            stat_card("Entropy p95", f"{fg.entropy_p95:.4f}"),
            stat_card("Peak count mean", f"{fg.peak_count_mean:.3f}"),
            stat_card("Sharpness mean", f"{fg.sharpness_mean:.4f}"),
            stat_card("HVG rho", f"{fg.hvg_spearman:.4f}"),
            stat_card("Overlap@100", f"{fg.overlap_at_100:.3f}"),
            stat_card("Overlap@500", f"{fg.overlap_at_500:.3f}"),
            stat_card("Entropy~Expr rho", f"{fg.entropy_expression_spearman:.4f}"),
            stat_card("Entropy~Sharp rho", f"{fg.entropy_sharpness_spearman:.4f}"),
            stat_card("Entropy~Peak rho", f"{fg.entropy_peak_spearman:.4f}"),
        ]
    )
    fg_rows = "".join(
        f"<tr><td>{idx + 1}</td><td>{escape(name)}</td><td>{entropy:.4f}</td></tr>"
        for idx, (name, entropy) in enumerate(result.top_entropy_genes)
    )
    fg_block = f'<section class="panel"><h3>F_g analysis</h3><div class="stat-grid">{fg_stats}</div><p class="muted">Low-expression entropy mean: {fg.low_expression_entropy_mean:.4f} | High-expression entropy mean: {fg.high_expression_entropy_mean:.4f}</p><div class="table-wrap"><table><thead><tr><th>Rank</th><th>Gene</th><th>F_g entropy</th></tr></thead><tbody>{fg_rows}</tbody></table></div></section>'
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
