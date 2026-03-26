from __future__ import annotations

from html import escape
from urllib.parse import quote_plus

from prism.server.services.analysis import GeneAnalysis, GeneSummary
from prism.server.services.datasets import GeneCandidate

from .layout import render_nav, render_page, stat_card


def _default_fit_params() -> dict[str, object]:
    return {
        "r": 0.05,
        "grid_size": 512,
        "sigma_bins": 1.0,
        "align_loss_weight": 1.0,
        "lr": 0.05,
        "n_iter": 100,
        "lr_min_ratio": 0.1,
        "init_temperature": 1.0,
        "cell_chunk_size": 512,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "torch_dtype": "float64",
        "device": "cuda",
    }


def _candidate_block(
    *, gene_name: str, candidates: list[GeneCandidate] | None = None
) -> str:
    if not candidates:
        return ""

    rows = "".join(
        f'<tr><td><a href="/gene?q={quote_plus(item.gene_name)}">{escape(item.gene_name)}</a></td>'
        f"<td>{item.gene_index}</td><td>{item.total_umi:,}</td></tr>"
        for item in candidates
        if item.gene_name != gene_name
    )
    if not rows:
        return ""

    return (
        '<section class="panel"><h2>Nearby matches</h2>'
        "<table><thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></section>"
    )


def _fit_form(search_query: str, fit_params: dict[str, object] | None) -> str:
    params = _default_fit_params() if fit_params is None else fit_params
    device_options = ["cuda", "cpu"]
    optimizer_options = ["adamw", "adam", "sgd", "rmsprop"]
    scheduler_options = ["cosine", "linear", "constant", "step"]
    dtype_options = ["float64", "float32"]
    device_html = "".join(
        f'<option value="{name}"{" selected" if str(params["device"]) == name else ""}>{name}</option>'
        for name in device_options
    )
    optimizer_html = "".join(
        f'<option value="{name}"{" selected" if str(params["optimizer"]) == name else ""}>{name}</option>'
        for name in optimizer_options
    )
    scheduler_html = "".join(
        f'<option value="{name}"{" selected" if str(params["scheduler"]) == name else ""}>{name}</option>'
        for name in scheduler_options
    )
    dtype_html = "".join(
        f'<option value="{name}"{" selected" if str(params["torch_dtype"]) == name else ""}>{name}</option>'
        for name in dtype_options
    )
    return f"""
      <section class="panel fit-panel">
        <form class="fit-form" action="/gene" method="get">
          <input type="hidden" name="q" value="{escape(search_query)}">
          <input type="hidden" name="fit" value="1">
          <div class="fit-header fit-hero">
            <div class="fit-title-stack">
              <div>
                <h2>On-demand gene fitting</h2>
                <p>Set the pool scale and optimization controls first, then launch fitting manually.</p>
              </div>
              <div class="fit-badges">
                <span class="fit-badge">manual start</span>
                <span class="fit-badge">terminal progress</span>
                <span class="fit-badge">device default: cuda</span>
              </div>
            </div>
            <div class="fit-action-card">
              <p>Nothing starts on page load. Submit once the settings look right.</p>
              <button type="submit">Fit / Refresh</button>
            </div>
          </div>
          <div class="fit-overview">
            <label class="fit-inline-field fit-inline-field-strong"><span>r</span><input type="number" step="any" name="r" value="{params["r"]}" placeholder="0.05"><small>recommended 0.05</small></label>
            <label class="fit-inline-field"><span>grid size</span><input type="number" min="16" step="1" name="grid_size" value="{params["grid_size"]}" placeholder="512"><small>512 is a solid default</small></label>
            <label class="fit-inline-field"><span>iterations</span><input type="number" min="1" step="1" name="n_iter" value="{params["n_iter"]}" placeholder="100"><small>start with 100</small></label>
            <label class="fit-inline-field"><span>device</span><select name="device">{device_html}</select><small>use `cuda` when available</small></label>
          </div>
          <details class="fit-advanced" open>
            <summary>
              <span>Advanced controls</span>
              <span class="muted">prior shape, optimizer, scheduler, and runtime knobs</span>
            </summary>
            <div class="fit-sections">
            <div class="fit-section fit-section-primary">
              <h3>Scale and prior grid</h3>
              <div class="fit-fields">
                <label class="fit-field"><span>sigma bins</span><input type="number" step="any" name="sigma_bins" value="{params["sigma_bins"]}" placeholder="1.0"><small>1.0 keeps smoothing moderate</small></label>
                <label class="fit-field"><span>align weight</span><input type="number" step="any" name="align_loss_weight" value="{params["align_loss_weight"]}" placeholder="1.0"><small>1.0 balances fit and self-alignment</small></label>
              </div>
            </div>
            <div class="fit-section">
              <h3>Optimization</h3>
              <div class="fit-fields">
                <label class="fit-field"><span>learning rate</span><input type="number" step="any" name="lr" value="{params["lr"]}" placeholder="0.05"><small>0.05 works well for one-gene fits</small></label>
                <label class="fit-field"><span>lr min ratio</span><input type="number" step="any" name="lr_min_ratio" value="{params["lr_min_ratio"]}" placeholder="0.1"><small>end at 10% of the start lr</small></label>
                <label class="fit-field fit-field-wide"><span>init temperature</span><input type="number" step="any" name="init_temperature" value="{params["init_temperature"]}" placeholder="1.0"><small>1.0 keeps the initialization neutral</small></label>
              </div>
            </div>
            <div class="fit-section">
              <h3>Runtime</h3>
              <div class="fit-fields">
                <label class="fit-field"><span>cell chunk</span><input type="number" min="1" step="1" name="cell_chunk_size" value="{params["cell_chunk_size"]}" placeholder="512"><small>reduce if GPU memory is tight</small></label>
                <label class="fit-field"><span>torch dtype</span><select name="torch_dtype">{dtype_html}</select><small>`float64` is more stable</small></label>
                <label class="fit-field"><span>optimizer</span><select name="optimizer">{optimizer_html}</select><small>`adamw` is the default choice</small></label>
                <label class="fit-field"><span>scheduler</span><select name="scheduler">{scheduler_html}</select><small>`cosine` is usually the best starting point</small></label>
              </div>
            </div>
            </div>
          </details>
        </form>
      </section>
    """


def render_gene_pending_page(
    *,
    gene_name: str,
    gene_index: int,
    summary: GeneSummary,
    search_query: str,
    fit_params: dict[str, object] | None = None,
    candidates: list[GeneCandidate] | None = None,
    figures: dict[str, str] | None = None,
) -> str:
    figures = {} if figures is None else figures
    stats = "".join(
        [
            stat_card("Gene", gene_name),
            stat_card("Index", str(gene_index)),
            stat_card("Source", "raw-only"),
            stat_card("Mean count", f"{summary.mean_count:.3f}"),
            stat_card("Detected frac", f"{summary.detected_frac:.3f}"),
            stat_card("Zero frac", f"{summary.zero_frac:.3f}"),
            stat_card("P99 count", f"{summary.p99_count:.3f}"),
            stat_card("Depth corr", f"{summary.count_total_correlation:.3f}"),
        ]
    )
    overview_block = ""
    if "gene_overview" in figures:
        overview_block = f'<section class="panel figure-wide"><h2>Gene overview</h2><div class="figure"><img src="{figures["gene_overview"]}"></div></section>'

    body = f"""
      {render_nav(current_query=search_query)}
      {_fit_form(search_query, fit_params)}
      <section class="panel"><h2>Ready to fit</h2><p>No prior is attached to <strong>{escape(gene_name)}</strong> yet. Review the hyperparameters above, then click <strong>Fit / Refresh</strong> to start the on-demand fit. Terminal progress will appear while fitting runs.</p></section>
      <section class="panel"><h2>Raw summary</h2><div class="stat-grid">{stats}</div></section>
      {_candidate_block(gene_name=gene_name, candidates=candidates)}
      {overview_block}
    """
    return render_page(title=f"PRISM Gene: {gene_name}", body=body)


def render_gene_page(
    *,
    analysis: GeneAnalysis,
    figures: dict[str, str],
    search_query: str,
    candidates: list[GeneCandidate] | None = None,
    fit_params: dict[str, object] | None = None,
    treatment_block_html: str = "",
    include_3d: bool = False,
) -> str:
    stats = "".join(
        [
            stat_card("Gene", analysis.gene_name),
            stat_card("Index", str(analysis.gene_index)),
            stat_card("Source", analysis.source),
            stat_card("s_hat", f"{analysis.s_hat:.3f}"),
            stat_card("Mean count", f"{analysis.summary.mean_count:.3f}"),
            stat_card("Detected frac", f"{analysis.summary.detected_frac:.3f}"),
            stat_card("Mean signal", f"{float(analysis.signal.mean()):.3f}"),
            stat_card(
                "Mean post. entropy",
                f"{float(analysis.confidence.mean()):.3f}",
            ),
            stat_card(
                "Mean prior entropy", f"{float(analysis.prior_entropy.mean()):.3f}"
            ),
            stat_card(
                "Mean mutual info", f"{float(analysis.mutual_information.mean()):.3f}"
            ),
            stat_card("Depth corr", f"{analysis.summary.count_total_correlation:.3f}"),
        ]
    )
    metric_rows = "".join(
        f"<tr><td>{escape(name)}</td><td>{metrics.mean:.4f}</td><td>{metrics.median:.4f}</td><td>{metrics.std:.4f}</td><td>{metrics.var:.4f}</td><td>{metrics.p95:.4f}</td><td>{metrics.nonzero_frac:.4f}</td><td>{metrics.depth_corr:.4f}</td><td>{metrics.depth_mi:.4f}</td><td>{'-' if metrics.sparsity_corr is None else f'{metrics.sparsity_corr:.4f}'}</td><td>{'-' if metrics.fisher_ratio is None else f'{metrics.fisher_ratio:.4f}'}</td><td>{'-' if metrics.kruskal_h is None else f'{metrics.kruskal_h:.4f}'}</td><td>{'-' if metrics.kruskal_p is None else f'{metrics.kruskal_p:.3e}'}</td><td>{'-' if metrics.auroc_ovr is None else f'{metrics.auroc_ovr:.4f}'}</td><td>{'-' if metrics.zero_consistency is None else f'{metrics.zero_consistency:.4f}'}</td><td>{'-' if metrics.zero_rank_tau is None else f'{metrics.zero_rank_tau:.4f}'}</td><td>{'-' if metrics.dropout_recovery is None else f'{metrics.dropout_recovery:.4f}'}</td><td>{'-' if metrics.treatment_cv is None else f'{metrics.treatment_cv:.4f}'}</td></tr>"
        for name, metrics in analysis.representation_metrics.items()
    )

    stage0_block = ""
    if "stage0" in figures:
        stage0_block = f'<section class="panel figure-wide"><h2>Stage 0: Sampling-pool scale</h2><div class="figure"><img src="{figures["stage0"]}"></div></section>'

    trace_block = ""
    if "loss_trace" in figures:
        trace_block = f'<section class="panel figure-wide"><h2>Stage 1: Optimization trace</h2><div class="figure"><img src="{figures["loss_trace"]}"></div></section>'

    init_block = ""
    if "init_comparison" in figures:
        init_block = f'<section class="panel figure-wide"><h2>Initialization comparison</h2><div class="figure"><img src="{figures["init_comparison"]}"></div></section>'

    signal_3d_block = (
        f'<section class="panel figure-wide"><h2>3D signal space</h2><div class="figure">{figures["signal_3d"]}</div></section>'
        if include_3d and "signal_3d" in figures
        else f'<section class="panel"><h2>3D signal space</h2><p>Load the interactive 3D view on demand to avoid blocking the first render.</p><p><a href="/gene?q={quote_plus(search_query)}&view3d=1">Open 3D view</a></p></section>'
    )

    body = f"""
      {render_nav(current_query=search_query)}
      {_fit_form(search_query, fit_params)}
      <section class="panel"><h2>Gene summary</h2><div class="stat-grid">{stats}</div></section>
      <section class="panel"><h2>Baseline metrics</h2><div class="table-wrap"><table class="compact-table"><thead><tr><th>Signal</th><th>Mean</th><th>Median</th><th>Std</th><th>Var</th><th>P95</th><th>Nonzero frac</th><th>Depth corr</th><th>Depth MI</th><th>Sparsity corr</th><th>Fisher ratio</th><th>Kruskal H</th><th>Kruskal p</th><th>AUROC OVR</th><th>Zero-group consistency</th><th>Zero rank tau</th><th>Dropout recovery</th><th>Treatment CV</th></tr></thead><tbody>{metric_rows}</tbody></table></div></section>
      {_candidate_block(gene_name=analysis.gene_name, candidates=candidates)}
      <section class="panel figure-wide"><h2>Gene overview</h2><div class="figure"><img src="{figures["gene_overview"]}"></div></section>
      {treatment_block_html}
      {stage0_block}
      {trace_block}
      {init_block}
      <section class="panel figure-wide"><h2>Prior and self-consistency</h2><div class="figure"><img src="{figures["prior_fit"]}"></div></section>
      <section class="panel figure-wide"><h2>Signal interface</h2><div class="figure"><img src="{figures["signal_interface"]}"></div></section>
      {signal_3d_block}
      <section class="panel figure-wide"><h2>Posterior gallery</h2><div class="figure"><img src="{figures["posterior_gallery"]}"></div></section>
    """
    return render_page(title=f"PRISM Gene: {analysis.gene_name}", body=body)
