from __future__ import annotations

from html import escape
from typing import cast
from urllib.parse import quote_plus

from prism.server.services.datasets import GeneCandidate

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
) -> str:
    loader = render_loader(h5ad_path=h5ad_path, ckpt_path=ckpt_path, layer=layer)
    if dataset_summary is None:
        message = (
            f'<section class="panel"><h2>Load error</h2><p>{escape(error_message)}</p></section>'
            if error_message
            else '<section class="panel"><h2>Welcome</h2><p>Load an h5ad file and a merged PRISM checkpoint to start browsing and analysis.</p></section>'
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
            stat_card("s_hat", f"{s_hat:.3f}"),
            stat_card("Median total", f"{median_total:.1f}"),
        ]
    )

    search_block = ""
    if search_results is not None:
        rows = "".join(_candidate_row(candidate) for candidate in search_results)
        search_block = f"""
        <section class=\"panel\">
          <h2>Search results</h2>
          <table>
            <thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th><th>Detected cells</th><th>Detected frac</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </section>
        """

    rows = "".join(_candidate_row(candidate) for candidate in top_genes)
    body = f"""
      {render_nav(current_query=search_query)}
      {loader}
      <section class=\"panel\">
        <h2>Dataset snapshot</h2>
        <div class=\"stat-grid\">{stats}</div>
        <div class=\"meta-block\">
          <div><strong>h5ad</strong><span>{escape(str(dataset_summary["h5ad_path"]))}</span></div>
          <div><strong>checkpoint</strong><span>{escape(str(dataset_summary["ckpt_path"]))}</span></div>
        </div>
      </section>
      {search_block}
      <section class=\"panel\">
        <h2>Top fitted genes</h2>
        <table>
          <thead><tr><th>Gene</th><th>Index</th><th>Total UMI</th><th>Detected cells</th><th>Detected frac</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </section>
    """
    return render_page(title="PRISM Analysis Server", body=body)


def _candidate_row(candidate: GeneCandidate) -> str:
    href = f"/gene?q={quote_plus(candidate.gene_name)}"
    return (
        f'<tr><td><a href="{href}">{escape(candidate.gene_name)}</a></td>'
        f"<td>{candidate.gene_index}</td>"
        f"<td>{candidate.total_umi:,}</td>"
        f"<td>{candidate.detected_cells:,}</td>"
        f"<td>{candidate.detected_fraction:.3f}</td></tr>"
    )
