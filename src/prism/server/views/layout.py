from __future__ import annotations

from html import escape

from .components import render_chip_row, render_section_header


def render_page(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{escape(title)}</title>
    <link rel=\"stylesheet\" href=\"/assets/base.css?v=20260409b\">
  </head>
  <body>
    <main class=\"page\"><div class=\"page-shell\">{body}</div></main>
  </body>
</html>"""


def render_nav(*, current_query: str = "") -> str:
    chips = render_chip_row(
        (
            ("Single Context", "info"),
            ("Posterior Diagnostics", "neutral"),
            ("kBulk Ready", "warning"),
        )
    )
    return f"""
    <section class=\"panel hero\">
      <div class=\"hero-copy-block\">
        <p class=\"eyebrow\">Local Interactive Panel</p>
        <div class=\"hero-heading-row\">
          <h1>PRISM Server</h1>
          {chips}
        </div>
        <p class=\"hero-copy\">Load an <code>.h5ad</code> dataset, inspect checkpoint priors, run posterior inference, fit one gene on demand, and compare kBulk samples by class.</p>
        <div class=\"hero-note-grid\">
          <div class=\"hero-note\"><span>Dataset</span><strong>Snapshot, browse, search</strong></div>
          <div class=\"hero-note\"><span>Gene</span><strong>Raw, posterior, fit</strong></div>
          <div class=\"hero-note\"><span>Group</span><strong>Label-aware kBulk views</strong></div>
        </div>
      </div>
      <form class=\"search search-card\" action=\"/gene\" method=\"get\">
        <label class=\"field field-grow\"><span>Gene Lookup</span><input type=\"text\" name=\"q\" value=\"{escape(current_query)}\" placeholder=\"Enter gene name or index\"></label>
        <div class=\"action-row\">
          <button type=\"submit\">Open Gene</button>
          <a class=\"button ghost\" href=\"/\">Home</a>
        </div>
      </form>
    </section>
    """


def render_loader(*, h5ad_path: str = "", ckpt_path: str = "", layer: str = "") -> str:
    section_header = render_section_header(
        "Load Dataset",
        "Attach a local dataset and an optional checkpoint to unlock browsing, posterior inspection, and kBulk comparison.",
        eyebrow="Workspace",
    )
    return f"""
    <section class=\"panel\">
      {section_header}
      <form class=\"loader\" action=\"/load\" method=\"get\">
        <label class=\"field\"><span>Dataset (.h5ad)</span><input type=\"text\" name=\"h5ad\" value=\"{escape(h5ad_path)}\" placeholder=\"/path/to/data.h5ad\"></label>
        <label class=\"field\"><span>Checkpoint (optional)</span><input type=\"text\" name=\"ckpt\" value=\"{escape(ckpt_path)}\" placeholder=\"/path/to/model.ckpt\"></label>
        <label class=\"field\"><span>Layer (optional)</span><input type=\"text\" name=\"layer\" value=\"{escape(layer)}\" placeholder=\"layer name\"></label>
        <div class=\"form-actions form-actions-inline\"><button type=\"submit\">Load</button></div>
      </form>
    </section>
    """


def stat_card(label: str, value: str) -> str:
    return (
        '<div class="stat">'
        f'<span class="stat-label">{escape(label)}</span>'
        f'<strong class="stat-value">{escape(value)}</strong>'
        "</div>"
    )


def render_message(message: str, *, level: str = "info") -> str:
    tone = "Error" if level == "error" else "Notice"
    return f'<section class="panel notice notice-{escape(level)}" role="alert"><div class="notice-head"><span class="notice-tag">{tone}</span><p>{escape(message)}</p></div></section>'
