from __future__ import annotations

from html import escape


def render_page(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>{escape(title)}</title>
    <link rel=\"stylesheet\" href=\"/assets/base.css?v=20260321b\">
  </head>
  <body>
    <main class=\"page\">{body}</main>
  </body>
</html>"""


def render_nav(*, current_query: str = "") -> str:
    return f"""
    <section class=\"panel hero\">
      <div>
        <h1>PRISM Analysis Server</h1>
        <p>Checkpoint-aware gene inspection with PRISM diagnostics, baseline comparisons, and rich signal-space plots.</p>
      </div>
      <form class=\"search\" action=\"/gene\" method=\"get\">
        <input type=\"text\" name=\"q\" value=\"{escape(current_query)}\" placeholder=\"Enter a gene name or index\">
        <button type=\"submit\">Inspect gene</button>
        <a class=\"button ghost\" href=\"/\">Home</a>
      </form>
    </section>
    """


def render_loader(*, h5ad_path: str = "", ckpt_path: str = "", layer: str = "") -> str:
    return f"""
    <section class=\"panel\">
      <h2>Load Dataset</h2>
      <form class=\"loader\" action=\"/load\" method=\"get\">
        <input type=\"text\" name=\"h5ad\" value=\"{escape(h5ad_path)}\" placeholder=\"/path/to/data.h5ad\">
        <input type=\"text\" name=\"ckpt\" value=\"{escape(ckpt_path)}\" placeholder=\"/path/to/merged.ckpt (optional)\">
        <input type=\"text\" name=\"layer\" value=\"{escape(layer)}\" placeholder=\"layer name (optional)\">
        <button type=\"submit\">Load</button>
      </form>
    </section>
    """


def stat_card(label: str, value: str) -> str:
    return f'<div class="stat"><span class="stat-label">{escape(label)}</span><span class="stat-value">{escape(value)}</span></div>'
