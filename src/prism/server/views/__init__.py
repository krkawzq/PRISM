from __future__ import annotations

"""Legacy server-rendered views kept for compatibility tests."""

from .gene import render_gene_page
from .home import render_home_page
from .layout import render_loader, render_message, render_nav, render_page, stat_card

__all__ = [
    "render_gene_page",
    "render_home_page",
    "render_loader",
    "render_message",
    "render_nav",
    "render_page",
    "stat_card",
]
