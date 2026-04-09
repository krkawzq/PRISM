from __future__ import annotations

from html import escape


def render_section_header(
    title: str,
    description: str = "",
    *,
    eyebrow: str = "",
) -> str:
    parts = ['<div class="section-head">']
    if eyebrow:
        parts.append(f'<p class="eyebrow">{escape(eyebrow)}</p>')
    parts.append(f"<h2>{escape(title)}</h2>")
    if description:
        parts.append(f'<p class="section-copy">{escape(description)}</p>')
    parts.append("</div>")
    return "".join(parts)


def render_chip(label: str, *, tone: str = "neutral") -> str:
    return f'<span class="chip chip-{escape(tone)}">{escape(label)}</span>'


def render_chip_row(chips: list[tuple[str, str]] | tuple[tuple[str, str], ...]) -> str:
    if not chips:
        return ""
    rendered = "".join(render_chip(label, tone=tone) for label, tone in chips)
    return f'<div class="chip-row">{rendered}</div>'


def render_detail_grid(items: list[tuple[str, str]]) -> str:
    rendered = "".join(
        (
            '<div class="detail-item">'
            f"<dt>{escape(label)}</dt>"
            f"<dd>{escape(value)}</dd>"
            "</div>"
        )
        for label, value in items
        if value
    )
    if not rendered:
        return ""
    return f'<dl class="detail-grid">{rendered}</dl>'


__all__ = [
    "render_chip",
    "render_chip_row",
    "render_detail_grid",
    "render_section_header",
]
