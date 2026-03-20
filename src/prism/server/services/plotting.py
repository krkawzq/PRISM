from __future__ import annotations

from html import escape

import numpy as np


def histogram_svg(
    values: np.ndarray,
    *,
    title: str,
    color: str,
    bins: int = 40,
    width: int = 520,
    height: int = 260,
) -> str:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return _empty_figure(title, width, height)

    hist, edges = np.histogram(values, bins=bins)
    max_count = max(float(np.max(hist)), 1.0)
    bar_width = width / max(len(hist), 1)
    bars = []
    for idx, count in enumerate(hist):
        bar_h = (count / max_count) * (height - 48)
        x = idx * bar_width
        y = height - 28 - bar_h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 1.0, 1.0):.2f}" '
            f'height="{bar_h:.2f}" rx="2" fill="{escape(color)}" opacity="0.85" />'
        )
    return _wrap_svg(
        title,
        width,
        height,
        "".join(bars),
        footer=f"min={edges[0]:.2f} max={edges[-1]:.2f}",
    )


def scatter_svg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    color: str,
    width: int = 520,
    height: int = 260,
    max_points: int = 2500,
) -> str:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size == 0 or y.size == 0:
        return _empty_figure(title, width, height)

    if x.size > max_points:
        rng = np.random.default_rng(0)
        keep = np.sort(rng.choice(np.arange(x.size), size=max_points, replace=False))
        x = x[keep]
        y = y[keep]

    x_lo, x_hi = _limits(x)
    y_lo, y_hi = _limits(y)
    points = []
    for xv, yv in zip(x, y, strict=True):
        px = 18 + (xv - x_lo) / max(x_hi - x_lo, 1e-12) * (width - 36)
        py = height - 24 - (yv - y_lo) / max(y_hi - y_lo, 1e-12) * (height - 52)
        points.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="2.2" fill="{escape(color)}" opacity="0.35" />'
        )
    return _wrap_svg(title, width, height, "".join(points), footer=f"n={x.size}")


def multi_line_svg(
    x: np.ndarray,
    lines: np.ndarray,
    *,
    title: str,
    width: int = 520,
    height: int = 260,
) -> str:
    x = np.asarray(x, dtype=float)
    lines = np.asarray(lines, dtype=float)
    if x.size == 0 or lines.size == 0:
        return _empty_figure(title, width, height)

    x_lo, x_hi = _limits(x)
    y_lo, y_hi = _limits(lines)
    palette = ["#155e75", "#1d4ed8", "#c2410c", "#0f766e", "#7c3aed", "#b45309"]
    paths = []
    for idx, line in enumerate(lines):
        coords = []
        for xv, yv in zip(x, line, strict=True):
            px = 18 + (xv - x_lo) / max(x_hi - x_lo, 1e-12) * (width - 36)
            py = height - 24 - (yv - y_lo) / max(y_hi - y_lo, 1e-12) * (height - 52)
            coords.append(f"{px:.2f},{py:.2f}")
        color = palette[idx % len(palette)]
        paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.8" opacity="0.85" points="{" ".join(coords)}" />'
        )
    return _wrap_svg(title, width, height, "".join(paths))


def _wrap_svg(title: str, width: int, height: int, body: str, footer: str = "") -> str:
    label = f'<text x="16" y="22" class="plot-title">{escape(title)}</text>'
    foot = (
        f'<text x="16" y="{height - 8}" class="plot-footer">{escape(footer)}</text>'
        if footer
        else ""
    )
    border = f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="14" class="plot-frame" />'
    return f'<svg viewBox="0 0 {width} {height}" class="plot-svg" role="img">{border}{label}{body}{foot}</svg>'


def _empty_figure(title: str, width: int, height: int) -> str:
    message = (
        '<text x="50%" y="54%" text-anchor="middle" class="plot-empty">No data</text>'
    )
    return _wrap_svg(title, width, height, message)


def _limits(values: np.ndarray) -> tuple[float, float]:
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi
