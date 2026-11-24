"""Chart style shared by the notebooks.

Every figure is drawn twice, once for a light page and once for a dark page, so
the README can show the right one with a <picture> element. The three series
colors (blue, orange, aqua) come from a categorical palette that was checked for
color-vision-deficiency separation, lightness band and contrast on both
surfaces. Text always uses the ink colors; color is only used on marks.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "blue": "#2a78d6",
        "orange": "#eb6834",
        "aqua": "#1baf7a",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "blue": "#3987e5",
        "orange": "#d95926",
        "aqua": "#199e70",
    },
}


def _apply(t: dict) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "text.color": t["ink"],
            "axes.labelcolor": t["ink2"],
            "axes.edgecolor": t["axis"],
            "xtick.color": t["ink2"],
            "ytick.color": t["ink2"],
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
        }
    )


def new_axes(t: dict, figsize: tuple[float, float], grid: str = "x", hide_left: bool = True):
    """One axes with a recessive chrome: hairline solid grid, no top/right frame."""
    _apply(t)
    fig, ax = plt.subplots(figsize=figsize, facecolor=t["surface"])
    ax.set_facecolor(t["surface"])
    for side in ("top", "right") + (("left",) if hide_left else ()):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(t["axis"])
    ax.tick_params(length=0)
    if grid:
        ax.grid(axis=grid if grid in ("x", "y") else "both", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def title(ax, t: dict, text: str) -> None:
    ax.set_title(text, loc="left", color=t["ink"], fontsize=12, fontweight="bold", pad=14)


def legend(ax, t: dict, handles, **kwargs):
    return ax.legend(
        handles=handles, frameon=False, fontsize=9, labelcolor=t["ink2"], **kwargs
    )


def show_and_save(draw, name: str, fig_dir) -> None:
    """Render `draw(theme)` for dark and light, save both PNGs, show the light one."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("dark", "light"):
        t = THEMES[mode]
        fig = draw(t)
        fig.savefig(
            fig_dir / f"{name}-{mode}.png",
            dpi=200,
            facecolor=t["surface"],
            bbox_inches="tight",
            pad_inches=0.25,
        )
        if mode == "dark":
            plt.close(fig)
    plt.show()
    plt.close("all")


def dot(color: str, label: str, line: bool = False):
    """Legend handle: a filled dot, optionally with a short line through it."""
    from matplotlib.lines import Line2D

    return Line2D(
        [0], [0],
        marker="o",
        color=color if line else "none",
        linewidth=2 if line else 0,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0,
        markersize=8,
        label=label,
    )
