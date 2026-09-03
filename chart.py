"""The client-facing graphic. Everything visual lives here."""
from __future__ import annotations

import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from .config import ChartConfig, UNDERLYING_COLORS, WORST_OF_COLOR
from .engine import Prepared, level_value

GREY_TEXT = "#5D6D7E"
GRID = "#E6EAED"
FONTS = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONTS,
        "axes.edgecolor": "#BFC9D0",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def build_figure(prep: Prepared, cfg: ChartConfig, figsize=(12.2, 6.4)) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.065, right=0.86, top=0.80, bottom=0.20)

    if prep.empty:
        ax.text(0.5, 0.5, "No data to plot", ha="center", va="center",
                color=GREY_TEXT, transform=ax.transAxes)
        ax.set_axis_off()
        return fig

    # ---------------- underlyings ---------------- #
    handles = []
    for i, col in enumerate(prep.plotted.columns):
        color = UNDERLYING_COLORS[i % len(UNDERLYING_COLORS)]
        ln, = ax.plot(prep.plotted.index, prep.plotted[col], lw=1.8,
                      color=color, label=col, solid_joinstyle="round", zorder=5)
        handles.append(ln)

    if cfg.show_worst_of and prep.plotted.shape[1] > 1:
        # drawn UNDER the underlyings as a wide translucent band: the worst-of sits
        # exactly on top of whichever leg is lowest, and would hide it otherwise
        ln, = ax.plot(prep.worst_of.index, prep.worst_of.values, lw=4.5, alpha=0.45,
                      color=WORST_OF_COLOR, label=cfg.worst_of_label,
                      solid_capstyle="round", zorder=4)
        handles.append(ln)

    # ---------------- barrier / trigger lines ---------------- #
    active_levels = [lv for lv in cfg.levels if lv.enabled]
    for lv in active_levels:
        y = level_value(lv.pct, prep.base)
        ax.axhline(y, color=lv.hex_color(), lw=1.15, ls=(0, (6, 4)), zorder=3,
                   label=lv.label if cfg.levels_in_legend else None)
        if not cfg.levels_in_legend:
            ax.annotate(f"{lv.label}", xy=(1.008, y), xycoords=("axes fraction", "data"),
                        va="center", ha="left", fontsize=8.2, color=lv.hex_color())

    # ---------------- autocall observation dates ---------------- #
    obs = [pd.Timestamp(d) for d in cfg.autocall_dates
           if prep.calendar[0] <= pd.Timestamp(d) <= prep.calendar[-1]]
    if obs:
        trigger = next((lv for lv in active_levels if lv.kind == "autocall"), None)
        for d in obs:
            ax.axvline(d, color="#C9D2D8", lw=0.7, ls=":", zorder=2)
        if trigger:
            y = level_value(trigger.pct, prep.base)
            ax.scatter(obs, [y] * len(obs), s=26, marker="v",
                       color=trigger.hex_color(), zorder=7, clip_on=False)

    # ---------------- initial fixing marker ---------------- #
    if cfg.mark_fixing:
        fx = pd.Timestamp(cfg.fixing_date)
        if prep.calendar[0] <= fx <= prep.calendar[-1]:
            ax.axvline(fx, color="#AEB9C2", lw=0.9, ls="-", zorder=2)
            ax.annotate("Initial fixing", xy=(fx, 1.012), xycoords=("data", "axes fraction"),
                        ha="left", va="bottom", fontsize=8, color=GREY_TEXT,
                        xytext=(3, 0), textcoords="offset points")

    # ---------------- last-value labels ---------------- #
    if cfg.annotate_last:
        marks = []
        for i, col in enumerate(prep.plotted.columns):
            s = prep.plotted[col].dropna()
            if len(s):
                marks.append((float(s.iloc[-1]), s.index[-1],
                              UNDERLYING_COLORS[i % len(UNDERLYING_COLORS)]))
        lo, hi = _ylim(prep, active_levels)
        for val, x, color, y_text in _spread(marks, lo, hi):
            ax.annotate(f"{val:,.1f}", xy=(x, val), xytext=(7, 0),
                        textcoords="offset points", fontsize=8.5, fontweight="bold",
                        va="center", ha="left", color=color, annotation_clip=False,
                        xycoords="data") if abs(y_text - val) < 1e-9 else \
                ax.annotate(f"{val:,.1f}", xy=(x, y_text), xytext=(7, 0),
                            textcoords="offset points", fontsize=8.5, fontweight="bold",
                            va="center", ha="left", color=color, annotation_clip=False)

    # ---------------- axes ---------------- #
    ax.set_ylabel("Performance rebased to 100" if cfg.scale == "rebased" else "Price level",
                  fontsize=9, color=GREY_TEXT, labelpad=8)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=GREY_TEXT, labelsize=8.5, length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.set_xlim(prep.calendar[0], prep.calendar[-1])
    ax.set_ylim(*_ylim(prep, active_levels))

    # ---------------- legend ---------------- #
    if cfg.levels_in_legend:
        handles = None
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.09),
                    ncol=min(4, len(prep.plotted.columns) + 1), frameon=False,
                    fontsize=9, handlelength=2.2, columnspacing=2.0)
    for t in leg.get_texts():
        t.set_color("#1F1F1F")

    # ---------------- titles & footnotes ---------------- #
    fig.text(0.065, 0.945, cfg.texts.title, fontsize=14, fontweight="bold",
             color="#1F1F1F", ha="left", va="top")
    if cfg.texts.subtitle:
        fig.text(0.065, 0.895, cfg.texts.subtitle, fontsize=10,
                 color=GREY_TEXT, ha="left", va="top")

    foot = cfg.texts.footnote or _default_footnote(cfg)
    fig.text(0.065, 0.075, foot, fontsize=7.6, color=GREY_TEXT, ha="left", va="top")
    if cfg.texts.disclaimer:
        fig.text(0.065, 0.043, cfg.texts.disclaimer, fontsize=6.8, color="#8C9BA5",
                 ha="left", va="top", wrap=True)

    # brand rule
    fig.add_artist(plt.Line2D([0.065, 0.935], [0.985, 0.985], color="#00915A", lw=3,
                              transform=fig.transFigure))
    return fig


def _spread(marks, lo, hi, min_gap_frac: float = 0.035):
    """Nudge end-of-line labels apart when two series finish at the same level."""
    if not marks:
        return []
    gap = (hi - lo) * min_gap_frac
    ordered = sorted(marks, key=lambda m: m[0])
    ys = [m[0] for m in ordered]
    for i in range(1, len(ys)):                       # push up
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    overshoot = ys[-1] - hi
    if overshoot > 0:                                 # then slide the block back down
        ys = [y - overshoot for y in ys]
    return [(m[0], m[1], m[2], y) for m, y in zip(ordered, ys)]


def _ylim(prep: Prepared, levels) -> tuple[float, float]:
    """Fit the data, but never crop a barrier out of the picture."""
    vals = [prep.plotted.to_numpy(dtype="float64").ravel()]
    if prep.worst_of is not None and len(prep.worst_of):
        vals.append(prep.worst_of.to_numpy(dtype="float64"))
    data = np.concatenate(vals)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 100.0
    lo, hi = float(data.min()), float(data.max())
    for lv in levels:
        y = level_value(lv.pct, prep.base)
        lo, hi = min(lo, y), max(hi, y)
    pad = max((hi - lo) * 0.08, hi * 0.01)
    return lo - pad, hi + pad


def _default_footnote(cfg: ChartConfig) -> str:
    return (f"{cfg.texts.source} - daily closing prices, "
            f"{cfg.start:%d %b %Y} to {cfg.end:%d %b %Y}. "
            f"Past performance is not a reliable indicator of future performance.")


def save(fig: plt.Figure, path: str, dpi: int = 200) -> str:
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=None)
    return path
