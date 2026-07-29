#!/usr/bin/env python3
"""Generate a single-column combined positioning figure.

Panel (a) redraws the original ETTh1 PatchTST example. Panel (b) redraws the
information-boundary synthetic example. The two panels share one y-axis label:
``value``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gen_fig_information_boundary_refined import build_curves
from gen_fig_position_etth1_refined import SOURCE, digitize_curve


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = "fig_position_combined"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 5.8,
            "axes.titlesize": 6.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.0,
            "legend.fontsize": 4.9,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def draw_etth1_panel(ax: plt.Axes) -> None:
    rgb = np.asarray(Image.open(SOURCE).convert("RGB"))
    x_pred, y_pred = digitize_curve(rgb, "orange")
    x_true, y_true = digitize_curve(rgb, "blue")

    origin = 336
    xmax = max(x_pred.max(), x_true.max())
    blue = "#1F77B4"
    orange = "#D95F02"
    gray = "#5F6B73"

    ax.axvspan(0, origin, color="#EEF2F5", alpha=0.62, linewidth=0)
    ax.axvspan(origin, xmax, color="#FFF4E8", alpha=0.70, linewidth=0)
    ax.axvline(origin, color=gray, lw=0.65, ls="--", zorder=1)

    context = x_pred <= origin
    forecast = x_pred >= origin
    future = x_true >= origin - 4
    ax.plot(x_pred[context], y_pred[context], color=blue, lw=1.05, label="GroundTruth", zorder=4)
    ax.plot(x_true[future], y_true[future], color=blue, lw=1.05, zorder=4)
    ax.plot(x_pred[forecast], y_pred[forecast], color=orange, lw=1.05, label="Prediction", zorder=3)

    ax.set_xlim(0, xmax)
    ax.set_ylim(-2.02, -0.74)
    ax.set_title("(a) Closed-world failure", pad=2.2)
    ax.set_xlabel("Time step", labelpad=1.2)
    ax.legend(
        loc="upper left",
        handlelength=1.45,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderpad=0.18,
        labelspacing=0.18,
    )
    ax.text(origin / 2, -1.93, "context", ha="center", va="bottom", fontsize=4.7, color=gray)
    ax.text(origin + (xmax - origin) / 2, -1.93, "horizon", ha="center", va="bottom", fontsize=4.7, color=gray)


def draw_boundary_panel(ax: plt.Axes) -> None:
    origin = 336.0
    x_hist, y_hist, x_fut, world_up, world_down, closed_pred = build_curves()
    xmax = float(x_fut[-1])
    y0 = float(y_hist[-1])

    history_blue = "#4C6475"
    world1_blue = "#0072B2"
    world2_purple = "#7B3294"
    orange = "#D95F02"
    gray = "#5F6B73"

    ax.axvspan(0, origin, color="#EEF2F5", alpha=0.62, linewidth=0)
    ax.axvspan(origin, xmax, color="#FFF4E8", alpha=0.70, linewidth=0)
    ax.axvline(origin, color=gray, lw=0.65, ls="--", zorder=1)

    ax.plot(x_hist, y_hist, color=history_blue, lw=1.05, label=r"History $H_t$", zorder=4)
    ax.plot(x_fut, world_up, color=world1_blue, lw=1.05, label="Real world 1", zorder=4)
    ax.plot(x_fut, world_down, color=world2_purple, lw=1.05, label="Real world 2", zorder=4)
    ax.plot(x_fut, closed_pred, color=orange, lw=1.05, label="Closed pred.", zorder=5)
    ax.scatter([origin], [y0], s=9, facecolor="white", edgecolor=gray, linewidth=0.65, zorder=6)

    ax.set_xlim(0, 540)
    ax.set_ylim(-2.02, -0.74)
    ax.set_title("(b) Information boundary", pad=2.2)
    ax.set_xlabel("Time step", labelpad=1.2)
    ax.legend(
        loc="upper left",
        handlelength=1.25,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderpad=0.16,
        labelspacing=0.12,
    )
    ax.text(origin / 2, -1.93, "same history", ha="center", va="bottom", fontsize=4.7, color=gray)
    ax.text(origin + (xmax - origin) / 2, -1.93, "branches", ha="center", va="bottom", fontsize=4.7, color=gray)


def make_figure() -> None:
    set_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(3.35, 1.85),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.08},
    )
    draw_etth1_panel(axes[0])
    draw_boundary_panel(axes[1])
    axes[0].set_ylabel("value")
    axes[1].tick_params(labelleft=False)

    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.pdf")
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.png", dpi=300)
    fig.savefig(FIGURE_DIR / "Figure1.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(f"Saved {OUT_STEM}.pdf/png to {FIGURE_DIR}")
