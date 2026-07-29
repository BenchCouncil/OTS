#!/usr/bin/env python3
"""Redraw the synthetic information-boundary example in Figure-1 style.

The figure is a deterministic visual construction: two possible worlds share
the same observed history before the forecast origin, then diverge because of a
world state that is outside the closed input.  The closed-world forecaster sees
only the shared input and therefore emits one branch-agnostic prediction.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
OUT_STEM = "fig_information_boundary_refined"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "legend.fontsize": 6.1,
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


def smoothstep(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def build_curves() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin = 336.0
    x_hist = np.linspace(0.0, origin, 337)
    x_fut = np.linspace(origin, 520.0, 185)

    # A compact synthetic history that visually echoes the ETTh1 example in
    # Figure 1 while remaining deterministic and independent of any dataset.
    y_hist = (
        -1.42
        + 0.15 * np.sin(0.040 * x_hist - 0.35)
        + 0.075 * np.sin(0.115 * x_hist + 1.25)
        + 0.035 * np.sin(0.255 * x_hist - 0.55)
        - 0.00040 * x_hist
    )
    y0 = float(y_hist[-1])

    s = x_fut - origin
    gate = smoothstep(s / 20.0)
    shared_osc = 0.060 * np.sin(0.150 * s + 0.35) + 0.030 * np.sin(0.47 * s)

    world_up = y0 + gate * (
        0.0028 * s
        + 0.040 * np.sin(0.045 * s + 0.5)
        + shared_osc
    )
    world_down = y0 + gate * (
        -0.0021 * s
        + 0.050 * np.sin(0.050 * s + 2.3)
        - 0.85 * shared_osc
    )
    closed_pred = y0 + gate * (
        -0.00010 * s
        + 0.082 * np.sin(0.245 * s + 0.15)
        + 0.012 * np.sin(0.70 * s)
    )
    return x_hist, y_hist, x_fut, world_up, world_down, closed_pred


def make_figure() -> None:
    set_style()

    origin = 336.0
    x_hist, y_hist, x_fut, world_up, world_down, closed_pred = build_curves()
    xmax = float(x_fut[-1])
    y0 = float(y_hist[-1])

    history_blue = "#4C6475"
    world1_blue = "#0072B2"
    world2_purple = "#7B3294"
    orange = "#D95F02"
    gray = "#5F6B73"

    fig, ax = plt.subplots(figsize=(3.35, 2.38))
    ax.axvspan(0, origin, color="#EEF2F5", alpha=0.62, linewidth=0)
    ax.axvspan(origin, xmax, color="#FFF4E8", alpha=0.70, linewidth=0)
    ax.axvline(origin, color=gray, lw=0.85, ls="--", zorder=1)

    ax.plot(x_hist, y_hist, color=history_blue, lw=1.55, label=r"Shared history $H_t$", zorder=4)
    ax.plot(x_fut, world_up, color=world1_blue, lw=1.55, label="Real world 1", zorder=4)
    ax.plot(x_fut, world_down, color=world2_purple, lw=1.55, label="Real world 2", zorder=4)
    ax.plot(x_fut, closed_pred, color=orange, lw=1.55, label="Closed-world pred.", zorder=5)
    ax.scatter([origin], [y0], s=18, facecolor="white", edgecolor=gray, linewidth=0.85, zorder=6)

    ax.set_xlim(0, 555)
    ax.set_ylim(-2.02, -0.74)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Synthetic value")
    ax.set_title("Same history, different futures", pad=4, fontsize=8.8)
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper left",
        ncol=1,
        labelspacing=0.17,
        handlelength=1.65,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderpad=0.22,
    )

    ax.text(
        origin / 2,
        -1.94,
        "observed context",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=gray,
    )
    ax.text(
        origin + (xmax - origin) / 2,
        -1.94,
        "forecast horizon",
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=gray,
    )
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.pdf")
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(f"Saved {OUT_STEM}.pdf/png to {FIGURE_DIR}")
