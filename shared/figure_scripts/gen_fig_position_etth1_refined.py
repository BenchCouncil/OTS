#!/usr/bin/env python3
"""Redraw the ETTh1 PatchTST positioning example from the old plot image.

The old figure is the only available source for this qualitative case study.
This script digitizes the two colored curves from that image and redraws them
with publication-style layout, labels, and annotations.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


FIGURE_DIR = Path(__file__).resolve().parent
SOURCE = FIGURE_DIR / "fig_position_etth1_patchtst.jpg"
OUT_STEM = "fig_position_etth1_refined"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.8,
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


def smooth_nan(y: np.ndarray, window: int = 3) -> np.ndarray:
    """Small median smoother that preserves NaNs at the boundaries."""
    if window <= 1:
        return y
    out = y.copy()
    radius = window // 2
    for i in range(len(y)):
        lo = max(0, i - radius)
        hi = min(len(y), i + radius + 1)
        vals = y[lo:hi]
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[i] = np.median(vals)
    return out


def digitize_curve(rgb: np.ndarray, color: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract a colored curve from the legacy Matplotlib JPEG."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    # Plot area detected from the black frame in the original figure.
    left, right = 139, 1331
    top, bottom = 28, 915
    in_axes = (
        (np.arange(rgb.shape[1])[None, :] >= left)
        & (np.arange(rgb.shape[1])[None, :] <= right)
        & (np.arange(rgb.shape[0])[:, None] >= top)
        & (np.arange(rgb.shape[0])[:, None] <= bottom)
    )

    if color == "blue":
        mask = (b > 105) & (g > 55) & (g < 185) & (r < 95)
        # Remove the legend sample at the top-left.
        mask &= ~((np.arange(rgb.shape[1])[None, :] < 520) & (np.arange(rgb.shape[0])[:, None] < 155))
    elif color == "orange":
        mask = (r > 155) & (g > 65) & (g < 190) & (b < 95)
        # Remove the legend sample at the top-left.
        mask &= ~((np.arange(rgb.shape[1])[None, :] < 520) & (np.arange(rgb.shape[0])[:, None] < 175))
    else:
        raise ValueError(color)

    mask &= in_axes
    yy, xx = np.where(mask)
    if len(xx) == 0:
        raise RuntimeError(f"No pixels found for {color}")

    # Pixel-to-data calibration from the original axes:
    # x=0 at the first data point, x=500 at the old 500 tick;
    # y=-2.0 at the bottom spine and y=-0.75 at the top spine.
    x_data = (xx - 193.0) * (500.0 / (1221.0 - 193.0))
    y_data = -2.0 + (bottom - yy) * (1.25 / (bottom - top))

    # Aggregate thick JPEG line pixels to one y per integer step.
    x_grid = np.arange(int(np.floor(np.nanmin(x_data))), int(np.nanmax(x_data)) + 1)
    y_grid = np.full_like(x_grid, np.nan, dtype=float)
    for i, x0 in enumerate(x_grid):
        vals = y_data[(x_data >= x0 - 0.5) & (x_data < x0 + 0.5)]
        if len(vals):
            y_grid[i] = np.median(vals)

    good = np.isfinite(y_grid)
    if good.sum() < 5:
        raise RuntimeError(f"Too few points found for {color}")
    # Fill very small gaps caused by JPEG antialiasing.
    y_grid = np.interp(x_grid, x_grid[good], y_grid[good])
    y_grid = smooth_nan(y_grid, window=3)
    return x_grid.astype(float), y_grid


def make_figure() -> None:
    set_style()
    rgb = np.asarray(Image.open(SOURCE).convert("RGB"))

    x_pred, y_pred = digitize_curve(rgb, "orange")
    x_true, y_true = digitize_curve(rgb, "blue")

    # The old plot uses a long context-like left segment and a future segment
    # beginning at approximately 336.
    origin = 336
    xmax = max(x_pred.max(), x_true.max())

    blue = "#1F77B4"
    orange = "#D95F02"
    gray = "#5F6B73"

    fig, ax = plt.subplots(figsize=(3.35, 2.38))
    ax.axvspan(0, origin, color="#EEF2F5", alpha=0.62, linewidth=0)
    ax.axvspan(origin, xmax, color="#FFF4E8", alpha=0.70, linewidth=0)
    ax.axvline(origin, color=gray, lw=0.85, ls="--", zorder=1)

    # In the legacy image the left segment was drawn in the prediction color,
    # although it represents the observed/history side of the example.  Redraw
    # the context as GroundTruth and reserve orange for the forecast horizon.
    context = x_pred <= origin
    forecast = x_pred >= origin
    future = x_true >= origin - 4

    ax.plot(x_pred[context], y_pred[context], color=blue, lw=1.55, label="GroundTruth", zorder=4)
    ax.plot(x_true[future], y_true[future], color=blue, lw=1.55, zorder=4)
    ax.plot(x_pred[forecast], y_pred[forecast], color=orange, lw=1.55, label="Prediction", zorder=3)

    ax.set_xlim(0, xmax)
    ax.set_ylim(-2.02, -0.74)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Normalized OT")
    ax.set_title("ETTh1: PatchTST + Patch Loss", pad=4, fontsize=8.8)
    ax.legend(
        loc="upper left",
        handlelength=2.2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        borderpad=0.25,
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
    ax.text(
        origin + 5,
        -1.86,
        "forecast origin",
        ha="left",
        va="bottom",
        fontsize=6.6,
        color=gray,
    )
    ax.annotate(
        "periodic forecast",
        xy=(452, np.interp(452, x_pred, y_pred)),
        xytext=(396, -1.78),
        arrowprops=dict(arrowstyle="->", lw=0.75, color=orange),
        fontsize=6.8,
        color=orange,
    )
    ax.annotate(
        "state shift",
        xy=(482, np.interp(482, x_true, y_true)),
        xytext=(432, -0.96),
        arrowprops=dict(arrowstyle="->", lw=0.75, color=blue),
        fontsize=6.8,
        color=blue,
    )

    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.pdf")
    fig.savefig(FIGURE_DIR / f"{OUT_STEM}.png", dpi=300)
    # TeXPage's latest cloud draft introduced this alias. Keep it synchronized
    # with the refined Figure 1 to avoid two visually different Figure 1 files.
    fig.savefig(FIGURE_DIR / "Figure1.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print(f"Saved {OUT_STEM}.pdf/png to {FIGURE_DIR}")
