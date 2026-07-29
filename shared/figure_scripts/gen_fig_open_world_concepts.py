#!/usr/bin/env python3
"""Generate the two conceptual figures for the open-world position paper."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


FIGURE_DIR = Path(__file__).resolve().parent

BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7A5195"
GRAY = "#5F6B73"
LIGHT = "#F4F6F8"
DARK = "#1F2933"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def box(ax, xy, width, height, text, *, fc="white", ec=GRAY, lw=0.9,
        fontsize=8.0, weight="normal", radius=0.02, linestyle="-"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
    )
    return patch


def arrow(ax, start, end, *, color=GRAY, lw=1.1, style="-|>", mutation=8,
          connectionstyle="arc3,rad=0"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(patch)
    return patch


def wave(ax, x0, x1, y, *, color=BLUE, amp=0.025, phase=0.0, lw=1.35,
         trend=0.0, linestyle="-"):
    x = np.linspace(x0, x1, 120)
    span = max(x1 - x0, 1e-6)
    yy = y + amp * np.sin(5.2 * np.pi * (x - x0) / span + phase)
    yy += trend * (x - x0) / span
    ax.plot(x, yy, color=color, lw=lw, linestyle=linestyle, solid_capstyle="round")


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def research_position_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.35, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    panels = [
        (0.018, 0.300, "(a) History-only", "closed interface: historical channels", BLUE),
        (0.350, 0.300, "(b) Fixed exogenous", "larger table, still predeclared", PURPLE),
        (0.682, 0.300, "(c) Open-world interface", "auditable evidence, revisable forecast", GREEN),
    ]

    for x0, width, heading, subtitle, color in panels:
        ax.add_patch(
            FancyBboxPatch(
                (x0, 0.075),
                width,
                0.845,
                boxstyle="round,pad=0.010,rounding_size=0.022",
                facecolor="#FBFCFD",
                edgecolor="#CBD3DA",
                linewidth=0.95,
            )
        )
        ax.add_patch(Rectangle((x0, 0.075), 0.007, 0.845, facecolor=color,
                               edgecolor="none", alpha=0.95))
        ax.text(x0 + 0.018, 0.875, heading, ha="left", va="center",
                fontsize=9.1, fontweight="bold", color=DARK)
        ax.text(x0 + 0.018, 0.822, subtitle, ha="left", va="center",
                fontsize=6.8, color=GRAY)

    # Panel (a): a finite channel table mapped to one future.
    x0, width = panels[0][0], panels[0][1]
    ax.add_patch(Rectangle((x0 + 0.030, 0.385), 0.105, 0.300, facecolor=LIGHT,
                           edgecolor="#AAB5BF", linewidth=0.85))
    for k, (yy, color) in enumerate([(0.640, BLUE), (0.560, GREEN), (0.480, ORANGE), (0.410, PURPLE)]):
        wave(ax, x0 + 0.043, x0 + 0.122, yy, color=color, amp=0.014,
             phase=0.7 * k, lw=1.15)
    ax.text(x0 + 0.082, 0.720, r"history window $H$", ha="center",
            fontsize=7.2, color=DARK, fontweight="bold")
    ax.text(x0 + 0.082, 0.348, r"fixed $L\times C$ table", ha="center",
            fontsize=6.7, color=GRAY)
    box(ax, (x0 + 0.168, 0.465), 0.070, 0.120, "forecast\nmodel", fc="#EAF3F8",
        ec=BLUE, fontsize=7.0, weight="bold", radius=0.012)
    arrow(ax, (x0 + 0.138, 0.525), (x0 + 0.166, 0.525), color=GRAY, lw=1.0)
    arrow(ax, (x0 + 0.238, 0.525), (x0 + 0.270, 0.525), color=GRAY, lw=1.0)
    wave(ax, x0 + 0.272, x0 + 0.292, 0.525, color=VERMILION, amp=0.020, lw=1.45)
    ax.text(x0 + 0.203, 0.655, r"$\hat{Y}=f(H)$", ha="center", fontsize=8.0, color=DARK)
    box(ax, (x0 + 0.042, 0.172), 0.232, 0.090,
        "implicit claim:\nobserved history is sufficient",
        fc="white", ec="#D1D8DE", lw=0.75, fontsize=6.5, radius=0.010)

    # Panel (b): a larger but still predeclared boundary.
    x0, width = panels[1][0], panels[1][1]
    ax.add_patch(
        FancyBboxPatch(
            (x0 + 0.026, 0.325), 0.158, 0.390,
            boxstyle="round,pad=0.010,rounding_size=0.014",
            facecolor="#FAFBFC", edgecolor="#8996A1", linewidth=0.95,
            linestyle="--",
        )
    )
    ax.text(x0 + 0.105, 0.740, "locked schema", ha="center",
            fontsize=7.0, color=DARK, fontweight="bold")
    for k, (yy, color) in enumerate([(0.650, BLUE), (0.580, GREEN), (0.510, ORANGE)]):
        wave(ax, x0 + 0.044, x0 + 0.119, yy, color=color, amp=0.012,
             phase=0.8 * k, lw=1.05)
    box(ax, (x0 + 0.044, 0.405), 0.120, 0.048, "calendar", fc="white", ec=SKY,
        lw=0.80, fontsize=6.6, radius=0.008)
    box(ax, (x0 + 0.044, 0.348), 0.120, 0.048, "known covariates", fc="white",
        ec=PURPLE, lw=0.80, fontsize=6.4, radius=0.008)
    ax.text(x0 + 0.105, 0.282, r"$I_c=\{H,K_{\mathrm{fixed}}\}$", ha="center",
            fontsize=6.8, color=GRAY)
    box(ax, (x0 + 0.207, 0.465), 0.066, 0.120, "forecast\nmodel", fc="#EFEAF8",
        ec=PURPLE, fontsize=7.0, weight="bold", radius=0.012)
    arrow(ax, (x0 + 0.186, 0.525), (x0 + 0.206, 0.525), color=GRAY, lw=1.0)
    arrow(ax, (x0 + 0.273, 0.525), (x0 + 0.292, 0.525), color=GRAY, lw=1.0)
    wave(ax, x0 + 0.292, x0 + 0.300, 0.525, color=VERMILION, amp=0.020, lw=1.3)
    box(ax, (x0 + 0.042, 0.172), 0.232, 0.090,
        "external variables are useful,\nbut availability rules are fixed",
        fc="white", ec="#D1D8DE", lw=0.75, fontsize=6.35, radius=0.010)

    # Panel (c): an auditable, revisable information interface.
    x0, width = panels[2][0], panels[2][1]
    sources = [
        (0.024, 0.690, "events", VERMILION),
        (0.105, 0.690, "plans", ORANGE),
        (0.186, 0.690, "sensors", SKY),
        (0.024, 0.615, "text", PURPLE),
        (0.105, 0.615, "policy", GREEN),
        (0.186, 0.615, "weather", BLUE),
    ]
    for dx, yy, label, color in sources:
        box(ax, (x0 + dx, yy - 0.024), 0.066, 0.048, label, fc="white", ec=color,
            lw=0.85, fontsize=6.1, radius=0.008)
        arrow(ax, (x0 + dx + 0.033, yy - 0.025), (x0 + 0.126, 0.590),
              color=color, lw=0.65, mutation=6, connectionstyle="arc3,rad=0.10")
    box(ax, (x0 + 0.052, 0.425), 0.146, 0.170,
        "evidence\nsnapshot\n"
        r"$\tau_{\mathrm{pub}}\leq t$"
        "\nsource/version",
        fc="#FFF6DF", ec=ORANGE, fontsize=6.25, weight="bold", radius=0.012)
    box(ax, (x0 + 0.218, 0.462), 0.060, 0.096, "forecast\nmodel", fc="#E8F5F0",
        ec=GREEN, fontsize=6.7, weight="bold", radius=0.012)
    arrow(ax, (x0 + 0.198, 0.510), (x0 + 0.217, 0.510), color=ORANGE, lw=1.0)
    arrow(ax, (x0 + 0.248, 0.462), (x0 + 0.248, 0.372), color=GREEN, lw=0.95,
          mutation=7)
    wave(ax, x0 + 0.058, x0 + 0.278, 0.310, color=VERMILION, amp=0.018,
         trend=0.075, lw=1.45)
    wave(ax, x0 + 0.058, x0 + 0.278, 0.228, color=BLUE, amp=0.018,
         trend=-0.045, phase=0.8, lw=1.45)
    box(ax, (x0 + 0.045, 0.126), 0.228, 0.052,
        "revise when evidence changes",
        fc="white", ec="#D1D8DE", lw=0.75, fontsize=6.2, radius=0.010)

    save_both(fig, "fig_research_position")


def core_viewpoint_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Identical histories and a shared forecast origin.
    cutoff = 0.285
    ax.plot([cutoff, cutoff], [0.13, 0.86], color="#82909B", lw=0.85, linestyle="--")
    ax.text(cutoff, 0.90, "forecast origin", ha="center", va="center", fontsize=7.2, color=GRAY)
    for y in (0.68, 0.32):
        wave(ax, 0.035, cutoff - 0.012, y, color=BLUE, amp=0.045, phase=0.3, lw=1.7)
    ax.text(0.052, 0.80, r"identical history $H$", fontsize=8.0, color=DARK, fontweight="bold")
    ax.text(0.047, 0.55, "World 0", fontsize=7.2, color=GRAY)
    ax.text(0.047, 0.19, "World 1", fontsize=7.2, color=GRAY)

    # Two futures separate only after the cutoff.
    x = np.linspace(cutoff + 0.012, 0.485, 130)
    base = 0.68 + 0.042 * np.sin(np.linspace(0.3, 5.2 * np.pi, len(x)))
    event = 0.32 + 0.042 * np.sin(np.linspace(0.3, 5.2 * np.pi, len(x)))
    event += 0.15 * (1 - np.exp(-np.linspace(0, 4.0, len(x))))
    ax.plot(x, base, color=BLUE, lw=1.65)
    ax.plot(x, event, color=VERMILION, lw=1.65)
    box(ax, (0.315, 0.505), 0.100, 0.075, r"world state $Z$", fc="#FFF1EA", ec=VERMILION,
        fontsize=7.3, weight="bold", radius=0.010)
    arrow(ax, (0.365, 0.505), (0.405, 0.465), color=VERMILION, lw=1.0)
    ax.text(0.395, 0.76, r"$Y^{(0)}$", color=BLUE, fontsize=8.0, fontweight="bold")
    ax.text(0.430, 0.335, r"$Y^{(1)}$", color=VERMILION, fontsize=8.0, fontweight="bold")

    # Closed-world model: identical inputs imply identical outputs.
    ax.add_patch(
        FancyBboxPatch(
            (0.515, 0.535), 0.205, 0.33,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F7F8F9", edgecolor="#98A4AD", linewidth=0.9,
        )
    )
    ax.text(0.617, 0.825, "Closed world", ha="center", fontsize=8.7,
            fontweight="bold", color=DARK)
    ax.text(0.617, 0.765, r"$\hat{Y}=f(H)$", ha="center", fontsize=8.7, color=DARK)
    wave(ax, 0.545, 0.690, 0.655, color=ORANGE, amp=0.025, phase=0.5, lw=1.7)
    ax.text(0.617, 0.585, r"same input $\Rightarrow$ same forecast", ha="center",
            fontsize=7.0, color=GRAY)
    arrow(ax, (0.485, 0.68), (0.515, 0.70), color=BLUE, lw=0.9)
    arrow(ax, (0.485, 0.43), (0.515, 0.66), color=VERMILION, lw=0.9)

    # Open-world model: world information makes the branches identifiable.
    ax.add_patch(
        FancyBboxPatch(
            (0.750, 0.135), 0.225, 0.73,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            facecolor="#F5FBF8", edgecolor=GREEN, linewidth=1.05,
        )
    )
    ax.text(0.862, 0.825, "Open world", ha="center", fontsize=8.7,
            fontweight="bold", color=DARK)
    ax.text(0.862, 0.765, r"$\hat{Y}=g(H,Z)$", ha="center", fontsize=8.7, color=DARK)
    wave(ax, 0.782, 0.942, 0.615, color=BLUE, amp=0.030, phase=0.3, lw=1.65)
    x2 = np.linspace(0.782, 0.942, 120)
    y2 = 0.365 + 0.030 * np.sin(np.linspace(0.3, 5.2 * np.pi, len(x2)))
    y2 += 0.105 * (1 - np.exp(-np.linspace(0, 4.0, len(x2))))
    ax.plot(x2, y2, color=VERMILION, lw=1.65)
    ax.text(0.862, 0.235, "external information separates futures", ha="center",
            fontsize=6.7, color=GRAY)
    arrow(ax, (0.720, 0.70), (0.750, 0.70), color=GRAY)
    arrow(ax, (0.415, 0.540), (0.750, 0.445), color=VERMILION, lw=1.0,
          connectionstyle="arc3,rad=0.06")

    ax.text(0.500, 0.055,
            "The forecasting limit is set by the information boundary, not by model capacity alone.",
            ha="center", va="center", fontsize=8.0, color=DARK, fontweight="bold")
    save_both(fig, "fig_core_viewpoint")


def main() -> None:
    set_style()
    research_position_figure()
    core_viewpoint_figure()
    print(f"Saved conceptual figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
