#!/usr/bin/env python3
"""Generate paper figures for the paired-future open-world experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = Path(__file__).resolve().parent
RUNS_CSV = (
    ROOT
    / "open_world_experiments"
    / "server_results_20260720"
    / "results_full"
    / "all_runs.csv"
)

DATASETS = ["ETTm1", "weather", "ETTh2"]
CAPACITIES = [32, 128, 512]
INFO_CONDITIONS = ["info0", "info50", "info100"]
INFO_LABELS = {"info0": 0, "info50": 50, "info100": 100}
CAPACITY_COLORS = {32: "#0072B2", 128: "#009E73", 512: "#D55E00"}
CONTROL_ORDER = ["info0", "shuffled", "placebo", "info50", "info100"]
CONTROL_LABELS = ["History", "Shuffled $Z$", "Time placebo", "50% valid $Z$", "100% valid $Z$"]
CONTROL_COLORS = ["#B0BEC5", "#8C8C8C", "#56B4E9", "#E69F00", "#D55E00"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 4.5,
        }
    )


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURE_DIR / f"{stem}.pdf")
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def information_capacity_figure(runs: pd.DataFrame) -> None:
    subset = runs[
        (runs["architecture"] == "mlp")
        & runs["capacity"].isin(CAPACITIES)
        & runs["condition"].isin(INFO_CONDITIONS)
    ].copy()

    fig, axes = plt.subplots(
        2, 3, figsize=(6.75, 4.35), sharey="row", constrained_layout=True
    )
    markers = {32: "o", 128: "s", 512: "^"}

    for col, dataset in enumerate(DATASETS):
        dataset_runs = subset[subset["dataset"] == dataset]
        top = axes[0, col]
        bottom = axes[1, col]

        for capacity in CAPACITIES:
            cap_runs = dataset_runs[dataset_runs["capacity"] == capacity]
            means = []
            stds = []
            for condition in INFO_CONDITIONS:
                values = cap_runs.loc[cap_runs["condition"] == condition, "mse"].to_numpy()
                means.append(values.mean())
                stds.append(values.std(ddof=1))
            top.errorbar(
                [0, 50, 100],
                means,
                yerr=stds,
                color=CAPACITY_COLORS[capacity],
                marker=markers[capacity],
                capsize=2.5,
                label=f"Width {capacity}",
            )

        for condition, label, color, marker in [
            ("info0", "History only", "#7F8C8D", "o"),
            ("info100", "History + valid $Z$", "#D55E00", "s"),
        ]:
            condition_runs = dataset_runs[dataset_runs["condition"] == condition]
            points = (
                condition_runs.groupby("capacity", as_index=False)
                .agg(
                    parameter_count=("parameter_count", "mean"),
                    mse=("mse", "mean"),
                    mse_std=("mse", "std"),
                )
                .sort_values("capacity")
            )
            bottom.errorbar(
                points["parameter_count"],
                points["mse"],
                yerr=points["mse_std"],
                color=color,
                marker=marker,
                capsize=2.5,
                label=label,
            )

        top.set_title(dataset)
        top.set_xticks([0, 50, 100])
        top.set_xlabel("Valid information coverage (%)")
        bottom.set_xscale("log")
        bottom.set_xlabel("Trainable parameters (log scale)")

        if col == 0:
            top.set_ylabel("Normalized MSE")
            bottom.set_ylabel("Normalized MSE")
        if col > 0:
            top.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)

    axes[0, 0].legend(loc="upper right", handlelength=2.2)
    axes[1, 0].legend(loc="upper right", handlelength=2.2)
    axes[0, 0].text(-0.24, 1.08, "(a)", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[1, 0].text(-0.24, 1.08, "(b)", transform=axes[1, 0].transAxes, fontweight="bold")
    save_both(fig, "fig_open_world_information_capacity")


def negative_control_figure(runs: pd.DataFrame) -> None:
    subset = runs[
        (runs["architecture"] == "mlp")
        & (runs["capacity"] == 128)
        & runs["condition"].isin(CONTROL_ORDER)
    ].copy()

    fig, axes = plt.subplots(
        1, 3, figsize=(6.75, 2.55), sharey=True, constrained_layout=True
    )
    for col, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        dataset_runs = subset[subset["dataset"] == dataset]
        means = []
        stds = []
        for condition in CONTROL_ORDER:
            values = dataset_runs.loc[dataset_runs["condition"] == condition, "mse"].to_numpy()
            means.append(values.mean())
            stds.append(values.std(ddof=1))

        x = np.arange(len(CONTROL_ORDER))
        ax.bar(
            x,
            means,
            yerr=stds,
            color=CONTROL_COLORS,
            width=0.72,
            capsize=2.5,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_title(dataset)
        ax.set_xticks(x)
        ax.set_xticklabels(["H", "Shuffle", "Time", "50% $Z$", "100% $Z$"], rotation=24, ha="right")
        if col == 0:
            ax.set_ylabel("Normalized MSE")
        else:
            ax.tick_params(labelleft=False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in CONTROL_COLORS]
    fig.legend(
        handles,
        CONTROL_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.09),
        ncol=5,
        columnspacing=1.1,
        handlelength=1.4,
    )
    save_both(fig, "fig_open_world_negative_controls")


def main() -> None:
    set_style()
    runs = pd.read_csv(RUNS_CSV)
    if len(runs) != 117 or runs["run_id"].nunique() != 117:
        raise RuntimeError("Expected 117 unique completed runs")
    if not (runs["status"] == "completed").all():
        raise RuntimeError("At least one run is incomplete")
    information_capacity_figure(runs)
    negative_control_figure(runs)
    print(f"Saved paper figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
