#!/usr/bin/env python3
"""Generate the post-review evidence figure from revision-control CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "gray": "#8C8C8C",
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 5.0,
            "axes.titlesize": 5.6,
            "axes.titleweight": "bold",
            "axes.labelsize": 5.0,
            "xtick.labelsize": 4.8,
            "ytick.labelsize": 4.8,
            "legend.fontsize": 4.7,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.linestyle": "-",
            "lines.linewidth": 1.05,
            "lines.markersize": 3.1,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "open_world_experiments"
        / "revision_controls_20260721",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    direction = pd.read_csv(args.result_dir / "synthetic_direction_calibration.csv")
    planned = pd.read_csv(args.result_dir / "planned_information_runs.csv")
    probability = pd.read_csv(args.result_dir / "probabilistic_branch_scores.csv")

    style()
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(3.35, 1.35),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [0.95, 1.42, 1.05], "wspace": 0.18},
    )

    # (a) Calibration: a reversible negative control should concentrate at 1;
    # an irreversible non-Gaussian MA process is a positive control.
    axis = axes[0]
    process_order = ["reversible_gaussian_ma", "irreversible_nongaussian_ma"]
    process_labels = ["Rev.\nMA", "Irrev.\nMA"]
    process_colors = [COLORS["sky"], COLORS["vermillion"]]
    rng = np.random.default_rng(221)
    for position, (process, label, color) in enumerate(
        zip(process_order, process_labels, process_colors)
    ):
        values = direction[direction["process"] == process]["reverse_forward_ratio"].to_numpy()
        jitter = rng.normal(0.0, 0.035, size=len(values))
        axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=8,
            facecolor=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
            zorder=3,
        )
        axis.errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1),
            marker="D",
            color="#222222",
            capsize=2.0,
            linewidth=0.8,
            markersize=2.8,
            zorder=4,
        )
    axis.axhline(1.0, color="#444444", linestyle="--", linewidth=0.65)
    axis.set_xticks([0, 1], process_labels)
    axis.set_ylabel(r"$\rho_{\mathrm{rev}}$")
    axis.set_title("(a) Direction")
    axis.set_ylim(0.885, 1.025)

    # (b) Non-oracle forecast-time information. Normalize within dataset to the
    # history-only mean so cross-dataset raw MSE scales are not mixed.
    axis = axes[1]
    condition_order = ["history", "delayed50", "noisy", "clean", "latent_oracle_upper_bound"]
    condition_labels = ["Hist", "D50", "Noisy", "Plan", "Oracle"]
    condition_colors = [
        COLORS["gray"],
        COLORS["orange"],
        COLORS["sky"],
        COLORS["green"],
        COLORS["purple"],
    ]
    datasets = ["ETTm1", "weather", "ETTh2"]
    markers = ["o", "s", "^"]
    dataset_colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    x = np.arange(len(condition_order))
    for dataset, marker, color in zip(datasets, markers, dataset_colors):
        subset = planned[planned["dataset"] == dataset]
        history = subset[subset["condition"] == "history"]["mse"].mean()
        means = []
        stds = []
        for condition in condition_order:
            values = subset[subset["condition"] == condition]["mse"] / history
            means.append(values.mean())
            stds.append(values.std(ddof=1))
        axis.errorbar(
            x,
            means,
            yerr=stds,
            marker=marker,
            capsize=1.5,
            color=color,
            markerfacecolor="white",
            markeredgewidth=0.75,
            zorder=4,
        )
        axis.text(
            x[-1] + 0.13,
            means[-1],
            dataset,
            color=color,
            fontsize=4.8,
            va="center",
            ha="left",
            clip_on=False,
        )
    for position, color in zip(x, condition_colors):
        axis.axvspan(position - 0.43, position + 0.43, color=color, alpha=0.10, linewidth=0)
    axis.axhline(1.0, color="#555555", linestyle="--", linewidth=0.62)
    axis.set_xticks(x, condition_labels, rotation=28, ha="right")
    axis.set_xlim(-0.5, len(condition_order) - 0.18)
    axis.set_ylabel("MSE / Hist")
    axis.set_title("(b) Plan info")

    # (c) Proper multivariate energy score on intervention residuals. This panel
    # recognizes the valid history-only probabilistic solution explicitly.
    axis = axes[2]
    grouped = probability.groupby("dataset", as_index=False).agg(
        history_mean=("history_mixture_energy_score", "mean"),
        history_std=("history_mixture_energy_score", "std"),
        plan_mean=("plan_conditional_energy_score", "mean"),
        plan_std=("plan_conditional_energy_score", "std"),
    )
    positions = np.arange(len(datasets))
    width = 0.34
    grouped = grouped.set_index("dataset").loc[datasets].reset_index()
    axis.bar(
        positions - width / 2,
        grouped["history_mean"],
        width,
        yerr=grouped["history_std"],
        capsize=1.5,
        color=COLORS["gray"],
        edgecolor="white",
        linewidth=0.5,
        label="History mixture",
    )
    axis.bar(
        positions + width / 2,
        grouped["plan_mean"],
        width,
        yerr=grouped["plan_std"],
        capsize=1.5,
        color=COLORS["green"],
        edgecolor="white",
        linewidth=0.5,
        hatch="///",
        label="Plan-conditioned",
    )
    axis.set_xticks(positions, datasets, rotation=18)
    axis.set_ylabel("Energy")
    axis.set_title("(c) Prob. score")
    axis.set_ylim(0.0, 0.245)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=1,
        handlelength=1.0,
        borderaxespad=0.0,
    )

    pdf = args.output_dir / "fig_revision_evidence.pdf"
    png = args.output_dir / "fig_revision_evidence.png"
    figure.savefig(pdf)
    figure.savefig(png, dpi=300)
    plt.close(figure)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
