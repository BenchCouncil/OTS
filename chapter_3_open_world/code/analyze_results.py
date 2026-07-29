#!/usr/bin/env python3
"""Validate paired-world results and generate publication-ready figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASETS = ["ETTm1", "weather", "ETTh2"]
CAPACITIES = [32, 128, 512]
INFO_CONDITIONS = ["info0", "info50", "info100"]


def block_bootstrap_mean(
    values: np.ndarray,
    block_length: int,
    draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    block_length = max(1, min(block_length, n))
    n_blocks = math.ceil(n / block_length)
    maximum_start = max(1, n - block_length + 1)
    means = np.empty(draws, dtype=np.float64)
    offsets = np.arange(block_length)
    for draw in range(draws):
        starts = rng.integers(0, maximum_start, size=n_blocks)
        indices = (starts[:, None] + offsets[None, :]).reshape(-1)[:n]
        means[draw] = values[indices].mean()
    return means


def confirmatory_block_bootstrap(result_root: Path, runs: pd.DataFrame, draws: int) -> pd.DataFrame:
    groups = (
        runs[runs["condition"] == "info0"][["dataset", "architecture", "capacity"]]
        .drop_duplicates()
        .sort_values(["dataset", "architecture", "capacity"])
    )
    rows = []
    alpha_bonferroni = 0.05 / len(groups)
    for _, group in groups.iterrows():
        dataset = str(group["dataset"])
        architecture = str(group["architecture"])
        capacity = int(group["capacity"])
        seed_frames = []
        for seed in sorted(runs["seed"].unique()):
            history_id = f"{dataset}__{architecture}{capacity}__info0__seed{seed}"
            open_id = f"{dataset}__{architecture}{capacity}__info100__seed{seed}"
            history = pd.read_csv(result_root / "runs" / history_id / "pair_losses.csv")
            opened = pd.read_csv(result_root / "runs" / open_id / "pair_losses.csv")
            merged = history[["base_start", "pair_mse"]].merge(
                opened[["base_start", "pair_mse"]],
                on="base_start",
                suffixes=("_history", "_open"),
                validate="one_to_one",
            )
            merged["seed"] = int(seed)
            merged["delta"] = merged["pair_mse_history"] - merged["pair_mse_open"]
            seed_frames.append(merged[["base_start", "seed", "delta"]])
        combined = pd.concat(seed_frames, ignore_index=True)
        by_window = combined.groupby("base_start", as_index=False)["delta"].mean().sort_values("base_start")
        starts = by_window["base_start"].to_numpy()
        median_spacing = float(np.median(np.diff(starts))) if len(starts) > 1 else 1.0
        block_length = int(math.ceil((96 + 96) / max(1.0, median_spacing)))
        values = by_window["delta"].to_numpy()
        seed_payload = f"{dataset}|{architecture}|{capacity}|block-bootstrap".encode("utf-8")
        rng_seed = int.from_bytes(hashlib.blake2b(seed_payload, digest_size=4).digest(), "little")
        sampled = block_bootstrap_mean(values, block_length, draws, np.random.default_rng(rng_seed))
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "capacity": capacity,
                "seeds": int(combined["seed"].nunique()),
                "n_unique_windows": len(values),
                "median_start_spacing": median_spacing,
                "block_length_pairs": block_length,
                "delta_open": float(values.mean()),
                "ci95_low": float(np.quantile(sampled, 0.025)),
                "ci95_high": float(np.quantile(sampled, 0.975)),
                "bonferroni_family": len(groups),
                "ci_bonf_low": float(np.quantile(sampled, alpha_bonferroni / 2)),
                "ci_bonf_high": float(np.quantile(sampled, 1 - alpha_bonferroni / 2)),
            }
        )
    return pd.DataFrame(rows)


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
        }
    )


def make_information_capacity_figure(runs: pd.DataFrame, figure_dir: Path) -> None:
    publication_style()
    colors = {32: "#56B4E9", 128: "#009E73", 512: "#E69F00"}
    info_colors = {"info0": "#8C8C8C", "info100": "#D55E00"}
    figure, axes = plt.subplots(2, 3, figsize=(7.0, 4.7), constrained_layout=True)
    for column, dataset in enumerate(DATASETS):
        subset = runs[(runs["dataset"] == dataset) & (runs["architecture"] == "mlp")]
        top = axes[0, column]
        for capacity in CAPACITIES:
            means = []
            stds = []
            for condition in INFO_CONDITIONS:
                values = subset[(subset["capacity"] == capacity) & (subset["condition"] == condition)]["mse"]
                means.append(values.mean())
                stds.append(values.std(ddof=1))
            top.errorbar(
                [0, 50, 100],
                means,
                yerr=stds,
                marker="o",
                capsize=2.5,
                color=colors[capacity],
                label=f"width {capacity}",
            )
        top.set_title(dataset)
        top.set_xlabel("Pairs with event information (%)")
        if column == 0:
            top.set_ylabel("Normalized MSE")
        top.set_xticks([0, 50, 100])

        bottom = axes[1, column]
        for condition, marker, label in (
            ("info0", "o", "History only"),
            ("info100", "s", "History + valid Z"),
        ):
            parameter_counts = []
            means = []
            stds = []
            for capacity in CAPACITIES:
                values = subset[(subset["capacity"] == capacity) & (subset["condition"] == condition)]
                parameter_counts.append(values["parameter_count"].mean())
                means.append(values["mse"].mean())
                stds.append(values["mse"].std(ddof=1))
            bottom.errorbar(
                parameter_counts,
                means,
                yerr=stds,
                marker=marker,
                capsize=2.5,
                color=info_colors[condition],
                label=label,
            )
        bottom.set_xscale("log")
        bottom.set_xlabel("Trainable parameters (log scale)")
        if column == 0:
            bottom.set_ylabel("Normalized MSE")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.025))
    handles, labels = axes[1, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.035))
    figure.savefig(figure_dir / "fig_information_vs_capacity.pdf")
    figure.savefig(figure_dir / "fig_information_vs_capacity.png", dpi=300)
    plt.close(figure)


def make_negative_control_figure(runs: pd.DataFrame, figure_dir: Path) -> None:
    publication_style()
    conditions = ["info0", "shuffled", "placebo", "info50", "info100"]
    labels = ["History only", "Shuffled Z", "Time placebo", "50% valid Z", "100% valid Z"]
    colors = ["#B0BEC5", "#8C8C8C", "#0072B2", "#E69F00", "#D55E00"]
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), constrained_layout=True)
    for axis, dataset in zip(axes, DATASETS):
        subset = runs[
            (runs["dataset"] == dataset)
            & (runs["architecture"] == "mlp")
            & (runs["capacity"] == 128)
        ]
        means = [subset[subset["condition"] == condition]["mse"].mean() for condition in conditions]
        stds = [subset[subset["condition"] == condition]["mse"].std(ddof=1) for condition in conditions]
        positions = np.arange(len(conditions))
        axis.bar(
            positions,
            means,
            yerr=stds,
            capsize=2.5,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.set_title(dataset)
        axis.set_xticks(positions)
        axis.set_xticklabels(["H", "Shuffle", "Time", "50% Z", "100% Z"], rotation=25, ha="right")
        if axis is axes[0]:
            axis.set_ylabel("Normalized MSE")
    from matplotlib.patches import Patch

    figure.legend(
        [Patch(facecolor=color, edgecolor="none") for color in colors],
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.08),
    )
    figure.savefig(figure_dir / "fig_negative_controls.pdf")
    figure.savefig(figure_dir / "fig_negative_controls.png", dpi=300)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.result_root / "all_runs.csv")
    if len(runs) != 117 or runs["run_id"].nunique() != 117:
        raise RuntimeError("Expected 117 unique completed runs")
    if not (runs["status"] == "completed").all():
        raise RuntimeError("At least one run is not completed")

    information = runs[
        (runs["architecture"] == "mlp") & runs["condition"].isin(INFO_CONDITIONS)
    ].pivot(index=["dataset", "capacity", "seed"], columns="condition", values="mse")
    information["strictly_monotone"] = (
        (information["info100"] < information["info50"])
        & (information["info50"] < information["info0"])
    )
    information.reset_index().to_csv(args.output_dir / "information_monotonicity.csv", index=False)

    small_open = runs[
        (runs["architecture"] == "mlp")
        & (runs["capacity"] == 32)
        & (runs["condition"] == "info100")
    ][["dataset", "seed", "mse"]].rename(columns={"mse": "small_open_mse"})
    large_history = runs[
        (runs["architecture"] == "mlp")
        & (runs["capacity"] == 512)
        & (runs["condition"] == "info0")
    ][["dataset", "seed", "mse"]].rename(columns={"mse": "large_history_mse"})
    crossover = small_open.merge(large_history, on=["dataset", "seed"], validate="one_to_one")
    crossover["relative_reduction"] = 1 - crossover["small_open_mse"] / crossover["large_history_mse"]
    crossover.to_csv(args.output_dir / "small_open_vs_large_history.csv", index=False)
    crossover.groupby("dataset", as_index=False).agg(
        seeds=("seed", "nunique"),
        small_open_mse=("small_open_mse", "mean"),
        large_history_mse=("large_history_mse", "mean"),
        relative_reduction_mean=("relative_reduction", "mean"),
        relative_reduction_std=("relative_reduction", "std"),
    ).to_csv(args.output_dir / "small_open_vs_large_history_summary.csv", index=False)

    controls = runs[
        (runs["architecture"] == "mlp")
        & (runs["capacity"] == 128)
        & runs["condition"].isin(["info0", "shuffled", "placebo", "info50", "info100"])
    ]
    controls.groupby(["dataset", "condition"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        separation_mse_mean=("separation_mse", "mean"),
    ).to_csv(args.output_dir / "negative_control_summary.csv", index=False)

    block = confirmatory_block_bootstrap(args.result_root, runs, args.bootstrap_draws)
    block.to_csv(args.output_dir / "confirmatory_block_bootstrap.csv", index=False)

    make_information_capacity_figure(runs, figure_dir)
    make_negative_control_figure(runs, figure_dir)

    summary = {
        "completed_unique_runs": int(runs["run_id"].nunique()),
        "datasets": sorted(runs["dataset"].unique().tolist()),
        "seeds": sorted(int(value) for value in runs["seed"].unique()),
        "strictly_monotone_information_curves": int(information["strictly_monotone"].sum()),
        "total_information_curves": int(len(information)),
        "small_open_beats_large_history": int(
            (crossover["small_open_mse"] < crossover["large_history_mse"]).sum()
        ),
        "small_open_large_history_comparisons": int(len(crossover)),
        "confirmatory_comparisons": int(len(block)),
        "bonferroni_block_ci_excluding_zero": int((block["ci_bonf_low"] > 0).sum()),
    }
    with (args.output_dir / "analysis_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
