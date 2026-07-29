"""Generate the audit-grade appendix tables from the archived result files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[3]
CHAPTER2 = RELEASE_ROOT / "chapter_2_closed_world"
CHAPTER3 = RELEASE_ROOT / "chapter_3_open_world"
OUT = CHAPTER2 / "results" / "paper_table_inputs" / "generated"
REVISION = CHAPTER3 / "results" / "revision_controls_20260729_extended"
ORACLE = (
    CHAPTER3
    / "results"
    / "server_results_20260720"
    / "results_full"
)
REVERSAL = (
    CHAPTER2
    / "results"
    / "reversal_analysis_tables"
    / "deep_ltsf_fixed_lr_metrics.csv"
)


DATASET_ORDER = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "electricity",
    "exchange_rate",
    "illness",
    "weather",
]
MODEL_ORDER = [
    "DLinear",
    "GRU",
    "PatchTST",
    "TimeFilter",
    "TimeMixer",
    "TimesNet",
    "iTransformer",
]
PLAN_ORDER = [
    "history",
    "clean",
    "noisy",
    "delayed50",
    "shuffled",
    "misleading_test",
    "latent_oracle_upper_bound",
]
PLAN_LABEL = {
    "history": "History",
    "clean": "Clean plan",
    "noisy": "Noisy plan",
    "delayed50": "50\\% delayed",
    "shuffled": "Shuffled",
    "misleading_test": "Misleading",
    "latent_oracle_upper_bound": "Latent oracle",
}
ORACLE_CONDITION_LABEL = {
    "info0": "History",
    "info50": "50\\% oracle",
    "info100": "100\\% oracle",
    "shuffled": "Shuffled",
    "placebo": "Time placebo",
}


def fmt(value: float | int, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 10_000 or abs(value) < 10 ** (-(digits + 1)):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def esc(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write(name: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def direction_table() -> None:
    df = pd.read_csv(REVISION / "synthetic_direction_calibration.csv")
    fields = ["forward_mse", "reverse_mse", "reverse_forward_ratio"]
    wide = df.pivot(index="seed", columns="process", values=fields)
    rows = []
    for seed in sorted(wide.index):
        g = [wide.loc[seed, (f, "reversible_gaussian_ma")] for f in fields]
        n = [wide.loc[seed, (f, "irreversible_nongaussian_ma")] for f in fields]
        rows.append(
            f"{seed} & {fmt(g[0])} & {fmt(g[1])} & {fmt(g[2])} "
            f"& {fmt(n[0])} & {fmt(n[1])} & {fmt(n[2])} \\\\"
        )
    content = r"""
\begin{table*}[t]
\centering
\caption{Full ten-seed direction calibration results. $F$ and $R$ denote forward and reverse test MSE.}
\label{tab:direction-calibration-seeds}
\small
\begin{tabular}{@{}rccc@{\qquad}ccc@{}}
\toprule
& \multicolumn{3}{c}{Gaussian MA} & \multicolumn{3}{c}{Non-Gaussian MA}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
Seed & $F$ & $R$ & $R/F$ & $F$ & $R$ & $R/F$\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("direction_seed_results.tex", content)


def same_block_table() -> None:
    df = pd.read_csv(REVISION / "real_same_block_controls.csv")
    order = {
        "forward": 0,
        "same_block_mirror": 1,
        "block_swap": 2,
        "input_only_reverse": 3,
        "target_only_reverse": 4,
    }
    label = {
        "forward": "Forward",
        "same_block_mirror": "Same-block mirror",
        "block_swap": "Block swap",
        "input_only_reverse": "Input-only reverse",
        "target_only_reverse": "Target-only reverse",
    }
    df["condition_order"] = df["condition"].map(order)
    df = df.sort_values(["dataset", "condition_order"])
    rows = [
        f"{esc(r.dataset)} & {label[r.condition]} & {fmt(r.alpha, 1)} "
        f"& {fmt(r.val_mse)} & {fmt(r.test_mse)} & {int(r.n_test_windows):,} "
        f"& {fmt(r.risk_ratio_vs_forward, 3)} \\\\"
        for r in df.itertuples()
    ]
    content = r"""
\begin{table*}[t]
\centering
\caption{All 15 deterministic same-block control results on real data. Risk ratios are relative to the Forward condition within the same dataset.}
\label{tab:same-block-all}
\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Dataset & Condition & Ridge $\alpha$ & Validation MSE & Test MSE & Test windows & Risk ratio\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("same_block_results.tex", content)


def plan_summary_table() -> None:
    df = pd.read_csv(REVISION / "planned_information_runs.csv")
    summary = (
        df.groupby(["dataset", "condition"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            residual=("residual_mse", "mean"),
            none=("none_mse", "mean"),
            event=("event_mse", "mean"),
            separation=("separation_mse", "mean"),
            windows=("n_test_pairs", "first"),
        )
    )
    summary["condition_order"] = summary["condition"].map(
        {c: i for i, c in enumerate(PLAN_ORDER)}
    )
    summary = summary.sort_values(["dataset", "condition_order"])
    rows = []
    for r in summary.itertuples():
        rows.append(
            f"{esc(r.dataset)} & {PLAN_LABEL[r.condition]} & {int(r.seeds)} "
            f"& {fmt(r.mse_mean)} $\\pm$ {fmt(r.mse_std)} "
            f"& {fmt(r.residual)} & {fmt(r.none)} & {fmt(r.event)} "
            f"& {fmt(r.separation)} & {int(r.windows):,} \\\\"
        )
    content = r"""
\begin{table*}[t]
\centering
\caption{Full condition summary for non-oracle plan experiments. MSE is reported as the mean $\pm$ standard deviation over ten world seeds; other losses are seed means. The latent oracle is an upper bound and is not a deployable input.}
\label{tab:planned-all-conditions}
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\begin{tabular}{@{}llrrrrrrr@{}}
\toprule
Dataset & Condition & Seeds & Test MSE & Residual MSE & No-event MSE & Event MSE & Separation MSE & Windows/seed\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("planned_summary_results.tex", content)


def natural_context_table() -> None:
    df = pd.read_csv(REVISION / "natural_context_summary.csv")
    condition_order = {"target_history": 0, "observed_context": 1}
    condition_label = {
        "target_history": "Target history",
        "observed_context": "Observed context",
    }
    df["condition_order"] = df["condition"].map(condition_order)
    df = df.sort_values(["dataset", "condition_order"])
    rows = [
        f"{esc(r.dataset)} & {condition_label[r.condition]} & {fmt(r.alpha, 1)} "
        f"& {fmt(r.val_mse)} & {fmt(r.test_mse)} & {int(r.n_test_windows):,} "
        f"& {100 * r.relative_reduction_vs_target_history:.1f}\\% \\\\"
        for r in df.itertuples()
    ]
    content = r"""
\begin{table*}[t]
\centering
\caption{Full results for the real multichannel-context negative control. Relative changes use Target history as the baseline; negative values indicate higher error.}
\label{tab:natural-context-all}
\small
\begin{tabular}{@{}llrrrrr@{}}
\toprule
Dataset & Condition & Ridge $\alpha$ & Validation MSE & Test MSE & Test windows & Relative reduction\\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    write("natural_context_results.tex", content)


def planned_seed_longtable() -> None:
    df = pd.read_csv(REVISION / "planned_information_runs.csv")
    wide = df.pivot(index=["dataset", "seed"], columns="condition", values="mse")
    rows = []
    for (dataset, seed), row in wide.sort_index().iterrows():
        values = " & ".join(fmt(row[c]) for c in PLAN_ORDER)
        rows.append(f"{esc(dataset)} & {int(seed)} & {values} \\\\")
    content = r"""
\begin{longtable}{@{}lrrrrrrrr@{}}
\caption{All 210 dataset-seed-condition test MSE values for the non-oracle plan-information experiment. Each row shares the same real history windows and world seed.}
\label{tab:planned-seed-complete}\\
\toprule
Dataset & Seed & History & Clean & Noisy & Delayed & Shuffled & Misleading & Oracle\\
\midrule
\endfirsthead
\multicolumn{9}{c}{Table~\thetable\ (continued)}\\
\toprule
Dataset & Seed & History & Clean & Noisy & Delayed & Shuffled & Misleading & Oracle\\
\midrule
\endhead
\midrule
\multicolumn{9}{r}{Continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
"""
    write("planned_seed_results.tex", content)


def probability_seed_longtable() -> None:
    df = pd.read_csv(REVISION / "probabilistic_branch_scores.csv").sort_values(
        ["dataset", "seed"]
    )
    rows = [
        f"{esc(r.dataset)} & {int(r.seed)} "
        f"& {fmt(r.history_mixture_energy_score)} "
        f"& {fmt(r.plan_conditional_energy_score)} "
        f"& {100 * r.relative_energy_score_reduction:.1f}\\% "
        f"& {int(r.evaluated_branch_outcomes):,} \\\\"
        for r in df.itertuples()
    ]
    content = r"""
\begin{longtable}{@{}lrrrrr@{}}
\caption{All 30 seed-level results for intervention-residual Energy Score.}
\label{tab:probability-seed-complete}\\
\toprule
Dataset & Seed & History mixture & Plan conditional & Relative reduction & Branch outcomes\\
\midrule
\endfirsthead
\multicolumn{6}{c}{Table~\thetable\ (continued)}\\
\toprule
Dataset & Seed & History mixture & Plan conditional & Relative reduction & Branch outcomes\\
\midrule
\endhead
\midrule
\multicolumn{6}{r}{Continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
"""
    write("probability_seed_results.tex", content)


def oracle_run_longtable() -> None:
    df = pd.read_csv(ORACLE / "all_runs.csv")
    dataset_rank = {"ETTh2": 0, "ETTm1": 1, "weather": 2}
    condition_rank = {c: i for i, c in enumerate(ORACLE_CONDITION_LABEL)}
    df["dataset_rank"] = df["dataset"].map(dataset_rank)
    df["condition_rank"] = df["condition"].map(condition_rank)
    df = df.sort_values(
        ["dataset_rank", "architecture", "capacity", "condition_rank", "seed"]
    )
    arch_label = {"mlp": "MLP", "gru": "GRU"}
    rows = []
    for r in df.itertuples():
        rows.append(
            f"{esc(r.dataset)} & {arch_label[r.architecture]} & {int(r.capacity)} "
            f"& {ORACLE_CONDITION_LABEL[r.condition]} & {int(r.seed)} "
            f"& {fmt(r.best_val_mse)} & {fmt(r.mse)} & {fmt(r.mae)} "
            f"& {int(r.parameter_count):,} & {int(r.epochs_completed)} "
            f"& {fmt(r.duration_seconds, 1)} \\\\"
        )
    content = r"""
\begin{landscape}
\begin{longtable}{@{}lllrlrrrrrr@{}}
\caption{All 117 training runs from the original oracle experiment. Every run status is completed; time is the single-run wall-clock duration recorded in the logs.}
\label{tab:oracle-117-complete}\\
\toprule
Dataset & Architecture & Width & Condition & Seed & Best validation MSE & Test MSE & Test MAE & Parameters & Epochs & Seconds\\
\midrule
\endfirsthead
\multicolumn{11}{c}{Table~\thetable\ (continued)}\\
\toprule
Dataset & Architecture & Width & Condition & Seed & Best validation MSE & Test MSE & Test MAE & Parameters & Epochs & Seconds\\
\midrule
\endhead
\midrule
\multicolumn{11}{r}{Continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
\end{landscape}
"""
    write("oracle_run_results.tex", content)


def reversal_cell_longtable() -> None:
    df = pd.read_csv(REVERSAL)
    keys = ["model", "dataset", "lr_tag", "seq_len", "pred_len"]
    wide = df.pivot(index=keys, columns="case", values=["mse", "mae"])
    wide = wide.reset_index()
    dataset_rank = {v: i for i, v in enumerate(DATASET_ORDER)}
    model_rank = {v: i for i, v in enumerate(MODEL_ORDER)}
    wide["dataset_rank"] = wide["dataset"].map(dataset_rank)
    wide["model_rank"] = wide["model"].map(model_rank)
    wide = wide.sort_values(
        ["dataset_rank", "model_rank", "seq_len", "pred_len"]
    )
    rows = []
    for _, r in wide.iterrows():
        lr = str(r[("lr_tag", "")]).replace("p", ".")
        values = []
        for case in ["NN", "RN", "NR", "RR"]:
            values.extend([fmt(r[("mse", case)]), fmt(r[("mae", case)])])
        rows.append(
            f"{esc(r[('dataset', '')])} & {esc(r[('model', '')])} "
            f"& {int(r[('seq_len', '')])} & {int(r[('pred_len', '')])} "
            f"& {lr} & " + " & ".join(values) + r" \\"
        )
    content = r"""
\setlength{\LTleft}{0pt}
\setlength{\LTright}{0pt}
\begin{longtable}{@{}llrrr*{8}{r}@{}}
\caption{Complete fixed-learning-rate main-grid results for the time-reversal experiment.}
\label{tab:reversal-cell-complete}\\
\toprule
& & & & & \multicolumn{2}{c}{NN} & \multicolumn{2}{c}{RN} & \multicolumn{2}{c}{NR} & \multicolumn{2}{c}{RR}\\
\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}
Dataset & Model & $L$ & $F$ & LR & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE\\
\midrule
\endfirsthead
\multicolumn{13}{c}{Table~\thetable\ (continued)}\\
\toprule
& & & & & \multicolumn{2}{c}{NN} & \multicolumn{2}{c}{RN} & \multicolumn{2}{c}{NR} & \multicolumn{2}{c}{RR}\\
\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}
Dataset & Model & $L$ & $F$ & LR & MSE & MAE & MSE & MAE & MSE & MAE & MSE & MAE\\
\midrule
\endhead
\midrule
\multicolumn{13}{r}{Continued on next page}\\
\endfoot
\bottomrule
\endlastfoot
""" + "\n".join(rows) + r"""
\end{longtable}
"""
    write("reversal_cell_results.tex", content)


def main() -> None:
    direction_table()
    same_block_table()
    plan_summary_table()
    natural_context_table()
    planned_seed_longtable()
    probability_seed_longtable()
    oracle_run_longtable()
    reversal_cell_longtable()


if __name__ == "__main__":
    main()
