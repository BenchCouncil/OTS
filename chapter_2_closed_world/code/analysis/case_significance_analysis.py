#!/usr/bin/env python3
"""Significance tests for NN/RN/NR/RR case differences.

The analysis treats each model-dataset-length configuration as a paired block.
Omnibus tests use the Friedman test. Pairwise post-hoc tests use Wilcoxon
signed-rank tests with Holm correction within each family.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "significance"
CASES = ["NN", "NR", "RN", "RR"]
FINAL_SEQ_LENS = {96, 336}
FINAL_PRED_LENS = {96, 336}


def pstars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def parse_tsl_result_dir(name: str) -> dict[str, object] | None:
    pattern = re.compile(
        r"^long_term_forecast_"
        r"(?P<model>.+?)_"
        r"(?P<dataset>.+?)_"
        r"(?P<case>NN|RN|NR|RR)_"
        r"lr(?P<lr>[^_]+)_s(?P<seq_len>\d+)_p(?P<pred_len>\d+)_"
    )
    match = pattern.match(name)
    if not match:
        return None
    row = match.groupdict()
    row["seq_len"] = int(row["seq_len"])
    row["pred_len"] = int(row["pred_len"])
    return row


def read_tsl_deep_results() -> pd.DataFrame:
    roots = [
        PROJECT_ROOT / "analysis_input" / "extracted_bundle_20260718" / "server_extracted" / "tsl" / "results",
        PROJECT_ROOT / "analysis_input" / "extracted_bundle_20260718" / "Time-Series-Library" / "results",
        PROJECT_ROOT / "Time-Series-Library" / "results",
    ]
    rows = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for metrics_path in sorted(root.glob("*/metrics.npy")):
            parsed = parse_tsl_result_dir(metrics_path.parent.name)
            if parsed is None:
                continue
            arr = np.load(metrics_path)
            if len(arr) < 2:
                continue
            key = (parsed["model"], parsed["dataset"], parsed["case"], parsed["lr"],
                   parsed["seq_len"], parsed["pred_len"], str(metrics_path.parent.name))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "source": "deep_ltsf",
                "model": parsed["model"],
                "dataset": parsed["dataset"],
                "case": parsed["case"],
                "lr": parsed["lr"],
                "seq_len": parsed["seq_len"],
                "pred_len": parsed["pred_len"],
                "mae": float(arr[0]),
                "mse": float(arr[1]),
                "result_dir": str(metrics_path.parent.relative_to(PROJECT_ROOT)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[df["seq_len"].isin(FINAL_SEQ_LENS) & df["pred_len"].isin(FINAL_PRED_LENS)].copy()
    return df


def read_shallow_ltsf_results() -> pd.DataFrame:
    files = [
        ("OLS", RESULTS_DIR / "reversal" / "ols_results.csv"),
        ("OLS", RESULTS_DIR / "long_input" / "ols_results.csv"),
        ("RevIN-OLS", RESULTS_DIR / "revin_ols" / "revin_ols_results.csv"),
        ("RevIN-OLS", RESULTS_DIR / "long_input" / "revin_ols_results.csv"),
        ("Ridge", RESULTS_DIR / "extra_models" / "ridge_results.csv"),
        ("KNN", RESULTS_DIR / "extra_models_fast" / "knn_results.csv"),
    ]
    rows = []
    for model_name, path in files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        needed = {"case", "dataset", "seq_len", "pred_len", "mse", "mae"}
        if not needed.issubset(df.columns):
            continue
        df = df[df["seq_len"].isin(FINAL_SEQ_LENS) & df["pred_len"].isin(FINAL_PRED_LENS)].copy()
        for _, row in df.iterrows():
            rows.append({
                "source": "shallow_ltsf",
                "model": model_name,
                "dataset": row["dataset"],
                "case": row["case"],
                "lr": "",
                "seq_len": int(row["seq_len"]),
                "pred_len": int(row["pred_len"]),
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "result_dir": str(path.relative_to(PROJECT_ROOT)),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(
        subset=["source", "model", "dataset", "case", "seq_len", "pred_len"],
        keep="last",
    )
    return df


def read_m4_deep_results() -> pd.DataFrame:
    path = RESULTS_DIR / "tables" / "m4_official_deep_by_seasonal.csv"
    if not path.exists():
        path = RESULTS_DIR / "tables" / "m4_official_deep_smape_mase_owa_by_group.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["group"].isin(["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"])].copy()
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "source": "deep_m4",
            "model": row["model"],
            "dataset": f"M4_{row['group']}",
            "case": row["case"],
            "lr": row.get("lr_tag", ""),
            "seq_len": np.nan,
            "pred_len": np.nan,
            "smape": float(row["smape"]),
            "mase": float(row["mase"]),
            "owa": float(row["owa"]),
            "result_dir": row.get("forecast_dir", ""),
        })
    return pd.DataFrame(rows)


def complete_blocks(df: pd.DataFrame, block_cols: list[str], metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.dropna(subset=[metric]).copy()
    work = work[work["case"].isin(CASES)]
    counts = work.groupby(block_cols)["case"].nunique()
    complete_index = counts[counts == len(CASES)].index
    if len(complete_index) == 0:
        return work.iloc[0:0].copy()
    complete = work.set_index(block_cols).loc[complete_index].reset_index()
    complete = complete.drop_duplicates(subset=block_cols + ["case"], keep="last")
    return complete


def pivot_metric(df: pd.DataFrame, block_cols: list[str], metric: str) -> pd.DataFrame:
    complete = complete_blocks(df, block_cols, metric)
    if complete.empty:
        return pd.DataFrame()
    wide = complete.pivot_table(index=block_cols, columns="case", values=metric, aggfunc="last")
    wide = wide.dropna(subset=CASES)
    return wide[CASES]


def friedman_summary(wide: pd.DataFrame) -> dict[str, float | int]:
    n = len(wide)
    k = len(CASES)
    if n < 2:
        return {"n_blocks": n, "statistic": np.nan, "p_value": np.nan, "kendall_w": np.nan}
    arrays = [wide[c].to_numpy(dtype=float) for c in CASES]
    stat, p_value = friedmanchisquare(*arrays)
    kendall_w = float(stat / (n * (k - 1))) if n > 0 else np.nan
    return {"n_blocks": n, "statistic": float(stat), "p_value": float(p_value), "kendall_w": kendall_w}


def rank_summary(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if wide.empty:
        return pd.DataFrame(rows)
    ranks = np.vstack([rankdata(row, method="average") for row in wide[CASES].to_numpy(dtype=float)])
    for idx, case in enumerate(CASES):
        rows.append({
            "case": case,
            "n_blocks": len(wide),
            "median": float(wide[case].median()),
            "mean": float(wide[case].mean()),
            "mean_rank": float(ranks[:, idx].mean()),
            "wins_best_count": int((ranks[:, idx] == ranks.min(axis=1)).sum()),
            "wins_best_rate": float((ranks[:, idx] == ranks.min(axis=1)).mean()),
        })
    return pd.DataFrame(rows)


def pairwise_tests(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(wide) < 2:
        return pd.DataFrame(rows)
    for a, b in combinations(CASES, 2):
        diff = wide[a] - wide[b]
        nonzero = diff[diff != 0]
        try:
            if len(nonzero) == 0:
                stat, p_value = np.nan, 1.0
            else:
                stat, p_value = wilcoxon(wide[a], wide[b], zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p_value = np.nan, np.nan
        rows.append({
            "case_a": a,
            "case_b": b,
            "n_blocks": len(wide),
            "wilcoxon_statistic": float(stat) if not pd.isna(stat) else np.nan,
            "p_value": float(p_value) if not pd.isna(p_value) else np.nan,
            "median_delta_a_minus_b": float(diff.median()),
            "mean_delta_a_minus_b": float(diff.mean()),
            "a_better_rate": float((diff < 0).mean()),
            "b_better_rate": float((diff > 0).mean()),
        })
    result = pd.DataFrame(rows)
    valid = result["p_value"].notna()
    if valid.any():
        result.loc[valid, "p_holm"] = multipletests(result.loc[valid, "p_value"], method="holm")[1]
    else:
        result["p_holm"] = np.nan
    result["significant_holm_0.05"] = result["p_holm"] < 0.05
    return result


def add_context(frame: pd.DataFrame, context: dict[str, object]) -> pd.DataFrame:
    if frame.empty:
        return frame
    for key, value in context.items():
        frame.insert(0, key, value)
    return frame


def analyze_family(
    df: pd.DataFrame,
    family: str,
    metric: str,
    block_cols: list[str],
    subgroup_cols: list[str] | None = None,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    omnibus_rows = []
    pairwise_frames = []
    rank_frames = []

    analyses: list[tuple[str, pd.DataFrame]] = [("ALL", df)]
    if subgroup_cols:
        for group_values, sub in df.groupby(subgroup_cols, dropna=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            label = "|".join(f"{col}={val}" for col, val in zip(subgroup_cols, group_values))
            analyses.append((label, sub))

    for group_label, subdf in analyses:
        wide = pivot_metric(subdf, block_cols, metric)
        summary = friedman_summary(wide)
        context = {
            "family": family,
            "subgroup": group_label,
            "metric": metric,
            "block_definition": "+".join(block_cols),
        }
        row = {**context, **summary}
        row["p_label"] = pstars(row["p_value"])
        omnibus_rows.append(row)
        pairwise = add_context(pairwise_tests(wide), context)
        ranks = add_context(rank_summary(wide), context)
        if not pairwise.empty:
            pairwise["p_label_holm"] = pairwise["p_holm"].map(pstars)
            pairwise_frames.append(pairwise)
        if not ranks.empty:
            rank_frames.append(ranks)
    return omnibus_rows, pairwise_frames, rank_frames


def summarize_dataset(df: pd.DataFrame, block_cols: list[str], metrics: list[str], family: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        wide = pivot_metric(df, block_cols, metric)
        for case in CASES:
            if case in wide:
                rows.append({
                    "family": family,
                    "metric": metric,
                    "case": case,
                    "n_blocks": len(wide),
                    "median": wide[case].median(),
                    "mean": wide[case].mean(),
                })
    return pd.DataFrame(rows)


def write_report(omnibus: pd.DataFrame, pairwise: pd.DataFrame, rank: pd.DataFrame) -> None:
    report_path = OUT_DIR / "case_significance_report.md"
    top_omnibus = omnibus[omnibus["subgroup"].eq("ALL")].copy()
    top_omnibus = top_omnibus.sort_values(["family", "metric"])
    sig_pairwise = pairwise[pairwise["significant_holm_0.05"]].copy()
    sig_pairwise = sig_pairwise.sort_values(["family", "metric", "subgroup", "p_holm"])

    lines = [
        "# NN/RN/NR/RR Case Significance Analysis",
        "",
        "Method: the four cases under the same model, dataset, length, and learning-rate configuration are treated as paired observations. Overall differences use the Friedman test, effect size uses Kendall's W, and pairwise comparisons use the Wilcoxon signed-rank test with Holm correction within each family. Lower is better for every metric.",
        "",
        "## Omnibus Results",
        "",
    ]
    for _, row in top_omnibus.iterrows():
        lines.append(
            f"- `{row['family']}` / `{row['metric']}`: n={int(row['n_blocks'])}, "
            f"Friedman p={row['p_value']:.3g} ({row['p_label']}), "
            f"Kendall W={row['kendall_w']:.3f}."
        )

    lines.extend(["", "## Significant Pairwise Differences After Holm Correction", ""])
    if sig_pairwise.empty:
        lines.append("- No pairwise comparison remains significant at 0.05 after Holm correction.")
    else:
        for _, row in sig_pairwise.head(40).iterrows():
            direction = "better" if row["median_delta_a_minus_b"] < 0 else "worse"
            lines.append(
                f"- `{row['family']}` / `{row['metric']}` / `{row['subgroup']}`: "
                f"{row['case_a']} versus {row['case_b']} has median delta "
                f"{row['median_delta_a_minus_b']:.4g}; {row['case_a']} is {direction}. "
                f"Holm p={row['p_holm']:.3g}."
            )

    lines.extend([
        "",
        "## Reading Notes",
        "",
        "- In `case_rank_summary.csv`, a smaller `mean_rank` indicates a better overall rank.",
        "- In `pairwise_wilcoxon_holm.csv`, `median_delta_a_minus_b < 0` means `case_a` has lower error than `case_b`.",
        "- M4 uses the official SMAPE/MASE/OWA tables; long-sequence deep models use MAE/MSE from TSL `metrics.npy` files.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_nn_focus_table(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pairwise.empty:
        return pd.DataFrame(rows)
    for _, row in pairwise.iterrows():
        case_a = row["case_a"]
        case_b = row["case_b"]
        if "NN" not in {case_a, case_b}:
            continue
        other = case_b if case_a == "NN" else case_a
        rows.append({
            "family": row["family"],
            "metric": row["metric"],
            "subgroup": row["subgroup"],
            "comparison": f"NN_vs_{other}",
            "n_blocks": row["n_blocks"],
            "median_delta_NN_minus_other": (
                row["median_delta_a_minus_b"] if case_a == "NN" else -row["median_delta_a_minus_b"]
            ),
            "mean_delta_NN_minus_other": (
                row["mean_delta_a_minus_b"] if case_a == "NN" else -row["mean_delta_a_minus_b"]
            ),
            "NN_better_rate": row["a_better_rate"] if case_a == "NN" else row["b_better_rate"],
            "other_better_rate": row["b_better_rate"] if case_a == "NN" else row["a_better_rate"],
            "p_value": row["p_value"],
            "p_holm": row["p_holm"],
            "significant_holm_0.05": row["significant_holm_0.05"],
            "p_label_holm": row["p_label_holm"],
        })
    return pd.DataFrame(rows).sort_values(["family", "metric", "subgroup", "comparison"])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    deep_ltsf = read_tsl_deep_results()
    shallow_ltsf = read_shallow_ltsf_results()
    deep_m4 = read_m4_deep_results()

    all_obnibus_rows: list[dict[str, object]] = []
    all_pairwise: list[pd.DataFrame] = []
    all_ranks: list[pd.DataFrame] = []
    descriptive_frames = []

    specs = [
        ("deep_ltsf", deep_ltsf, ["mse", "mae"], ["model", "dataset", "seq_len", "pred_len", "lr"], ["model"]),
        ("shallow_ltsf", shallow_ltsf, ["mse", "mae"], ["model", "dataset", "seq_len", "pred_len"], ["model"]),
        ("deep_m4_official", deep_m4, ["owa", "smape", "mase"], ["model", "dataset"], ["model"]),
    ]

    source_rows = []
    for family, df, metrics, block_cols, subgroup_cols in specs:
        if df.empty:
            continue
        df.to_csv(OUT_DIR / f"{family}_paired_input_long.csv", index=False)
        source_rows.append({
            "family": family,
            "raw_rows": len(df),
            "models": ",".join(map(str, sorted(df["model"].dropna().unique()))),
            "cases": ",".join(map(str, sorted(df["case"].dropna().unique()))),
        })
        descriptive_frames.append(summarize_dataset(df, block_cols, metrics, family))
        for metric in metrics:
            omnibus, pairwise, ranks = analyze_family(df, family, metric, block_cols, subgroup_cols)
            all_obnibus_rows.extend(omnibus)
            all_pairwise.extend(pairwise)
            all_ranks.extend(ranks)

    omnibus_df = pd.DataFrame(all_obnibus_rows)
    pairwise_df = pd.concat(all_pairwise, ignore_index=True) if all_pairwise else pd.DataFrame()
    rank_df = pd.concat(all_ranks, ignore_index=True) if all_ranks else pd.DataFrame()
    descriptive_df = pd.concat(descriptive_frames, ignore_index=True) if descriptive_frames else pd.DataFrame()
    source_df = pd.DataFrame(source_rows)
    nn_focus_df = make_nn_focus_table(pairwise_df)

    if not omnibus_df.empty:
        valid = omnibus_df["p_value"].notna()
        omnibus_df["p_fdr_bh_all_omnibus"] = np.nan
        if valid.any():
            omnibus_df.loc[valid, "p_fdr_bh_all_omnibus"] = multipletests(
                omnibus_df.loc[valid, "p_value"], method="fdr_bh"
            )[1]
        omnibus_df["significant_fdr_0.05"] = omnibus_df["p_fdr_bh_all_omnibus"] < 0.05

    omnibus_df.to_csv(OUT_DIR / "case_omnibus_friedman.csv", index=False)
    pairwise_df.to_csv(OUT_DIR / "case_pairwise_wilcoxon_holm.csv", index=False)
    nn_focus_df.to_csv(OUT_DIR / "nn_vs_other_pairwise_focus.csv", index=False)
    rank_df.to_csv(OUT_DIR / "case_rank_summary.csv", index=False)
    descriptive_df.to_csv(OUT_DIR / "case_descriptive_medians.csv", index=False)
    source_df.to_csv(OUT_DIR / "case_significance_sources.csv", index=False)

    xlsx_path = OUT_DIR / "case_significance_analysis.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        omnibus_df.to_excel(writer, sheet_name="omnibus_friedman", index=False)
        pairwise_df.to_excel(writer, sheet_name="pairwise_wilcoxon", index=False)
        nn_focus_df.to_excel(writer, sheet_name="NN_focus", index=False)
        rank_df.to_excel(writer, sheet_name="rank_summary", index=False)
        descriptive_df.to_excel(writer, sheet_name="descriptive", index=False)
        source_df.to_excel(writer, sheet_name="sources", index=False)

    write_report(omnibus_df, pairwise_df, rank_df)

    print(OUT_DIR)
    print("omnibus rows", len(omnibus_df))
    print("pairwise rows", len(pairwise_df))
    print("rank rows", len(rank_df))
    print("sources")
    print(source_df.to_string(index=False))


if __name__ == "__main__":
    main()
