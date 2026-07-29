#!/usr/bin/env python3
"""Recompute the evidence tables used by the Closed-or-Open draft.

The script only reads experiment outputs and writes compact CSV/JSON summaries
under this draft folder. Ratios are always paired within the same experimental
cell and aggregated geometrically, because the estimands are multiplicative.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[3]
CHAPTER2 = RELEASE_ROOT / "chapter_2_closed_world"
TABLES = CHAPTER2 / "results" / "paper_table_inputs" / "generated"
TABLES.mkdir(parents=True, exist_ok=True)

DEEP = CHAPTER2 / "results" / "reversal_analysis_tables" / "deep_ltsf_paired_effects.csv"
PARTIAL_INDEX = CHAPTER2 / "results" / "partial_channel_results_20260718" / "result_index.csv"
PARTIAL_CHANNELS = CHAPTER2 / "results" / "partial_channel_results_20260718" / "all_channel_metrics.csv.gz"
AUX_DIR = CHAPTER2 / "results" / "auxiliary_channel_study"
AUX = AUX_DIR / "results_comprehensive.csv"
AUX_DIRECTIONAL = AUX_DIR / "results_directional.csv"
AUX_ADDITIVE = AUX_DIR / "results_additive_fair.csv"

CASE_ORDER = ["NN", "RN", "NR", "RR"]


def geomean(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    x = x[x > 0]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def summarize_ratios(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = [((), df)] if not group_cols else df.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row.update(
            {
                "n_cells": int(len(group)),
                "RN_over_NN_gmean": geomean(group["RN_over_NN"]),
                "NR_over_NN_gmean": geomean(group["NR_over_NN"]),
                "RR_over_NN_gmean": geomean(group["RR_over_NN"]),
                "RR_better_than_NN_fraction": float((group["RR_over_NN"] < 1).mean()),
                "RR_within_5pct_fraction": float(
                    (np.abs(np.log(group["RR_over_NN"])) <= np.log(1.05)).mean()
                ),
                "RR_within_10pct_fraction": float(
                    (np.abs(np.log(group["RR_over_NN"])) <= np.log(1.10)).mean()
                ),
                "interaction_gmean": geomean(group["interaction"]),
                "abs_log_interaction_median": float(np.abs(np.log(group["interaction"])).median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def pivot_cases(df: pd.DataFrame, index_cols: list[str], value_col: str) -> pd.DataFrame:
    wide = df.pivot_table(index=index_cols, columns="case", values=value_col, aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.dropna(subset=CASE_ORDER).copy()
    wide["RN_over_NN"] = wide["RN"] / wide["NN"]
    wide["NR_over_NN"] = wide["NR"] / wide["NN"]
    wide["RR_over_NN"] = wide["RR"] / wide["NN"]
    wide["RN_over_NR"] = wide["RN"] / wide["NR"]
    wide["interaction"] = wide["RR"] * wide["NN"] / (wide["RN"] * wide["NR"])
    return wide


def summarize_deep() -> dict[str, int]:
    deep = pd.read_csv(DEEP)
    renamed = deep.rename(
        columns={
            "mse_RN_vs_NN": "RN_over_NN",
            "mse_NR_vs_NN": "NR_over_NN",
            "mse_RR_vs_NN": "RR_over_NN",
            "mse_RN_vs_NR": "RN_over_NR",
            "mse_interaction": "interaction",
        }
    )
    summarize_ratios(renamed, []).to_csv(TABLES / "global_reversal_overall.csv", index=False)
    summarize_ratios(renamed, ["model"]).to_csv(TABLES / "global_reversal_by_model.csv", index=False)
    summarize_ratios(renamed, ["dataset"]).to_csv(TABLES / "global_reversal_by_dataset.csv", index=False)
    return {"deep_complete_cells": int(len(deep))}


def summarize_partial() -> dict[str, int]:
    idx = pd.read_csv(PARTIAL_INDEX)
    idx_wide = pivot_cases(
        idx,
        ["model", "dataset", "seq_len", "pred_len"],
        "mean_mse_normalized",
    )
    summarize_ratios(idx_wide, []).to_csv(TABLES / "partial_all_channels_overall.csv", index=False)

    channels = pd.read_csv(PARTIAL_CHANNELS)
    channel_grouped = (
        channels.groupby(
            ["model", "dataset", "seq_len", "pred_len", "case", "selected_for_reversal"],
            as_index=False,
            observed=True,
        )["mse_normalized"]
        .mean()
    )
    partial = pivot_cases(
        channel_grouped,
        ["model", "dataset", "seq_len", "pred_len", "selected_for_reversal"],
        "mse_normalized",
    )
    partial["channel_group"] = np.where(
        partial["selected_for_reversal"], "selected", "unselected"
    )
    partial.to_csv(TABLES / "partial_channel_paired_cells.csv", index=False)
    summarize_ratios(partial, ["channel_group"]).to_csv(
        TABLES / "partial_channel_overall.csv", index=False
    )
    summarize_ratios(partial, ["model", "channel_group"]).to_csv(
        TABLES / "partial_channel_by_model.csv", index=False
    )
    summarize_ratios(partial, ["dataset", "channel_group"]).to_csv(
        TABLES / "partial_channel_by_dataset.csv", index=False
    )
    return {
        "partial_result_units": int(len(idx)),
        "partial_channel_metric_rows": int(len(channels)),
        "partial_complete_all_channel_cells": int(len(idx_wide)),
        "partial_complete_group_cells": int(len(partial)),
    }


def aux_summary_for(df: pd.DataFrame, label: str) -> pd.DataFrame:
    comparable = df[(df["mono"].isin(["append", "nomono"])) & (df["eval_space"] == "normalized")]
    wide = comparable.pivot_table(
        index=["dataset", "model", "paradigm"], columns="mono", values="mse", aggfunc="first"
    ).dropna(subset=["append", "nomono"])
    wide["append_over_nomono"] = wide["append"] / wide["nomono"]
    wide = wide.reset_index()

    rows = []
    for cols in [[], ["model"], ["dataset"], ["paradigm"]]:
        groups = [((), wide)] if not cols else wide.groupby(cols, sort=True, dropna=False)
        for key, group in groups:
            if not isinstance(key, tuple):
                key = (key,)
            row = {"deduplication": label, "grouping": "+".join(cols) or "overall"}
            row.update(dict(zip(cols, key)))
            row.update(
                {
                    "n_pairs": int(len(group)),
                    "append_over_nomono_gmean": geomean(group["append_over_nomono"]),
                    "append_over_nomono_median": float(group["append_over_nomono"].median()),
                    "append_better_fraction": float((group["append_over_nomono"] < 1).mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_auxiliary() -> dict[str, object]:
    raw = pd.read_csv(AUX, dtype=str)
    key = ["dataset", "model", "paradigm", "mono"]
    for col in ["mse", "mae", "n_feat"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    valid = raw.dropna(subset=key + ["mse", "mae", "eval_space"]).copy()
    valid["source_row"] = valid.index + 2

    duplicate_key_count = int(valid.duplicated(key, keep=False).groupby(valid[key].apply(tuple, axis=1)).any().sum())
    last = valid.sort_values("source_row").drop_duplicates(key, keep="last")
    median = (
        valid.groupby(key + ["eval_space"], as_index=False, dropna=False)
        .agg(mse=("mse", "median"), mae=("mae", "median"), n_feat=("n_feat", "median"))
    )

    expected = set(
        itertools.product(
            ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Exchange", "Weather"],
            ["iTransformer", "DLinear", "PatchTST", "TimesNet", "GRU"],
            ["original", "train_reversed", "test_reversed", "both_reversed"],
            ["append", "add", "nomono"],
        )
    )
    observed_numeric = set(last[key].itertuples(index=False, name=None))
    observed_all = set(raw.dropna(subset=key)[key].itertuples(index=False, name=None))
    absent = sorted(expected - observed_all)
    invalid = sorted((expected & observed_all) - observed_numeric)

    pd.concat(
        [aux_summary_for(last, "last_occurrence"), aux_summary_for(median, "duplicate_median")],
        ignore_index=True,
    ).to_csv(TABLES / "auxiliary_append_summary.csv", index=False)

    directional = pd.read_csv(AUX_DIRECTIONAL)
    additive = pd.read_csv(
        AUX_ADDITIVE,
        header=None,
        names=["dataset", "model", "paradigm", "pred_len", "mse", "mae"],
    )
    audit = {
        "comprehensive_csv_logical_records": int(len(raw)),
        "comprehensive_numeric_valid_rows": int(len(valid)),
        "comprehensive_unique_keys": int(len(last)),
        "comprehensive_expected_keys": int(len(expected)),
        "comprehensive_duplicate_keys": duplicate_key_count,
        "comprehensive_absent_keys": len(absent),
        "absent_key_list": [list(x) for x in absent],
        "comprehensive_invalid_keys": len(invalid),
        "invalid_key_list": [list(x) for x in invalid],
        "directional_rows": int(len(directional)),
        "directional_note": "append MSE is normalized, add/baseline MSE is original-space; delta_mse is not a cross-mode estimand.",
        "additive_fair_rows": int(len(additive)),
        "additive_fair_unique_grid_cells": int(
            len(additive.drop_duplicates(["dataset", "model", "paradigm", "pred_len"]))
        ),
        "additive_fair_note": "The file has no matched no-add baseline and therefore cannot identify an encoding effect.",
    }
    (TABLES / "auxiliary_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    audit: dict[str, object] = {}
    audit.update(summarize_deep())
    audit.update(summarize_partial())
    if all(path.exists() for path in (AUX, AUX_DIRECTIONAL, AUX_ADDITIVE)):
        audit["auxiliary"] = summarize_auxiliary()
    else:
        audit["auxiliary"] = {
            "status": "not_included",
            "note": "The earlier auxiliary-channel side experiment is not required for the main release tables.",
        }
    (TABLES / "evidence_summary.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
