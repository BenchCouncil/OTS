from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "analysis_input" / "extracted_bundle_20260718"
RESULTS = BUNDLE / "results"
TSL_SERVER = BUNDLE / "server_extracted" / "tsl" / "results"
M4_SERVER = BUNDLE / "server_extracted" / "m4" / "m4_results"
M4_DATA = ROOT / "dataset" / "m4"
OUT = ROOT / "analysis_input" / "reversal_analysis_tables"
OUT.mkdir(parents=True, exist_ok=True)

FIXED_LR = {
    "DLinear": "0p005",
    "GRU": "0p001",
    "PatchTST": "0p0001",
    "TimeFilter": "0p0001",
    "TimeMixer": "0p01",
    "TimesNet": "0p0001",
    "iTransformer": "0p0001",
}

CASE_ORDER = ["NN", "RN", "NR", "RR"]
PATTERN_ORDER = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]


def gmean(values: pd.Series | np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.exp(np.mean(np.log(values)))) if len(values) else float("nan")


def parse_tsl_metrics(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    pattern = re.compile(
        r"long_term_forecast_([^_]+)_(.+?)_(NN|RN|NR|RR)_lr([^_]+)_s(\d+)_p(\d+)_"
    )
    for path in root.glob("*/metrics.npy"):
        match = pattern.match(path.parent.name)
        if not match:
            continue
        model, dataset, case, lr, seq_len, pred_len = match.groups()
        values = np.load(path).reshape(-1)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "case": case,
                "lr_tag": lr,
                "seq_len": int(seq_len),
                "pred_len": int(pred_len),
                "mae": float(values[0]),
                "mse": float(values[1]),
                "rmse": float(values[2]),
                "mape": float(values[3]),
                "mspe": float(values[4]),
                "result_dir": path.parent.name,
            }
        )
    return pd.DataFrame(rows)


def paired_effects(df: pd.DataFrame, keys: list[str], metric: str) -> pd.DataFrame:
    pivot = df.pivot_table(index=keys, columns="case", values=metric, aggfunc="first")
    pivot = pivot.dropna(subset=CASE_ORDER).reset_index()
    for case in ["RN", "NR", "RR"]:
        pivot[f"{case}_vs_NN"] = pivot[case] / pivot["NN"]
    pivot["RN_vs_NR"] = pivot["RN"] / pivot["NR"]
    pivot["train_main"] = np.sqrt((pivot["RN"] / pivot["NN"]) * (pivot["RR"] / pivot["NR"]))
    pivot["test_main"] = np.sqrt((pivot["NR"] / pivot["NN"]) * (pivot["RR"] / pivot["RN"]))
    pivot["interaction"] = (pivot["RR"] * pivot["NN"]) / (pivot["RN"] * pivot["NR"])
    pivot["abs_log_interaction"] = np.abs(np.log(pivot["interaction"]))
    return pivot


def summarize_effects(effects: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    ratio_cols = ["RN_vs_NN", "NR_vs_NN", "RR_vs_NN", "RN_vs_NR", "train_main", "test_main", "interaction"]
    rows: list[dict] = []
    grouped = effects.groupby(groups, dropna=False) if groups else [((), effects)]
    for group_key, frame in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(groups, group_key, strict=True))
        row["n"] = len(frame)
        for col in ratio_cols:
            row[f"{col}_gmean"] = gmean(frame[col])
            row[f"{col}_median"] = float(frame[col].median())
            row[f"{col}_better_share"] = float((frame[col] < 1).mean())
        row["median_abs_log_interaction"] = float(frame["abs_log_interaction"].median())
        row["p90_abs_log_interaction"] = float(frame["abs_log_interaction"].quantile(0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def add_metric_effects(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for metric in metrics:
        one = paired_effects(df, keys, metric)
        base = one[keys]
        value_cols = [c for c in one.columns if c not in keys + CASE_ORDER]
        one = pd.concat([base, one[value_cols].add_prefix(f"{metric}_")], axis=1)
        merged = one if merged is None else merged.merge(one, on=keys, how="inner")
    assert merged is not None
    return merged


def summary_for_prefixed_effects(effects: pd.DataFrame, groups: list[str], prefix: str) -> pd.DataFrame:
    renames = {
        f"{prefix}_{name}": name
        for name in [
            "RN_vs_NN",
            "NR_vs_NN",
            "RR_vs_NN",
            "RN_vs_NR",
            "train_main",
            "test_main",
            "interaction",
            "abs_log_interaction",
        ]
    }
    return summarize_effects(effects.rename(columns=renames), groups)


def load_shallow_ltsf() -> pd.DataFrame:
    nn = pd.read_csv(RESULTS / "full" / "ols_results.csv").assign(case="NN")
    other = pd.read_csv(RESULTS / "reversal" / "ols_results.csv")
    ols = pd.concat([nn, other], ignore_index=True).assign(model="OLS")
    revin = pd.read_csv(RESULTS / "revin_ols" / "revin_ols_results.csv").assign(model="RevIN-OLS")
    ridge = pd.read_csv(RESULTS / "extra_models" / "ridge_results.csv").assign(model="Ridge")
    knn = pd.read_csv(RESULTS / "extra_models_fast" / "knn_results.csv").assign(model="KNN-fast")
    long_ols = pd.read_csv(RESULTS / "long_input" / "ols_results.csv").assign(model="OLS")
    long_revin = pd.read_csv(RESULTS / "long_input" / "revin_ols_results.csv").assign(model="RevIN-OLS")
    keep = ["model", "case", "dataset", "seq_len", "pred_len", "features", "mse", "mae"]
    return pd.concat([x[keep] for x in [ols, revin, ridge, knn, long_ols, long_revin]], ignore_index=True)


def clean_series_array(values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.float64)[~np.isnan(np.asarray(v, dtype=np.float64))] for v in values]


def smape(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.abs(pred) + np.abs(target)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(200.0 * np.abs(pred - target) / denom))


def seasonal_naive(histories: list[np.ndarray], horizon: int, frequency: int) -> np.ndarray:
    rows = []
    for history in histories:
        lag = max(1, min(int(frequency), len(history)))
        season = history[-lag:]
        rows.append(np.tile(season, int(math.ceil(horizon / lag)))[:horizon])
    return np.stack(rows)


def evaluate_m4_deep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    info = pd.read_csv(M4_DATA / "M4-info.csv")
    train_all = np.load(M4_DATA / "training.npz", allow_pickle=True)
    test_all = np.load(M4_DATA / "test.npz", allow_pickle=True)
    naive2_all = pd.read_csv(M4_DATA / "submission-Naive2.csv").iloc[:, 1:].to_numpy(dtype=float)

    corrected_rows: list[dict] = []
    official_rows: list[dict] = []
    hash_rows: list[dict] = []

    for folder in sorted(M4_SERVER.iterdir()):
        if not folder.is_dir():
            continue
        match = re.match(r"(.+?)_(NN|RN|NR|RR)_lr(.+)", folder.name)
        if not match:
            continue
        model, case, lr_tag = match.groups()
        for pattern in PATTERN_ORDER:
            path = folder / f"{pattern}_forecast.csv"
            if not path.exists():
                continue
            mask = info["SP"].to_numpy() == pattern
            horizon = int(info.loc[mask, "Horizon"].iloc[0])
            frequency = int(info.loc[mask, "Frequency"].iloc[0])
            pred = pd.read_csv(path).to_numpy(dtype=float)
            train = clean_series_array(train_all[mask])
            test = np.stack([v[:horizon] for v in clean_series_array(test_all[mask])])
            naive2 = naive2_all[mask, :horizon]
            scales = np.array(
                [
                    np.mean(np.abs(x[frequency:] - x[:-frequency])) if len(x) > frequency else 1.0
                    for x in train
                ],
                dtype=float,
            )
            scales = np.where(scales > 1e-12, scales, 1e-12)

            # The implemented M4 deep inference context is controlled by the train-side flag.
            context_series = [x[::-1].copy() if case[0] == "R" else x for x in train]
            seq_len = 2 * horizon
            contexts = [x[-seq_len:] for x in context_series]
            context_naive = seasonal_naive(contexts, horizon, frequency)

            for mode, target, sink in [
                ("corrected_target_order", test[:, ::-1] if case[1] == "R" else test, corrected_rows),
                ("official_normal_target", test, official_rows),
            ]:
                err = pred - target
                model_smape = smape(pred, target)
                model_mase = float(np.mean(np.mean(np.abs(err), axis=1) / scales))
                naive2_smape = smape(naive2, target)
                naive2_mase = float(np.mean(np.mean(np.abs(naive2 - target), axis=1) / scales))
                context_smape = smape(context_naive, target)
                context_mase = float(np.mean(np.mean(np.abs(context_naive - target), axis=1) / scales))
                sink.append(
                    {
                        "model": model,
                        "case": case,
                        "seasonal_pattern": pattern,
                        "frequency": frequency,
                        "horizon": horizon,
                        "series_count": int(mask.sum()),
                        "lr_tag": lr_tag,
                        "target_mode": mode,
                        "mse": float(np.mean(err**2)),
                        "mae": float(np.mean(np.abs(err))),
                        "smape": model_smape,
                        "mase": model_mase,
                        "owa_naive2": 0.5 * (model_smape / naive2_smape + model_mase / naive2_mase),
                        "owa_context": 0.5 * (model_smape / context_smape + model_mase / context_mase),
                        "naive2_smape": naive2_smape,
                        "naive2_mase": naive2_mase,
                        "context_naive_smape": context_smape,
                        "context_naive_mase": context_mase,
                    }
                )
            hash_rows.append(
                {
                    "model": model,
                    "case": case,
                    "seasonal_pattern": pattern,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    return pd.DataFrame(corrected_rows), pd.DataFrame(official_rows), pd.DataFrame(hash_rows)


def load_shallow_m4() -> pd.DataFrame:
    base = pd.read_csv(RESULTS / "m4_reversal" / "m4_timesnet_metrics.csv")
    base["model"] = base["model_name"].map({"ols": "OLS-default", "revin_ols": "RevIN-OLS-default"})
    ridge = pd.read_csv(RESULTS / "extra_models" / "m4_ridge_results.csv").assign(model="Ridge-default")
    knn = pd.read_csv(RESULTS / "extra_models_fast" / "m4_knn_results.csv").assign(model="KNN-default")
    long_ols = pd.read_csv(RESULTS / "long_input" / "m4_ols_results.csv").assign(model="OLS-long")
    long_revin = pd.read_csv(RESULTS / "long_input" / "m4_revin_ols_results.csv").assign(model="RevIN-OLS-long")
    keep = ["model", "case", "seasonal_pattern", "seq_len", "pred_len", "smape", "mase", "owa"]
    return pd.concat([x[keep] for x in [base, ridge, knn, long_ols, long_revin]], ignore_index=True)


def m4_weighted_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, case), frame in df.groupby(["model", "case"]):
        weights = frame["series_count"].to_numpy(dtype=float)
        weighted = lambda column: float(np.average(frame[column], weights=weights))
        model_smape = weighted("smape")
        model_mase = weighted("mase")
        naive2_smape = weighted("naive2_smape")
        naive2_mase = weighted("naive2_mase")
        context_smape = weighted("context_naive_smape")
        context_mase = weighted("context_naive_mase")
        rows.append(
            {
                "model": model,
                "case": case,
                "smape": model_smape,
                "mase": model_mase,
                "owa_naive2": 0.5 * (model_smape / naive2_smape + model_mase / naive2_mase),
                "owa_context": 0.5 * (model_smape / context_smape + model_mase / context_mase),
            }
        )
    return pd.DataFrame(rows)


def property_correlations(deep_effects: pd.DataFrame) -> pd.DataFrame:
    props = pd.read_csv(RESULTS / "property_analysis" / "dataset_properties.csv")
    condition = pd.read_csv(RESULTS / "open_time_series_alignment" / "condition_space_with_last_value.csv")
    lv = (
        condition[(condition["model"] == "ols") & (condition["case"] == "NN")]
        .assign(last_value_competitiveness=lambda x: x["mse"] / x["last_value_mse"])
        .groupby("dataset", as_index=False)["last_value_competitiveness"]
        .median()
    )
    agg = deep_effects.groupby("dataset", as_index=False).agg(
        RN_vs_NN=("RN_vs_NN", gmean),
        NR_vs_NN=("NR_vs_NN", gmean),
        RR_vs_NN=("RR_vs_NN", gmean),
        train_main=("train_main", gmean),
        test_main=("test_main", gmean),
        RN_vs_NR=("RN_vs_NR", gmean),
        abs_log_interaction=("abs_log_interaction", "median"),
    )
    joined = agg.merge(props, on="dataset", how="left").merge(lv, on="dataset", how="left")
    joined.to_csv(OUT / "deep_ltsf_dataset_effects_with_properties.csv", index=False)
    features = [
        "trend_r2",
        "trend_slope_abs",
        "lag1_autocorr",
        "period_autocorr",
        "roughness_std_diff",
        "train_test_mean_shift",
        "train_test_scale_shift",
        "forward_similarity",
        "reverse_similarity",
        "reverse_similarity_gain",
        "last_value_competitiveness",
    ]
    targets = ["RN_vs_NN", "NR_vs_NN", "RR_vs_NN", "train_main", "test_main", "RN_vs_NR", "abs_log_interaction"]
    rows = []
    for feature in features:
        for target in targets:
            pair = joined[[feature, target]].dropna()
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n": len(pair),
                    "pearson": pair[feature].corr(pair[target], method="pearson"),
                    "spearman": pair[feature].corr(pair[target], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def local_tsl_audit(server: pd.DataFrame) -> dict:
    local_root = BUNDLE / "Time-Series-Library" / "results"
    local = parse_tsl_metrics(local_root)
    if local.empty:
        return {"local_metric_files": 0}
    merged = local.merge(server, on="result_dir", suffixes=("_local", "_server"), how="left")
    return {
        "local_metric_files": int(len(local)),
        "matched_server_dirs": int(merged["mse_server"].notna().sum()),
        "exact_metric_matches": int(
            np.isclose(merged["mse_local"], merged["mse_server"], rtol=0, atol=0, equal_nan=False).sum()
        ),
        "local_rows": local[["model", "dataset", "case", "seq_len", "pred_len", "mse", "mae"]].to_dict("records"),
    }


def main() -> None:
    server_all = parse_tsl_metrics(TSL_SERVER)
    server_all.to_csv(OUT / "deep_ltsf_all_parsed_metrics.csv", index=False)
    formal = server_all[
        server_all["seq_len"].isin([96, 336])
        & server_all["pred_len"].isin([96, 336])
        & server_all.apply(lambda row: row["lr_tag"] == FIXED_LR[row["model"]], axis=1)
    ].copy()
    formal.to_csv(OUT / "deep_ltsf_fixed_lr_metrics.csv", index=False)

    best_case = (
        formal.pivot_table(
            index=["model", "dataset", "seq_len", "pred_len"], columns="case", values="mse"
        )
        .dropna(subset=CASE_ORDER)
        .reset_index()
    )
    best_case["best_case"] = best_case[CASE_ORDER].idxmin(axis=1)
    best_case["worst_case"] = best_case[CASE_ORDER].idxmax(axis=1)
    best_case.to_csv(OUT / "deep_ltsf_best_worst_case_by_cell.csv", index=False)
    (
        best_case.groupby(["model", "best_case"]).size().rename("count").reset_index()
    ).to_csv(OUT / "deep_ltsf_best_case_counts_by_model.csv", index=False)

    deep_keys = ["model", "dataset", "seq_len", "pred_len"]
    deep_effects = add_metric_effects(formal, deep_keys, ["mse", "mae"])
    deep_effects.to_csv(OUT / "deep_ltsf_paired_effects.csv", index=False)
    deep_mse = deep_effects.rename(
        columns={
            f"mse_{name}": name
            for name in [
                "RN_vs_NN",
                "NR_vs_NN",
                "RR_vs_NN",
                "RN_vs_NR",
                "train_main",
                "test_main",
                "interaction",
                "abs_log_interaction",
            ]
        }
    )
    deep_model_summary = summarize_effects(deep_mse, ["model"])
    deep_dataset_summary = summarize_effects(deep_mse, ["dataset"])
    deep_seqpred_summary = summarize_effects(deep_mse, ["seq_len", "pred_len"])
    deep_model_summary.to_csv(OUT / "deep_ltsf_mse_summary_by_model.csv", index=False)
    deep_dataset_summary.to_csv(OUT / "deep_ltsf_mse_summary_by_dataset.csv", index=False)
    deep_seqpred_summary.to_csv(OUT / "deep_ltsf_mse_summary_by_seq_pred.csv", index=False)

    # Learning-rate sensitivity: only configurations with all four cases are admissible.
    lr_effects = paired_effects(server_all, ["model", "dataset", "seq_len", "pred_len", "lr_tag"], "mse")
    lr_effects["formal_lr"] = lr_effects.apply(lambda row: row["lr_tag"] == FIXED_LR[row["model"]], axis=1)
    lr_effects.to_csv(OUT / "deep_ltsf_learning_rate_effects.csv", index=False)
    lr_multi = (
        lr_effects.groupby(["model", "dataset", "seq_len", "pred_len"])
        .filter(lambda x: x["lr_tag"].nunique() > 1)
        .groupby(["model", "dataset", "seq_len", "pred_len"], as_index=False)
        .agg(
            n_lr=("lr_tag", "nunique"),
            RN_log_range=("RN_vs_NN", lambda x: float(np.log(x).max() - np.log(x).min())),
            NR_log_range=("NR_vs_NN", lambda x: float(np.log(x).max() - np.log(x).min())),
            RR_log_range=("RR_vs_NN", lambda x: float(np.log(x).max() - np.log(x).min())),
            RN_sign_agreement=("RN_vs_NN", lambda x: float(max((x < 1).mean(), (x >= 1).mean()))),
            NR_sign_agreement=("NR_vs_NN", lambda x: float(max((x < 1).mean(), (x >= 1).mean()))),
            RR_sign_agreement=("RR_vs_NN", lambda x: float(max((x < 1).mean(), (x >= 1).mean()))),
        )
    )
    lr_multi.to_csv(OUT / "deep_ltsf_learning_rate_sensitivity.csv", index=False)

    shallow = load_shallow_ltsf()
    shallow.to_csv(OUT / "shallow_ltsf_metrics.csv", index=False)
    shallow_keys = ["model", "dataset", "seq_len", "pred_len"]
    shallow_effects = paired_effects(shallow, shallow_keys, "mse")
    shallow_effects.to_csv(OUT / "shallow_ltsf_mse_paired_effects.csv", index=False)
    shallow_summary = summarize_effects(shallow_effects, ["model", "seq_len"])
    shallow_summary.to_csv(OUT / "shallow_ltsf_mse_summary_by_model_input.csv", index=False)

    m4_corrected, m4_official, m4_hashes = evaluate_m4_deep()
    m4_corrected.to_csv(OUT / "m4_deep_metrics_corrected_target_order.csv", index=False)
    m4_official.to_csv(OUT / "m4_deep_metrics_official_normal_target.csv", index=False)
    m4_weighted_aggregate(m4_corrected).to_csv(
        OUT / "m4_deep_weighted_aggregate_corrected_target_order.csv", index=False
    )
    m4_weighted_aggregate(m4_official).to_csv(
        OUT / "m4_deep_weighted_aggregate_official_normal_target.csv", index=False
    )
    m4_hashes.to_csv(OUT / "m4_deep_forecast_hashes.csv", index=False)
    m4_keys = ["model", "seasonal_pattern"]
    m4_corr_effects = add_metric_effects(m4_corrected, m4_keys, ["smape", "mase", "owa_context"])
    m4_off_effects = add_metric_effects(m4_official, m4_keys, ["smape", "mase", "owa_naive2"])
    m4_corr_effects.to_csv(OUT / "m4_deep_paired_effects_corrected.csv", index=False)
    m4_off_effects.to_csv(OUT / "m4_deep_paired_effects_official.csv", index=False)
    for name, effects, prefix in [
        ("m4_deep_smape_summary_by_model_corrected.csv", m4_corr_effects, "smape"),
        ("m4_deep_smape_summary_by_pattern_corrected.csv", m4_corr_effects, "smape"),
        ("m4_deep_owa_summary_by_model_official.csv", m4_off_effects, "owa_naive2"),
    ]:
        groups = ["seasonal_pattern"] if "pattern" in name else ["model"]
        summary_for_prefixed_effects(effects, groups, prefix).to_csv(OUT / name, index=False)

    shallow_m4 = load_shallow_m4()
    shallow_m4.to_csv(OUT / "m4_shallow_metrics.csv", index=False)
    shallow_m4_effects = paired_effects(shallow_m4, ["model", "seasonal_pattern", "seq_len", "pred_len"], "owa")
    shallow_m4_effects.to_csv(OUT / "m4_shallow_owa_paired_effects.csv", index=False)
    shallow_m4_summary = summarize_effects(shallow_m4_effects, ["model"])
    shallow_m4_summary.to_csv(OUT / "m4_shallow_owa_summary_by_model.csv", index=False)

    correlations = property_correlations(deep_mse)
    correlations.to_csv(OUT / "deep_ltsf_property_correlations.csv", index=False)

    pair_equal = []
    for (model, pattern), group in m4_hashes.groupby(["model", "seasonal_pattern"]):
        hashes = group.set_index("case")["sha256"].to_dict()
        pair_equal.append(
            {
                "model": model,
                "seasonal_pattern": pattern,
                "NN_equals_NR": hashes.get("NN") == hashes.get("NR"),
                "RN_equals_RR": hashes.get("RN") == hashes.get("RR"),
            }
        )
    pair_equal_df = pd.DataFrame(pair_equal)
    pair_equal_df.to_csv(OUT / "m4_deep_forecast_pair_equality.csv", index=False)

    ols = shallow[(shallow["model"] == "OLS") & (shallow["seq_len"] != 336)]
    ridge = shallow[shallow["model"] == "Ridge"]
    ols_ridge = ols.merge(ridge, on=["case", "dataset", "seq_len", "pred_len"], suffixes=("_ols", "_ridge"))
    audit = {
        "server_all_metric_files": int(len(server_all)),
        "formal_fixed_lr_rows": int(len(formal)),
        "formal_complete_four_case_cells": int(len(deep_effects)),
        "formal_coverage_by_model_case": formal.groupby(["model", "case"]).size().unstack(fill_value=0).to_dict(),
        "m4_forecast_files": int(len(m4_hashes)),
        "m4_unique_forecast_hashes": int(m4_hashes["sha256"].nunique()),
        "m4_NN_NR_exact_pairs": int(pair_equal_df["NN_equals_NR"].sum()),
        "m4_RN_RR_exact_pairs": int(pair_equal_df["RN_equals_RR"].sum()),
        "m4_pair_count_each": int(len(pair_equal_df)),
        "ols_ridge_max_abs_mse_difference": float(np.abs(ols_ridge["mse_ols"] - ols_ridge["mse_ridge"]).max()),
        "learning_rate_complete_cells": int(len(lr_effects)),
        "learning_rate_multi_lr_base_cells": int(len(lr_multi)),
        "local_tsl": local_tsl_audit(server_all),
    }
    (OUT / "coverage_and_integrity_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nDEEP LTSF BY MODEL (MSE ratios; <1 is better)")
    cols = [
        "model",
        "n",
        "RN_vs_NN_gmean",
        "NR_vs_NN_gmean",
        "RR_vs_NN_gmean",
        "train_main_gmean",
        "test_main_gmean",
        "RN_vs_NR_gmean",
        "median_abs_log_interaction",
    ]
    print(deep_model_summary[cols].round(4).to_string(index=False))
    print("\nDEEP LTSF BY DATASET")
    print(deep_dataset_summary[["dataset"] + cols[1:]].round(4).to_string(index=False))
    print("\nDEEP LTSF BY SEQ/PRED")
    print(deep_seqpred_summary[["seq_len", "pred_len"] + cols[1:]].round(4).to_string(index=False))
    print("\nSHALLOW LTSF")
    print(shallow_summary[["model", "seq_len"] + cols[1:]].round(4).to_string(index=False))
    print("\nM4 DEEP CORRECTED TARGET ORDER: sMAPE ratios")
    m4_model_summary = summary_for_prefixed_effects(m4_corr_effects, ["model"], "smape")
    print(m4_model_summary[cols].round(4).to_string(index=False))
    print("\nM4 DEEP CORRECTED TARGET ORDER BY PATTERN: sMAPE ratios")
    m4_pattern_summary = summary_for_prefixed_effects(m4_corr_effects, ["seasonal_pattern"], "smape")
    print(m4_pattern_summary[["seasonal_pattern"] + cols[1:]].round(4).to_string(index=False))
    print("\nM4 DEEP OFFICIAL NORMAL TARGET: OWA ratios")
    m4_off_summary = summary_for_prefixed_effects(m4_off_effects, ["model"], "owa_naive2")
    print(m4_off_summary[cols].round(4).to_string(index=False))
    print("\nM4 SHALLOW OWA ratios")
    print(shallow_m4_summary[cols].round(4).to_string(index=False))
    print("\nAUDIT")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
