#!/usr/bin/env python3
"""Evaluate M4 forecasts with the official-style SMAPE, MASE, and OWA metrics."""

from __future__ import annotations

import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
M4_ROOT = PROJECT_ROOT / "dataset" / "m4"
SERVER_ARCHIVE = PROJECT_ROOT / "server_results" / "m4_results_and_logs_20260718_102519.tar.gz"
EXTRACT_ROOT = PROJECT_ROOT / "analysis_input" / "m4_official_eval_extracted"
DEEP_M4_RESULTS = EXTRACT_ROOT / "m4_results"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"

SEASONAL_PATTERNS = ["Yearly", "Quarterly", "Monthly", "Weekly", "Daily", "Hourly"]
REPORT_GROUPS = ["Yearly", "Quarterly", "Monthly", "Others", "Average"]
HORIZONS = {
    "Yearly": 6,
    "Quarterly": 8,
    "Monthly": 18,
    "Weekly": 13,
    "Daily": 14,
    "Hourly": 48,
}


def clean_series(values: np.ndarray) -> list[np.ndarray]:
    cleaned = []
    for value in values:
        series = np.asarray(value, dtype=np.float64)
        cleaned.append(series[~np.isnan(series)])
    return cleaned


def stack_group(values: np.ndarray, groups: np.ndarray, group_name: str) -> np.ndarray:
    cleaned = clean_series(values[groups == group_name])
    return np.stack(cleaned).astype(np.float64)


def list_group(values: np.ndarray, groups: np.ndarray, group_name: str) -> list[np.ndarray]:
    return clean_series(values[groups == group_name])


def smape_2(forecast: np.ndarray, target: np.ndarray) -> float:
    denom = np.abs(target) + np.abs(forecast)
    denom[denom == 0.0] = 1.0
    return float(np.mean(200.0 * np.abs(forecast - target) / denom))


def mase_one(forecast: np.ndarray, insample: np.ndarray, target: np.ndarray, frequency: int) -> float:
    scale = np.mean(np.abs(insample[frequency:] - insample[:-frequency]))
    if scale == 0.0:
        return np.nan
    return float(np.mean(np.abs(forecast - target)) / scale)


def mase_group(forecast: np.ndarray, insample: np.ndarray, target: np.ndarray, frequency: int) -> float:
    values = [mase_one(forecast[i], insample[i], target[i], frequency) for i in range(len(forecast))]
    return float(np.nanmean(values))


def summarize_official(groups_df: pd.DataFrame, series_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    by_group = groups_df.set_index("group")

    def weighted(group_names: list[str], column: str) -> float:
        numer = sum(float(by_group.loc[g, column]) * series_counts[g] for g in group_names)
        denom = sum(series_counts[g] for g in group_names)
        return numer / denom

    for group_name in ["Yearly", "Quarterly", "Monthly"]:
        row = by_group.loc[group_name].to_dict()
        row["group"] = group_name
        row["series_count"] = series_counts[group_name]
        row["owa"] = 0.5 * (row["smape"] / row["naive_smape"] + row["mase"] / row["naive_mase"])
        rows.append(row)

    for group_name, members in [
        ("Others", ["Weekly", "Daily", "Hourly"]),
        ("Average", SEASONAL_PATTERNS),
    ]:
        row = {
            "group": group_name,
            "series_count": sum(series_counts[g] for g in members),
            "smape": weighted(members, "smape"),
            "mase": weighted(members, "mase"),
            "naive_smape": weighted(members, "naive_smape"),
            "naive_mase": weighted(members, "naive_mase"),
        }
        row["owa"] = 0.5 * (row["smape"] / row["naive_smape"] + row["mase"] / row["naive_mase"])
        rows.append(row)

    return pd.DataFrame(rows)


def ensure_deep_results() -> Path:
    if DEEP_M4_RESULTS.exists() and any(DEEP_M4_RESULTS.glob("*/*_forecast.csv")):
        return DEEP_M4_RESULTS
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SERVER_ARCHIVE, "r:gz") as archive:
        archive.extractall(EXTRACT_ROOT)
    return DEEP_M4_RESULTS


def parse_result_dir_name(name: str) -> tuple[str, str, str]:
    match = re.match(r"(.+)_([NR]{2})_(lr.+)$", name)
    if not match:
        return name, "", ""
    return match.group(1), match.group(2), match.group(3)


def read_forecast_csv(path: Path, horizon: int) -> np.ndarray:
    df = pd.read_csv(path)
    values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if values.shape[1] > horizon:
        values = values[:, -horizon:]
    if values.shape[1] != horizon:
        raise ValueError(f"{path} has {values.shape[1]} numeric columns, expected {horizon}")
    return values


def evaluate_deep_forecast_dirs() -> tuple[pd.DataFrame, pd.DataFrame]:
    info = pd.read_csv(M4_ROOT / "M4-info.csv")
    groups = info["SP"].to_numpy()
    frequencies = info["Frequency"].to_numpy()
    series_counts = info.groupby("SP").size().to_dict()
    training_values = np.load(M4_ROOT / "training.npz", allow_pickle=True)
    test_values = np.load(M4_ROOT / "test.npz", allow_pickle=True)
    naive_values = pd.read_csv(M4_ROOT / "submission-Naive2.csv").iloc[:, 1:].to_numpy(dtype=np.float64)

    target_by_group = {g: stack_group(test_values, groups, g) for g in SEASONAL_PATTERNS}
    insample_by_group = {g: list_group(training_values, groups, g) for g in SEASONAL_PATTERNS}
    naive_by_group = {
        g: np.stack(clean_series(naive_values[groups == g])).astype(np.float64)
        for g in SEASONAL_PATTERNS
    }
    frequency_by_group = {g: int(frequencies[groups == g][0]) for g in SEASONAL_PATTERNS}

    detailed_rows = []
    summary_rows = []
    result_root = ensure_deep_results()

    for result_dir in sorted(p for p in result_root.iterdir() if p.is_dir()):
        model, case, lr_tag = parse_result_dir_name(result_dir.name)
        group_rows = []
        for group_name in SEASONAL_PATTERNS:
            forecast_path = result_dir / f"{group_name}_forecast.csv"
            if not forecast_path.exists():
                raise FileNotFoundError(forecast_path)
            forecast = read_forecast_csv(forecast_path, HORIZONS[group_name])
            target = target_by_group[group_name]
            insample = insample_by_group[group_name]
            naive = naive_by_group[group_name]
            if forecast.shape != target.shape:
                raise ValueError(f"{forecast_path} shape {forecast.shape} != target {target.shape}")

            row = {
                "source": "deep_tsl",
                "model": model,
                "case": case,
                "lr_tag": lr_tag,
                "group": group_name,
                "series_count": series_counts[group_name],
                "smape": smape_2(forecast, target),
                "mase": mase_group(forecast, insample, target, frequency_by_group[group_name]),
                "naive_smape": smape_2(naive, target),
                "naive_mase": mase_group(naive, insample, target, frequency_by_group[group_name]),
                "forecast_dir": str(result_dir.relative_to(PROJECT_ROOT)),
            }
            row["owa"] = 0.5 * (row["smape"] / row["naive_smape"] + row["mase"] / row["naive_mase"])
            detailed_rows.append(row)
            group_rows.append(row)

        official_summary = summarize_official(pd.DataFrame(group_rows), series_counts)
        for _, row in official_summary.iterrows():
            summary_rows.append({
                "source": "deep_tsl",
                "model": model,
                "case": case,
                "lr_tag": lr_tag,
                "group": row["group"],
                "series_count": int(row["series_count"]),
                "smape": row["smape"],
                "mase": row["mase"],
                "owa": row["owa"],
                "naive_smape": row["naive_smape"],
                "naive_mase": row["naive_mase"],
                "forecast_dir": str(result_dir.relative_to(PROJECT_ROOT)),
            })

    return pd.DataFrame(detailed_rows), pd.DataFrame(summary_rows)


def add_shallow_metric_files(summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_files = [
        ("shallow", PROJECT_ROOT / "results" / "long_input" / "m4_ols_results.csv"),
        ("shallow", PROJECT_ROOT / "results" / "long_input" / "m4_revin_ols_results.csv"),
        ("shallow", PROJECT_ROOT / "results" / "extra_models" / "m4_ridge_results.csv"),
        ("shallow", PROJECT_ROOT / "results" / "extra_models_fast" / "m4_knn_results.csv"),
    ]
    rows = []
    for source, path in metric_files:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        needed = {"case", "model_name", "seasonal_pattern", "series_count", "smape", "mase", "owa", "naive_smape", "naive_mase"}
        if not needed.issubset(df.columns):
            continue
        df = df.rename(columns={"model_name": "model", "seasonal_pattern": "group"})
        df["source"] = source
        df["lr_tag"] = ""
        df["forecast_dir"] = str(path.relative_to(PROJECT_ROOT))

        for (model, case), group_df in df.groupby(["model", "case"], sort=True):
            if set(group_df["group"]) >= set(SEASONAL_PATTERNS):
                official_summary = summarize_official(group_df, group_df.set_index("group")["series_count"].astype(int).to_dict())
                for _, row in official_summary.iterrows():
                    rows.append({
                        "source": source,
                        "model": model,
                        "case": case,
                        "lr_tag": "",
                        "group": row["group"],
                        "series_count": int(row["series_count"]),
                        "smape": row["smape"],
                        "mase": row["mase"],
                        "owa": row["owa"],
                        "naive_smape": row["naive_smape"],
                        "naive_mase": row["naive_mase"],
                        "forecast_dir": str(path.relative_to(PROJECT_ROOT)),
                    })

    if rows:
        return pd.concat([summary_df, pd.DataFrame(rows)], ignore_index=True)
    return summary_df


def make_average_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    average = summary_df[summary_df["group"].eq("Average")].copy()
    average["model_case"] = average["model"] + "_" + average["case"]
    average = average.sort_values(["source", "model", "case"])
    return average[["source", "model", "case", "smape", "mase", "owa", "naive_smape", "naive_mase", "forecast_dir"]]


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    detailed_deep, official_summary = evaluate_deep_forecast_dirs()
    official_all = add_shallow_metric_files(official_summary)

    for frame in [detailed_deep, official_summary, official_all]:
        metric_cols = ["smape", "mase", "owa", "naive_smape", "naive_mase"]
        for col in metric_cols:
            if col in frame.columns:
                frame[col] = frame[col].astype(float).round(6)

    deep_average_table = make_average_table(official_summary).round(6)
    average_table = make_average_table(official_all).round(6)

    detailed_deep.to_csv(TABLE_DIR / "m4_official_deep_by_seasonal.csv", index=False)
    official_summary.to_csv(TABLE_DIR / "m4_official_deep_smape_mase_owa_by_group.csv", index=False)
    deep_average_table.to_csv(TABLE_DIR / "m4_official_deep_smape_mase_owa_average.csv", index=False)
    official_all.to_csv(TABLE_DIR / "m4_official_all_smape_mase_owa_by_group.csv", index=False)
    average_table.to_csv(TABLE_DIR / "m4_official_smape_mase_owa_average.csv", index=False)

    xlsx_path = TABLE_DIR / "m4_official_smape_mase_owa.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        deep_average_table.to_excel(writer, sheet_name="Deep_Average", index=False)
        average_table.to_excel(writer, sheet_name="Average_with_baselines", index=False)
        official_summary.to_excel(writer, sheet_name="Deep_by_group", index=False)
        official_all.to_excel(writer, sheet_name="All_by_group", index=False)
        detailed_deep.to_excel(writer, sheet_name="Deep_6_seasonals", index=False)

    print(f"deep forecast dirs: {official_summary[official_summary['group'].eq('Average')].shape[0]}")
    print(f"average rows including shallow baselines: {average_table.shape[0]}")
    print(TABLE_DIR / "m4_official_deep_smape_mase_owa_average.csv")
    print(TABLE_DIR / "m4_official_smape_mase_owa_average.csv")
    print(xlsx_path)


if __name__ == "__main__":
    main()
