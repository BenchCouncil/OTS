from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataloader import TimeSeriesForecastDataset, get_dataset_config, list_benchmark_names

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent


KNOWN_PERIODS = {
    "ETTh1": 24,
    "ETTh2": 24,
    "ETTm1": 96,
    "ETTm2": 96,
    "electricity": 24,
    "traffic": 24,
    "weather": 144,
    "exchange_rate": 7,
    "illness": 52,
}


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def mean_channel_corr(data: np.ndarray, lag: int) -> float:
    if data.shape[0] <= lag + 2:
        return float("nan")
    values = []
    for i in range(data.shape[1]):
        values.append(safe_corr(data[:-lag, i], data[lag:, i]))
    return float(np.nanmean(values))


def trend_r2_and_slope(data: np.ndarray) -> tuple[float, float]:
    t = np.linspace(-0.5, 0.5, data.shape[0])
    t_centered = t - t.mean()
    denom = float(np.sum(t_centered * t_centered))
    r2_values = []
    slope_values = []
    for i in range(data.shape[1]):
        y = data[:, i]
        y_centered = y - y.mean()
        slope = float(np.sum(t_centered * y_centered) / denom)
        fitted = slope * t_centered + y.mean()
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        ss_res = float(np.sum((y - fitted) ** 2))
        r2_values.append(0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot))
        slope_values.append(abs(slope))
    return float(np.nanmean(r2_values)), float(np.nanmean(slope_values))


def split_dataset_values(dataset: str, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    config = get_dataset_config(dataset)
    train = TimeSeriesForecastDataset(
        config,
        data_root=data_root,
        flag="train",
        seq_len=config.default_seq_len,
        pred_len=config.default_pred_lens[0],
        scale=True,
    )
    test = TimeSeriesForecastDataset(
        config,
        data_root=data_root,
        flag="test",
        seq_len=config.default_seq_len,
        pred_len=config.default_pred_lens[0],
        scale=True,
        scaler=train.scaler,
    )
    return train.data, test.data


def compute_dataset_properties(data_root: Path) -> pd.DataFrame:
    rows = []
    for dataset in list_benchmark_names():
        train, test = split_dataset_values(dataset, data_root)
        full = np.concatenate([train, test], axis=0)
        period = KNOWN_PERIODS[dataset]
        trend_r2, trend_slope_abs = trend_r2_and_slope(full)
        roughness = float(np.nanmean(np.std(np.diff(full, axis=0), axis=0)))
        train_test_mean_shift = float(np.nanmean(np.abs(test.mean(axis=0) - train.mean(axis=0))))
        train_std = np.std(train, axis=0)
        test_std = np.std(test, axis=0)
        train_test_scale_shift = float(np.nanmean(np.abs(np.log((test_std + 1e-8) / (train_std + 1e-8)))))
        n = min(len(train), len(test))
        train_tail = train[-n:]
        test_head = test[:n]
        test_rev = test[::-1][:n]
        forward_similarity = float(np.nanmean([safe_corr(train_tail[:, i], test_head[:, i]) for i in range(full.shape[1])]))
        reverse_similarity = float(np.nanmean([safe_corr(train_tail[:, i], test_rev[:, i]) for i in range(full.shape[1])]))
        rows.append(
            {
                "dataset": dataset,
                "channels": full.shape[1],
                "train_rows": train.shape[0],
                "test_rows": test.shape[0],
                "known_period": period,
                "trend_r2": trend_r2,
                "trend_slope_abs": trend_slope_abs,
                "lag1_autocorr": mean_channel_corr(full, 1),
                "period_autocorr": mean_channel_corr(full, period),
                "roughness_std_diff": roughness,
                "train_test_mean_shift": train_test_mean_shift,
                "train_test_scale_shift": train_test_scale_shift,
                "forward_similarity": forward_similarity,
                "reverse_similarity": reverse_similarity,
                "reverse_similarity_gain": reverse_similarity - forward_similarity,
            }
        )
    return pd.DataFrame(rows)


def load_reversal_ratios(nn_path: Path, reversal_path: Path) -> pd.DataFrame:
    nn = pd.read_csv(nn_path)
    nn.insert(0, "case", "NN")
    reversal = pd.read_csv(reversal_path)
    common = ["case", "dataset", "pred_len", "mse", "mae"]
    all_results = pd.concat([nn[common], reversal[common]], ignore_index=True)
    base = all_results[all_results["case"] == "NN"][
        ["dataset", "pred_len", "mse", "mae"]
    ].rename(columns={"mse": "nn_mse", "mae": "nn_mae"})
    ratios = all_results.merge(base, on=["dataset", "pred_len"])
    ratios["mse_ratio_vs_nn"] = ratios["mse"] / ratios["nn_mse"]
    ratios["mae_ratio_vs_nn"] = ratios["mae"] / ratios["nn_mae"]
    return ratios


def summarize_reversal_by_dataset(ratios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in ratios[ratios["case"] != "NN"].groupby("dataset"):
        row = {"dataset": dataset}
        for case, case_group in group.groupby("case"):
            row[f"{case}_mse_ratio"] = float(case_group["mse_ratio_vs_nn"].mean())
            row[f"{case}_mae_ratio"] = float(case_group["mae_ratio_vs_nn"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def correlation_table(merged: pd.DataFrame) -> pd.DataFrame:
    metrics = [
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
    ]
    targets = [
        "RN_mse_ratio",
        "NR_mse_ratio",
        "RR_mse_ratio",
        "RN_mae_ratio",
        "NR_mae_ratio",
        "RR_mae_ratio",
    ]
    rows = []
    for metric in metrics:
        for target in targets:
            rows.append(
                {
                    "metric": metric,
                    "target": target,
                    "pearson": merged[metric].corr(merged[target], method="pearson"),
                    "spearman": merged[metric].corr(merged[target], method="spearman"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=str(WORKSPACE_DIR / "dataset"))
    parser.add_argument("--nn-results", default=str(WORKSPACE_DIR / "results/full/ols_results.csv"))
    parser.add_argument("--reversal-results", default=str(WORKSPACE_DIR / "results/reversal/ols_results.csv"))
    parser.add_argument("--output-dir", default=str(WORKSPACE_DIR / "results/property_analysis"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    properties = compute_dataset_properties(Path(args.data_root))
    ratios = summarize_reversal_by_dataset(
        load_reversal_ratios(Path(args.nn_results), Path(args.reversal_results))
    )
    merged = properties.merge(ratios, on="dataset")
    correlations = correlation_table(merged)

    properties.to_csv(output_dir / "dataset_properties.csv", index=False)
    merged.to_csv(output_dir / "properties_with_reversal_ratios.csv", index=False)
    correlations.to_csv(output_dir / "property_reversal_correlations.csv", index=False)

    print("Dataset properties + reversal ratios")
    print(merged.round(4).to_string(index=False))
    print("\nTop absolute Spearman correlations")
    top = correlations.assign(abs_spearman=correlations["spearman"].abs()).sort_values(
        "abs_spearman", ascending=False
    )
    print(top.head(18).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
