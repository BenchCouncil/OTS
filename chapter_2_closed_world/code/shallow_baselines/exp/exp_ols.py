from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from dataloader import TimeSeriesForecastDataset, get_dataset_config
from model import KNNForecaster, OLSForecaster, RevINOLSForecaster, RidgeForecaster, XGBoostForecaster

from .reference_metrics import get_dlinear_baseline


@dataclass
class ExperimentResult:
    case: str
    dataset: str
    seq_len: int
    pred_len: int
    features: str
    train_reversed: bool
    test_reversed: bool
    channels: int
    train_windows: int
    test_windows: int
    mse: float
    mae: float
    dlinear_mse: float | None
    dlinear_mae: float | None
    mse_ratio_vs_dlinear: float | None
    mae_ratio_vs_dlinear: float | None
    seconds: float
    checkpoint: str


class OLSExperiment:
    def __init__(
        self,
        dataset: str,
        data_root: str | Path = "dataset",
        seq_len: int | None = None,
        pred_len: int | None = None,
        features: str | None = None,
        target: str | None = None,
        scale: bool = True,
        fit_intercept: bool = True,
        ridge_alpha: float = 0.0,
        batch_windows: int = 32,
        output_dir: str | Path = "results",
        model_dir: str | Path = "checkpoints",
        case: str = "NN",
        model_name: str = "ols",
    ) -> None:
        self.config = get_dataset_config(dataset)
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.seq_len = seq_len or self.config.default_seq_len
        if pred_len is None:
            raise ValueError("pred_len is required for OLSExperiment.")
        self.pred_len = pred_len
        self.features = features or self.config.features
        self.target = target or self.config.target
        self.scale = scale
        self.fit_intercept = fit_intercept
        self.ridge_alpha = ridge_alpha
        self.batch_windows = batch_windows
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.case = normalize_reversal_case(case)
        self.train_reversed, self.test_reversed = reversal_flags(self.case)
        self.model_name = normalize_model_name(model_name)

    def run(self) -> ExperimentResult:
        start_time = time.perf_counter()
        train_data = TimeSeriesForecastDataset(
            self.config,
            data_root=self.data_root,
            flag="train",
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            features=self.features,
            target=self.target,
            scale=self.scale,
        )
        test_data = TimeSeriesForecastDataset(
            self.config,
            data_root=self.data_root,
            flag="test",
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            features=self.features,
            target=self.target,
            scale=self.scale,
            scaler=train_data.scaler,
        )

        model = build_model(
            model_name=self.model_name,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            fit_intercept=self.fit_intercept,
            ridge_alpha=self.ridge_alpha,
        )
        train_values = reverse_if_needed(train_data.data, self.train_reversed)
        test_values = reverse_if_needed(test_data.data, self.test_reversed)

        model.fit(train_values, batch_windows=self.batch_windows)
        evaluation = model.evaluate(test_values, batch_windows=self.batch_windows)

        checkpoint = self.model_dir / (
            f"{self.model_name}_{self.case}_{self.dataset}_s{self.seq_len}_p{self.pred_len}_{self.features}.npz"
        )
        model.save(checkpoint)

        baseline = get_dlinear_baseline(self.dataset, self.pred_len)
        dlinear_mse = baseline[0] if baseline else None
        dlinear_mae = baseline[1] if baseline else None
        result = ExperimentResult(
            case=self.case,
            dataset=self.dataset,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            features=self.features,
            train_reversed=self.train_reversed,
            test_reversed=self.test_reversed,
            channels=train_data.n_channels,
            train_windows=train_data.n_windows,
            test_windows=test_data.n_windows,
            mse=evaluation.mse,
            mae=evaluation.mae,
            dlinear_mse=dlinear_mse,
            dlinear_mae=dlinear_mae,
            mse_ratio_vs_dlinear=(evaluation.mse / dlinear_mse if dlinear_mse else None),
            mae_ratio_vs_dlinear=(evaluation.mae / dlinear_mae if dlinear_mae else None),
            seconds=time.perf_counter() - start_time,
            checkpoint=str(checkpoint),
        )
        self._write_result(result)
        self._write_latest_json(result)
        return result

    def _write_result(self, result: ExperimentResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.model_name}_results.csv"
        row = asdict(result)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _write_latest_json(self, result: ExperimentResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        filename = "latest_result.json" if self.model_name == "ols" else f"latest_{self.model_name}_result.json"
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, indent=2)


def run_grid(
    datasets: Iterable[str],
    data_root: str | Path,
    seq_len: int | None,
    pred_lens: Iterable[int] | None,
    features: str | None,
    target: str | None,
    scale: bool,
    fit_intercept: bool,
    ridge_alpha: float,
    batch_windows: int,
    output_dir: str | Path,
    model_dir: str | Path,
    cases: Iterable[str] | None = None,
    model_name: str = "ols",
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    active_cases = [normalize_reversal_case(case) for case in (cases or ["NN"])]
    for dataset in datasets:
        config = get_dataset_config(dataset)
        active_seq_len = seq_len or config.default_seq_len
        active_pred_lens = list(pred_lens) if pred_lens else list(config.default_pred_lens)
        for case in active_cases:
            for pred_len in active_pred_lens:
                exp = OLSExperiment(
                    dataset=dataset,
                    data_root=data_root,
                    seq_len=active_seq_len,
                    pred_len=pred_len,
                    features=features,
                    target=target,
                    scale=scale,
                    fit_intercept=fit_intercept,
                    ridge_alpha=ridge_alpha,
                    batch_windows=batch_windows,
                    output_dir=output_dir,
                    model_dir=model_dir,
                    case=case,
                    model_name=model_name,
                )
                results.append(exp.run())
    return results


def normalize_reversal_case(case: str) -> str:
    normalized = case.upper()
    if normalized not in {"NN", "RN", "NR", "RR"}:
        raise ValueError("case must be one of NN, RN, NR, or RR.")
    return normalized


def reversal_flags(case: str) -> tuple[bool, bool]:
    normalized = normalize_reversal_case(case)
    return normalized[0] == "R", normalized[1] == "R"


def reverse_if_needed(data, should_reverse: bool):
    if not should_reverse:
        return data
    return data[::-1].copy()


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower()
    if normalized not in {"ols", "revin_ols", "ridge", "knn", "xgboost"}:
        raise ValueError("model_name must be one of ols, revin_ols, ridge, knn, or xgboost.")
    return normalized


def build_model(
    model_name: str,
    seq_len: int,
    pred_len: int,
    fit_intercept: bool,
    ridge_alpha: float,
):
    normalized = normalize_model_name(model_name)
    if normalized == "revin_ols":
        return RevINOLSForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha,
        )
    if normalized == "ridge":
        return RidgeForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha if ridge_alpha > 0 else 1.0,
        )
    if normalized == "knn":
        return KNNForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha,
        )
    if normalized == "xgboost":
        return XGBoostForecaster(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha,
        )
    return OLSForecaster(
        seq_len=seq_len,
        pred_len=pred_len,
        fit_intercept=fit_intercept,
        ridge_alpha=ridge_alpha,
    )
