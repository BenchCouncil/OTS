from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dataloader import load_m4_seasonal
from .exp_ols import build_model, normalize_model_name, normalize_reversal_case, reversal_flags


@dataclass
class M4ExperimentResult:
    case: str
    model_name: str
    seasonal_pattern: str
    seq_len: int
    pred_len: int
    train_reversed: bool
    test_reversed: bool
    series_count: int
    train_windows: int
    mse: float
    mae: float
    smape: float
    mase: float
    owa: float
    naive_smape: float
    naive_mase: float
    seconds: float
    checkpoint: str


class M4OLSExperiment:
    def __init__(
        self,
        root_path: str | Path = "dataset/m4",
        seasonal_pattern: str = "Yearly",
        seq_len: int | None = None,
        fit_intercept: bool = True,
        ridge_alpha: float = 0.0,
        batch_windows: int = 128,
        output_dir: str | Path = "results",
        model_dir: str | Path = "checkpoints",
        case: str = "NN",
        model_name: str = "ols",
    ) -> None:
        self.root_path = Path(root_path)
        self.seasonal_pattern = seasonal_pattern
        self.seq_len = seq_len
        self.fit_intercept = fit_intercept
        self.ridge_alpha = ridge_alpha
        self.batch_windows = batch_windows
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)
        self.case = normalize_reversal_case(case)
        self.train_reversed, self.test_reversed = reversal_flags(self.case)
        self.model_name = normalize_model_name(model_name)

    def run(self) -> M4ExperimentResult:
        start_time = time.perf_counter()
        dataset = load_m4_seasonal(self.root_path, self.seasonal_pattern)
        seq_len = self.seq_len or dataset.default_seq_len

        model = build_model(
            model_name=self.model_name,
            seq_len=seq_len,
            pred_len=dataset.horizon,
            fit_intercept=self.fit_intercept,
            ridge_alpha=self.ridge_alpha,
        )
        train_series = self._reverse_collection(dataset.train_series, self.train_reversed)
        model.fit_collection(train_series, batch_windows=self.batch_windows)

        squared_error = 0.0
        absolute_error = 0.0
        smape_total = 0.0
        mase_total = 0.0
        naive_smape_total = 0.0
        naive_mase_total = 0.0
        n_values = 0
        n_series = 0
        eps = 1e-8

        for train, actual in zip(dataset.train_series, dataset.test_series, strict=True):
            history, target = self._build_eval_pair(train, actual, seq_len, dataset.horizon)
            pred = model.predict_univariate(history)[: len(target)]
            naive_pred = self._seasonal_naive_forecast(history, len(target), dataset.frequency)
            mase_scale = self._mase_scale(train, dataset.frequency, eps)
            err = pred - target
            naive_err = naive_pred - target
            squared_error += float(np.sum(err * err))
            absolute_error += float(np.sum(np.abs(err)))
            smape_total += float(np.sum(200.0 * np.abs(err) / (np.abs(pred) + np.abs(target) + eps)))
            mase_total += float(np.mean(np.abs(err)) / mase_scale)
            naive_smape_total += float(
                np.sum(200.0 * np.abs(naive_err) / (np.abs(naive_pred) + np.abs(target) + eps))
            )
            naive_mase_total += float(np.mean(np.abs(naive_err)) / mase_scale)
            n_values += int(len(target))
            n_series += 1

        smape = smape_total / n_values
        mase = mase_total / n_series
        naive_smape = naive_smape_total / n_values
        naive_mase = naive_mase_total / n_series

        checkpoint = (
            self.model_dir
            / f"{self.model_name}_m4_{self.case}_{self.seasonal_pattern}_s{seq_len}.npz"
        )
        model.save(checkpoint)

        result = M4ExperimentResult(
            case=self.case,
            model_name=self.model_name,
            seasonal_pattern=self.seasonal_pattern,
            seq_len=seq_len,
            pred_len=dataset.horizon,
            train_reversed=self.train_reversed,
            test_reversed=self.test_reversed,
            series_count=dataset.n_series,
            train_windows=model.n_train_windows_,
            mse=squared_error / n_values,
            mae=absolute_error / n_values,
            smape=smape,
            mase=mase,
            owa=0.5 * (smape / naive_smape + mase / naive_mase),
            naive_smape=naive_smape,
            naive_mase=naive_mase,
            seconds=time.perf_counter() - start_time,
            checkpoint=str(checkpoint),
        )
        self._write_result(result)
        self._write_latest_json(result)
        return result

    def _build_eval_pair(
        self,
        train: np.ndarray,
        actual: np.ndarray,
        seq_len: int,
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        history = self._tail_with_padding(train, seq_len)
        target = np.asarray(actual, dtype=np.float64)[:horizon]
        segment = np.concatenate([history, target])
        if self.test_reversed:
            segment = segment[::-1].copy()
        return segment[:seq_len], segment[seq_len : seq_len + horizon]

    @staticmethod
    def _mase_scale(series: np.ndarray, frequency: int, eps: float) -> float:
        values = np.asarray(series, dtype=np.float64).reshape(-1)
        lag = max(1, min(int(frequency), len(values) - 1))
        if lag <= 0:
            return 1.0
        scale = float(np.mean(np.abs(values[lag:] - values[:-lag])))
        return scale if scale > eps else eps

    @staticmethod
    def _seasonal_naive_forecast(history: np.ndarray, horizon: int, frequency: int) -> np.ndarray:
        values = np.asarray(history, dtype=np.float64).reshape(-1)
        lag = max(1, min(int(frequency), len(values)))
        season = values[-lag:]
        reps = int(np.ceil(horizon / lag))
        return np.tile(season, reps)[:horizon]

    def _write_result(self, result: M4ExperimentResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"m4_{self.model_name}_results.csv"
        row = asdict(result)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _write_latest_json(self, result: M4ExperimentResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"latest_m4_{self.model_name}_result.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(result), handle, indent=2)

    @staticmethod
    def _reverse_collection(
        series_list: list[np.ndarray],
        should_reverse: bool,
    ) -> list[np.ndarray]:
        if not should_reverse:
            return series_list
        return [np.asarray(series, dtype=np.float64)[::-1].copy() for series in series_list]

    @staticmethod
    def _tail_with_padding(series: np.ndarray, seq_len: int) -> np.ndarray:
        values = np.asarray(series, dtype=np.float64).reshape(-1)
        if len(values) >= seq_len:
            return values[-seq_len:].copy()
        padded = np.zeros(seq_len, dtype=np.float64)
        padded[-len(values) :] = values
        return padded
