from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Evaluation:
    mse: float
    mae: float
    n_values: int
    n_windows: int


class OLSForecaster:
    """Closed-form temporal OLS forecaster.

    The default shared-channel formulation treats every channel in every
    sliding window as one sample and solves one temporal map:

        [seq_len] -> [pred_len]

    This mirrors the shared temporal linear layer used by common LTSF linear
    baselines while keeping large multivariate datasets tractable.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        fit_intercept: bool = True,
        ridge_alpha: float = 0.0,
        rcond: float = 1e-10,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.fit_intercept = fit_intercept
        self.ridge_alpha = ridge_alpha
        self.rcond = rcond
        self.weights_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.n_train_windows_: int = 0
        self.n_train_examples_: int = 0

    def fit(self, data: np.ndarray, batch_windows: int = 32) -> "OLSForecaster":
        data = self._validate_series(data)
        n_windows = self._num_windows(data)
        if n_windows <= 0:
            raise ValueError("Not enough rows to build one training window.")

        n_features = self.seq_len + int(self.fit_intercept)
        gram = np.zeros((n_features, n_features), dtype=np.float64)
        cross = np.zeros((n_features, self.pred_len), dtype=np.float64)

        for starts in self._chunk_starts(n_windows, batch_windows):
            x_batch, y_batch = self._build_xy(data, starts)
            self._accumulate_normal_equations(gram, cross, x_batch, y_batch)

        if self.ridge_alpha > 0:
            reg = np.eye(n_features, dtype=np.float64) * self.ridge_alpha
            if self.fit_intercept:
                reg[-1, -1] = 0.0
            gram = gram + reg

        solution = np.linalg.pinv(gram, rcond=self.rcond) @ cross
        if self.fit_intercept:
            self.weights_ = solution[:-1]
            self.intercept_ = solution[-1]
        else:
            self.weights_ = solution
            self.intercept_ = np.zeros(self.pred_len, dtype=np.float64)

        self.n_train_windows_ = n_windows
        self.n_train_examples_ = n_windows * data.shape[1]
        return self

    def fit_collection(
        self,
        series_list: list[np.ndarray],
        batch_windows: int = 128,
    ) -> "OLSForecaster":
        """Fit OLS from a ragged collection of univariate series."""

        n_features = self.seq_len + int(self.fit_intercept)
        gram = np.zeros((n_features, n_features), dtype=np.float64)
        cross = np.zeros((n_features, self.pred_len), dtype=np.float64)
        total_windows = 0

        for series in series_list:
            values = self._clean_univariate(series)
            n_windows = len(values) - self.seq_len - self.pred_len + 1
            if n_windows <= 0:
                continue
            data = values[:, None]
            for starts in self._chunk_starts(n_windows, batch_windows):
                x_batch, y_batch = self._build_xy(data, starts)
                self._accumulate_normal_equations(gram, cross, x_batch, y_batch)
            total_windows += n_windows

        if total_windows <= 0:
            raise ValueError("Not enough M4 rows to build one collection training window.")

        if self.ridge_alpha > 0:
            reg = np.eye(n_features, dtype=np.float64) * self.ridge_alpha
            if self.fit_intercept:
                reg[-1, -1] = 0.0
            gram = gram + reg

        solution = np.linalg.pinv(gram, rcond=self.rcond) @ cross
        if self.fit_intercept:
            self.weights_ = solution[:-1]
            self.intercept_ = solution[-1]
        else:
            self.weights_ = solution
            self.intercept_ = np.zeros(self.pred_len, dtype=np.float64)

        self.n_train_windows_ = total_windows
        self.n_train_examples_ = total_windows
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        if self.weights_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before predict().")
        history = np.asarray(history, dtype=np.float64)
        if history.shape[0] != self.seq_len:
            raise ValueError(f"history must have seq_len={self.seq_len} rows.")
        if history.ndim == 1:
            history = history[:, None]
        pred_rows = history.T @ self.weights_ + self.intercept_
        return pred_rows.T

    def predict_univariate(self, history: np.ndarray) -> np.ndarray:
        values = self._clean_univariate(history)
        padded = np.zeros(self.seq_len, dtype=np.float64)
        tail = values[-self.seq_len :]
        padded[-len(tail) :] = tail
        return self.predict(padded).reshape(-1)

    def evaluate(self, data: np.ndarray, batch_windows: int = 32) -> Evaluation:
        if self.weights_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before evaluate().")
        data = self._validate_series(data)
        n_windows = self._num_windows(data)
        if n_windows <= 0:
            raise ValueError("Not enough rows to build one evaluation window.")

        squared_error = 0.0
        absolute_error = 0.0
        n_values = 0

        for starts in self._chunk_starts(n_windows, batch_windows):
            x_batch, y_batch = self._build_xy(data, starts)
            pred = x_batch @ self.weights_ + self.intercept_
            err = pred - y_batch
            squared_error += float(np.sum(err * err))
            absolute_error += float(np.sum(np.abs(err)))
            n_values += int(err.size)

        return Evaluation(
            mse=squared_error / n_values,
            mae=absolute_error / n_values,
            n_values=n_values,
            n_windows=n_windows,
        )

    def save(self, path: str | Path) -> None:
        if self.weights_ is None or self.intercept_ is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights_,
            intercept=self.intercept_,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            fit_intercept=self.fit_intercept,
            ridge_alpha=self.ridge_alpha,
            n_train_windows=self.n_train_windows_,
            n_train_examples=self.n_train_examples_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "OLSForecaster":
        payload = np.load(path)
        model = cls(
            seq_len=int(payload["seq_len"]),
            pred_len=int(payload["pred_len"]),
            fit_intercept=bool(payload["fit_intercept"]),
            ridge_alpha=float(payload["ridge_alpha"]),
        )
        model.weights_ = payload["weights"]
        model.intercept_ = payload["intercept"]
        model.n_train_windows_ = int(payload["n_train_windows"])
        model.n_train_examples_ = int(payload["n_train_examples"])
        return model

    def _accumulate_normal_equations(
        self,
        gram: np.ndarray,
        cross: np.ndarray,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> None:
        seq = self.seq_len
        gram[:seq, :seq] += x_batch.T @ x_batch
        cross[:seq] += x_batch.T @ y_batch

        if self.fit_intercept:
            x_sum = x_batch.sum(axis=0)
            y_sum = y_batch.sum(axis=0)
            gram[:seq, -1] += x_sum
            gram[-1, :seq] += x_sum
            gram[-1, -1] += x_batch.shape[0]
            cross[-1] += y_sum

    def _build_xy(self, data: np.ndarray, starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        channels = data.shape[1]
        rows = len(starts) * channels
        x_batch = np.empty((rows, self.seq_len), dtype=np.float64)
        y_batch = np.empty((rows, self.pred_len), dtype=np.float64)

        cursor = 0
        for start in starts:
            x = data[start : start + self.seq_len].T
            y_start = start + self.seq_len
            y = data[y_start : y_start + self.pred_len].T
            next_cursor = cursor + channels
            x_batch[cursor:next_cursor] = x
            y_batch[cursor:next_cursor] = y
            cursor = next_cursor
        return x_batch, y_batch

    def _chunk_starts(self, n_windows: int, batch_windows: int) -> list[np.ndarray]:
        if batch_windows <= 0:
            raise ValueError("batch_windows must be positive.")
        return [
            np.arange(start, min(start + batch_windows, n_windows), dtype=np.int64)
            for start in range(0, n_windows, batch_windows)
        ]

    def _num_windows(self, data: np.ndarray) -> int:
        return int(data.shape[0] - self.seq_len - self.pred_len + 1)

    @staticmethod
    def _validate_series(data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2:
            raise ValueError("data must be a 2D array shaped [time, channels].")
        if not np.isfinite(data).all():
            raise ValueError("data contains nan or infinite values.")
        return data

    @staticmethod
    def _clean_univariate(series: np.ndarray) -> np.ndarray:
        values = np.asarray(series, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        return values
