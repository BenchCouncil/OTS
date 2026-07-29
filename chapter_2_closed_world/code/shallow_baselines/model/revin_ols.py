from __future__ import annotations

from pathlib import Path

import numpy as np

from .ols import Evaluation, OLSForecaster


class RevINOLSForecaster(OLSForecaster):
    """Closed-form temporal OLS with reversible instance normalization.

    For each sliding window and channel, the input history is normalized by its
    own mean and standard deviation. Targets are normalized with the same input
    statistics during fitting. During evaluation, predictions are denormalized
    back to the window scale before MSE/MAE are computed.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        fit_intercept: bool = True,
        ridge_alpha: float = 0.0,
        rcond: float = 1e-10,
        eps: float = 1e-5,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha,
            rcond=rcond,
        )
        self.eps = eps

    def fit(self, data: np.ndarray, batch_windows: int = 32) -> "RevINOLSForecaster":
        data = self._validate_series(data)
        n_windows = self._num_windows(data)
        if n_windows <= 0:
            raise ValueError("Not enough rows to build one training window.")

        n_features = self.seq_len + int(self.fit_intercept)
        gram = np.zeros((n_features, n_features), dtype=np.float64)
        cross = np.zeros((n_features, self.pred_len), dtype=np.float64)

        for starts in self._chunk_starts(n_windows, batch_windows):
            x_batch, y_batch, _, _, _ = self._build_revin_xy(data, starts)
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
    ) -> "RevINOLSForecaster":
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
                x_batch, y_batch, _, _, _ = self._build_revin_xy(data, starts)
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
        if history.ndim == 1:
            history = history[:, None]
        if history.shape[0] != self.seq_len:
            raise ValueError(f"history must have seq_len={self.seq_len} rows.")

        history_by_channel = history.T
        mean = history_by_channel.mean(axis=1, keepdims=True)
        std = np.sqrt(history_by_channel.var(axis=1, keepdims=True) + self.eps)
        x_norm = (history_by_channel - mean) / std
        pred_norm = x_norm @ self.weights_ + self.intercept_
        pred = pred_norm * std + mean
        return pred.T

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
            x_batch, _, y_raw, mean, std = self._build_revin_xy(data, starts)
            pred_norm = x_batch @ self.weights_ + self.intercept_
            pred = pred_norm * std + mean
            err = pred - y_raw
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
            eps=self.eps,
            n_train_windows=self.n_train_windows_,
            n_train_examples=self.n_train_examples_,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RevINOLSForecaster":
        payload = np.load(path)
        model = cls(
            seq_len=int(payload["seq_len"]),
            pred_len=int(payload["pred_len"]),
            fit_intercept=bool(payload["fit_intercept"]),
            ridge_alpha=float(payload["ridge_alpha"]),
            eps=float(payload["eps"]),
        )
        model.weights_ = payload["weights"]
        model.intercept_ = payload["intercept"]
        model.n_train_windows_ = int(payload["n_train_windows"])
        model.n_train_examples_ = int(payload["n_train_examples"])
        return model

    def _build_revin_xy(
        self,
        data: np.ndarray,
        starts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        channels = data.shape[1]
        rows = len(starts) * channels
        x_batch = np.empty((rows, self.seq_len), dtype=np.float64)
        y_norm_batch = np.empty((rows, self.pred_len), dtype=np.float64)
        y_raw_batch = np.empty((rows, self.pred_len), dtype=np.float64)
        mean_batch = np.empty((rows, 1), dtype=np.float64)
        std_batch = np.empty((rows, 1), dtype=np.float64)

        cursor = 0
        for start in starts:
            x_raw = data[start : start + self.seq_len].T
            y_start = start + self.seq_len
            y_raw = data[y_start : y_start + self.pred_len].T
            mean = x_raw.mean(axis=1, keepdims=True)
            std = np.sqrt(x_raw.var(axis=1, keepdims=True) + self.eps)
            next_cursor = cursor + channels
            x_batch[cursor:next_cursor] = (x_raw - mean) / std
            y_norm_batch[cursor:next_cursor] = (y_raw - mean) / std
            y_raw_batch[cursor:next_cursor] = y_raw
            mean_batch[cursor:next_cursor] = mean
            std_batch[cursor:next_cursor] = std
            cursor = next_cursor

        return x_batch, y_norm_batch, y_raw_batch, mean_batch, std_batch
