from __future__ import annotations

from pathlib import Path

import numpy as np

from .ols import Evaluation, OLSForecaster
from .sampled import SampledWindowConfig, SampledWindowMixin


class KNNForecaster(SampledWindowMixin, OLSForecaster):
    """Approximate KNN forecaster over compact lag-window features."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        fit_intercept: bool = True,
        ridge_alpha: float = 0.0,
        n_neighbors: int = 4,
        candidate_pool: int = 32,
        max_train_examples: int = 12000,
        feature_points: int = 6,
        max_outputs: int = 24,
        clip_quantile: float = 0.002,
        predict_batch_size: int = 8192,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha,
        )
        self.n_neighbors = n_neighbors
        self.candidate_pool = candidate_pool
        self.max_outputs = max_outputs
        self.predict_batch_size = predict_batch_size
        self.sample_config = SampledWindowConfig(
            max_train_examples=max_train_examples,
            feature_points=feature_points,
            clip_quantile=clip_quantile,
        )
        self.train_features_: np.ndarray | None = None
        self.train_targets_: np.ndarray | None = None
        self.anchor_indices_: np.ndarray | None = None
        self.sorted_indices_: np.ndarray | None = None
        self.sorted_key_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.y_low_: np.ndarray | None = None
        self.y_high_: np.ndarray | None = None

    def fit(self, data: np.ndarray, batch_windows: int = 32) -> "KNNForecaster":
        x_train, y_train, n_windows, n_examples = self._sample_xy_from_data(data)
        self._fit_from_xy(x_train, y_train)
        self.n_train_windows_ = n_windows
        self.n_train_examples_ = n_examples
        return self

    def fit_collection(
        self,
        series_list: list[np.ndarray],
        batch_windows: int = 128,
    ) -> "KNNForecaster":
        x_train, y_train, n_windows, n_examples = self._sample_xy_from_collection(series_list)
        self._fit_from_xy(x_train, y_train)
        self.n_train_windows_ = n_windows
        self.n_train_examples_ = n_examples
        return self

    def predict(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=np.float64)
        if history.ndim == 1:
            history = history[:, None]
        if history.shape[0] != self.seq_len:
            raise ValueError(f"history must have seq_len={self.seq_len} rows.")
        pred_rows = self._predict_matrix(history.T)
        return pred_rows.T

    def evaluate(self, data: np.ndarray, batch_windows: int = 32) -> Evaluation:
        data = self._validate_series(data)
        n_windows = self._num_windows(data)
        if n_windows <= 0:
            raise ValueError("Not enough rows to build one evaluation window.")

        squared_error = 0.0
        absolute_error = 0.0
        n_values = 0
        for starts in self._chunk_starts(n_windows, batch_windows):
            x_batch, y_batch = self._build_xy(data, starts)
            pred = self._predict_matrix(x_batch)
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
        if self.train_features_ is None or self.train_targets_ is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            n_neighbors=self.n_neighbors,
            candidate_pool=self.candidate_pool,
            max_train_examples=self.sample_config.max_train_examples,
            feature_points=self.sample_config.feature_points,
            max_outputs=self.max_outputs,
            clip_quantile=self.sample_config.clip_quantile,
            anchor_indices=self.anchor_indices_,
            sampled_train_examples=len(self.train_targets_),
            n_train_windows=self.n_train_windows_,
            n_train_examples=self.n_train_examples_,
        )

    def _fit_from_xy(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        features = self._window_features(x_train)
        self.train_features_ = self._fit_feature_scaler(features)
        self.anchor_indices_ = self._anchor_indices()
        self._fit_clip_bounds(y_train)
        self.train_targets_ = np.asarray(y_train[:, self.anchor_indices_], dtype=np.float64)
        self.sorted_indices_ = np.argsort(self.train_features_[:, 0], kind="mergesort")
        self.sorted_key_ = self.train_features_[self.sorted_indices_, 0]

    def _predict_matrix(self, x_values: np.ndarray) -> np.ndarray:
        if (
            self.train_features_ is None
            or self.train_targets_ is None
            or self.anchor_indices_ is None
            or self.sorted_indices_ is None
            or self.sorted_key_ is None
        ):
            raise RuntimeError("Call fit() before predict/evaluate.")

        features = self._transform_features(self._window_features(x_values))
        predictions = np.empty((features.shape[0], self.pred_len), dtype=np.float64)
        n_train = len(self.train_targets_)
        k = max(1, min(self.n_neighbors, n_train))
        pool = max(k, min(self.candidate_pool, n_train))
        half = pool // 2
        offsets = np.arange(pool, dtype=np.int64) - half

        for start in range(0, len(features), self.predict_batch_size):
            stop = min(start + self.predict_batch_size, len(features))
            block = features[start:stop]
            positions = np.searchsorted(self.sorted_key_, block[:, 0], side="left")
            candidate_positions = np.clip(positions[:, None] + offsets[None, :], 0, n_train - 1)
            candidate_indices = self.sorted_indices_[candidate_positions]
            candidate_features = self.train_features_[candidate_indices]
            diff = candidate_features - block[:, None, :]
            distances = np.sum(diff * diff, axis=2)
            neighbor_cols = np.argpartition(distances, k - 1, axis=1)[:, :k]
            neighbor_indices = np.take_along_axis(candidate_indices, neighbor_cols, axis=1)
            neighbor_distances = np.take_along_axis(distances, neighbor_cols, axis=1)
            weights = 1.0 / (np.sqrt(neighbor_distances) + 1e-6)
            neighbor_targets = self.train_targets_[neighbor_indices]
            anchor_pred = np.sum(neighbor_targets * weights[:, :, None], axis=1) / np.sum(
                weights,
                axis=1,
                keepdims=True,
            )
            predictions[start:stop] = self._expand_anchors(anchor_pred)

        return self._clip_predictions(predictions)

    def _anchor_indices(self) -> np.ndarray:
        n_outputs = max(1, min(self.pred_len, self.max_outputs))
        if n_outputs == self.pred_len:
            return np.arange(self.pred_len, dtype=np.int64)
        return np.unique(np.linspace(0, self.pred_len - 1, n_outputs, dtype=np.int64))

    def _expand_anchors(self, anchor_pred: np.ndarray) -> np.ndarray:
        if self.anchor_indices_ is None:
            raise RuntimeError("Anchor indices are not fitted.")
        if len(self.anchor_indices_) == self.pred_len:
            return anchor_pred
        if len(self.anchor_indices_) == 1:
            return np.repeat(anchor_pred, self.pred_len, axis=1)
        horizons = np.arange(self.pred_len, dtype=np.float64)
        return np.vstack([np.interp(horizons, self.anchor_indices_, row) for row in anchor_pred])
