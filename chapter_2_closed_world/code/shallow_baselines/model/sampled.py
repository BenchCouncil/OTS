from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampledWindowConfig:
    max_train_examples: int = 20000
    feature_points: int = 8
    clip_quantile: float = 0.002


class SampledWindowMixin:
    """Utilities for non-parametric/tree models built on sampled windows."""

    sample_config: SampledWindowConfig
    feature_mean_: np.ndarray | None
    feature_scale_: np.ndarray | None
    y_low_: np.ndarray | None
    y_high_: np.ndarray | None

    def _sample_xy_from_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        data = self._validate_series(data)
        n_windows = self._num_windows(data)
        if n_windows <= 0:
            raise ValueError("Not enough rows to build one training window.")

        channels = data.shape[1]
        max_windows = max(1, self.sample_config.max_train_examples // max(1, channels))
        starts = self._sample_starts(n_windows, max_windows)
        x_train, y_train = self._build_xy(data, starts)
        x_train, y_train = self._thin_rows(x_train, y_train)
        return x_train, y_train, n_windows, n_windows * channels

    def _sample_xy_from_collection(
        self,
        series_list: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        valid: list[tuple[np.ndarray, int]] = []
        total_windows = 0
        for series in series_list:
            values = self._clean_univariate(series)
            n_windows = len(values) - self.seq_len - self.pred_len + 1
            if n_windows <= 0:
                continue
            valid.append((values, n_windows))
            total_windows += n_windows

        if total_windows <= 0:
            raise ValueError("Not enough M4 rows to build one collection training window.")

        max_series = min(len(valid), max(1, self.sample_config.max_train_examples))
        selected_positions = self._sample_starts(len(valid), max_series)
        per_series_cap = max(1, self.sample_config.max_train_examples // max(1, len(selected_positions)))

        x_parts: list[np.ndarray] = []
        y_parts: list[np.ndarray] = []
        sampled_rows = 0
        for pos in selected_positions:
            values, n_windows = valid[int(pos)]
            starts = self._sample_starts(n_windows, min(n_windows, per_series_cap))
            x_part, y_part = self._build_xy(values[:, None], starts)
            x_parts.append(x_part)
            y_parts.append(y_part)
            sampled_rows += len(x_part)
            if sampled_rows >= self.sample_config.max_train_examples:
                break

        x_train = np.concatenate(x_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)
        x_train, y_train = self._thin_rows(x_train, y_train)
        return x_train, y_train, total_windows, total_windows

    def _sample_starts(self, n_windows: int, cap: int) -> np.ndarray:
        cap = max(1, int(cap))
        if n_windows <= cap:
            return np.arange(n_windows, dtype=np.int64)
        starts = np.linspace(0, n_windows - 1, cap, dtype=np.int64)
        return np.unique(starts)

    def _thin_rows(self, x_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_rows = self.sample_config.max_train_examples
        if len(x_train) <= max_rows:
            return x_train, y_train
        rows = np.linspace(0, len(x_train) - 1, max_rows, dtype=np.int64)
        return x_train[rows], y_train[rows]

    def _window_features(self, x_values: np.ndarray) -> np.ndarray:
        x_values = np.asarray(x_values, dtype=np.float64)
        if x_values.ndim == 1:
            x_values = x_values[None, :]
        n_points = max(1, min(self.sample_config.feature_points, x_values.shape[1]))
        indices = np.linspace(0, x_values.shape[1] - 1, n_points, dtype=np.int64)
        sampled = x_values[:, indices]
        first = x_values[:, :1]
        last = x_values[:, -1:]
        mean = x_values.mean(axis=1, keepdims=True)
        std = x_values.std(axis=1, keepdims=True)
        trend = (last - first) / max(1, x_values.shape[1] - 1)
        return np.concatenate([last, mean, std, trend, sampled], axis=1)

    def _fit_feature_scaler(self, features: np.ndarray) -> np.ndarray:
        self.feature_mean_ = features.mean(axis=0)
        scale = features.std(axis=0)
        scale[scale < 1e-8] = 1.0
        self.feature_scale_ = scale
        return self._transform_features(features)

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("Feature scaler is not fitted.")
        return (features - self.feature_mean_) / self.feature_scale_

    def _fit_clip_bounds(self, y_train: np.ndarray) -> None:
        q = float(self.sample_config.clip_quantile)
        if q <= 0:
            self.y_low_ = None
            self.y_high_ = None
            return
        self.y_low_ = np.quantile(y_train, q, axis=0)
        self.y_high_ = np.quantile(y_train, 1.0 - q, axis=0)

    def _clip_predictions(self, predictions: np.ndarray) -> np.ndarray:
        if self.y_low_ is None or self.y_high_ is None:
            return predictions
        return np.clip(predictions, self.y_low_, self.y_high_)
