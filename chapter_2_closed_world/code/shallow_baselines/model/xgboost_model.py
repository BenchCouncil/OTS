from __future__ import annotations

from pathlib import Path

import numpy as np

from .ols import Evaluation, OLSForecaster
from .sampled import SampledWindowConfig, SampledWindowMixin


class XGBoostForecaster(SampledWindowMixin, OLSForecaster):
    """Direct multi-horizon forecaster backed by xgboost when available."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        fit_intercept: bool = True,
        ridge_alpha: float = 0.0,
        max_train_examples: int = 12000,
        feature_points: int = 12,
        max_outputs: int = 16,
        num_boost_round: int = 80,
        clip_quantile: float = 0.002,
        predict_batch_size: int = 8192,
    ) -> None:
        super().__init__(
            seq_len=seq_len,
            pred_len=pred_len,
            fit_intercept=fit_intercept,
            ridge_alpha=ridge_alpha,
        )
        self.sample_config = SampledWindowConfig(
            max_train_examples=max_train_examples,
            feature_points=feature_points,
            clip_quantile=clip_quantile,
        )
        self.max_outputs = max_outputs
        self.num_boost_round = num_boost_round
        self.predict_batch_size = predict_batch_size
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.y_low_: np.ndarray | None = None
        self.y_high_: np.ndarray | None = None
        self.anchor_indices_: np.ndarray | None = None
        self.boosters_: list[object] = []

    def fit(self, data: np.ndarray, batch_windows: int = 32) -> "XGBoostForecaster":
        x_train, y_train, n_windows, n_examples = self._sample_xy_from_data(data)
        self._fit_from_xy(x_train, y_train)
        self.n_train_windows_ = n_windows
        self.n_train_examples_ = n_examples
        return self

    def fit_collection(
        self,
        series_list: list[np.ndarray],
        batch_windows: int = 128,
    ) -> "XGBoostForecaster":
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
        if not self.boosters_ or self.anchor_indices_ is None:
            raise RuntimeError("Call fit() before save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            max_train_examples=self.sample_config.max_train_examples,
            feature_points=self.sample_config.feature_points,
            max_outputs=self.max_outputs,
            num_boost_round=self.num_boost_round,
            anchor_indices=self.anchor_indices_,
            n_boosters=len(self.boosters_),
            n_train_windows=self.n_train_windows_,
            n_train_examples=self.n_train_examples_,
        )

    def _fit_from_xy(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        xgb = self._require_xgboost()
        features = self._fit_feature_scaler(self._window_features(x_train))
        self._fit_clip_bounds(y_train)
        self.anchor_indices_ = self._anchor_indices()
        self.boosters_ = []
        params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 5.0,
            "tree_method": "hist",
            "verbosity": 0,
        }
        for horizon_index in self.anchor_indices_:
            dtrain = xgb.DMatrix(features, label=y_train[:, int(horizon_index)])
            booster = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)
            self.boosters_.append(booster)

    def _predict_matrix(self, x_values: np.ndarray) -> np.ndarray:
        xgb = self._require_xgboost()
        if not self.boosters_ or self.anchor_indices_ is None:
            raise RuntimeError("Call fit() before predict/evaluate.")

        features = self._transform_features(self._window_features(x_values))
        predictions = np.empty((features.shape[0], self.pred_len), dtype=np.float64)
        horizons = np.arange(self.pred_len, dtype=np.float64)
        for start in range(0, len(features), self.predict_batch_size):
            stop = min(start + self.predict_batch_size, len(features))
            dmatrix = xgb.DMatrix(features[start:stop])
            anchor_pred = np.column_stack([booster.predict(dmatrix) for booster in self.boosters_])
            if len(self.anchor_indices_) == self.pred_len:
                block_pred = anchor_pred
            elif len(self.anchor_indices_) == 1:
                block_pred = np.repeat(anchor_pred, self.pred_len, axis=1)
            else:
                block_pred = np.vstack(
                    [
                        np.interp(horizons, self.anchor_indices_, row)
                        for row in anchor_pred
                    ]
                )
            predictions[start:stop] = block_pred
        return self._clip_predictions(predictions)

    def _anchor_indices(self) -> np.ndarray:
        n_outputs = max(1, min(self.pred_len, self.max_outputs))
        if n_outputs == self.pred_len:
            return np.arange(self.pred_len, dtype=np.int64)
        return np.unique(np.linspace(0, self.pred_len - 1, n_outputs, dtype=np.int64))

    @staticmethod
    def _require_xgboost():
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it in the active Python environment "
                "before running --model xgboost."
            ) from exc
        return xgb
