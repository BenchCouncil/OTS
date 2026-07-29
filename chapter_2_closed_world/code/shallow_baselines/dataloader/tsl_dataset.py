from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    folder: str
    filename: str
    data_kind: str
    default_seq_len: int = 96
    default_pred_lens: tuple[int, ...] = (96, 192, 336, 720)
    target: str = "OT"
    features: str = "M"


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "ETTh1": BenchmarkConfig("ETTh1", "ETT-small", "ETTh1.csv", "ett_hour"),
    "ETTh2": BenchmarkConfig("ETTh2", "ETT-small", "ETTh2.csv", "ett_hour"),
    "ETTm1": BenchmarkConfig("ETTm1", "ETT-small", "ETTm1.csv", "ett_minute"),
    "ETTm2": BenchmarkConfig("ETTm2", "ETT-small", "ETTm2.csv", "ett_minute"),
    "electricity": BenchmarkConfig("electricity", "electricity", "electricity.csv", "custom"),
    "exchange_rate": BenchmarkConfig("exchange_rate", "exchange_rate", "exchange_rate.csv", "custom"),
    "traffic": BenchmarkConfig("traffic", "traffic", "traffic.csv", "custom"),
    "weather": BenchmarkConfig("weather", "weather", "weather.csv", "custom"),
    "illness": BenchmarkConfig(
        "illness",
        "illness",
        "national_illness.csv",
        "custom",
        default_seq_len=104,
        default_pred_lens=(24, 36, 48, 60),
    ),
}


class StandardScaler:
    """Small numpy equivalent of sklearn.preprocessing.StandardScaler."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "StandardScaler":
        self.mean_ = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale[scale == 0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        return (values - self.mean_) / self.scale_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before inverse_transform().")
        return values * self.scale_ + self.mean_


class TimeSeriesForecastDataset:
    """Long-term forecasting dataset using Time-Series-Library split rules."""

    def __init__(
        self,
        config: BenchmarkConfig,
        data_root: str | Path,
        flag: str,
        seq_len: int,
        pred_len: int,
        features: str | None = None,
        target: str | None = None,
        scale: bool = True,
        scaler: StandardScaler | None = None,
    ) -> None:
        if flag not in {"train", "val", "test"}:
            raise ValueError("flag must be one of train, val, or test")

        self.config = config
        self.data_root = Path(data_root)
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features or config.features
        self.target = target or config.target
        self.scale = scale
        self.scaler = scaler or StandardScaler()

        self.raw_frame = self._read_frame()
        self.feature_names = self._select_feature_names(self.raw_frame)
        self.border1s, self.border2s = self._split_borders(len(self.raw_frame))
        self.border1, self.border2 = self._border_for_flag(flag)
        self.all_data = self._build_scaled_values()
        self.data = self.all_data[self.border1 : self.border2]

    @property
    def path(self) -> Path:
        return self.data_root / self.config.folder / self.config.filename

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_windows(self) -> int:
        return max(0, len(self.data) - self.seq_len - self.pred_len + 1)

    def _read_frame(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        frame = pd.read_csv(self.path)
        if "date" not in frame.columns:
            raise ValueError(f"{self.path} must contain a date column.")
        if self.target not in frame.columns:
            raise ValueError(f"{self.path} must contain target column {self.target!r}.")

        if self.config.data_kind == "custom":
            cols = list(frame.columns)
            cols.remove("date")
            cols.remove(self.target)
            frame = frame[["date"] + cols + [self.target]]
        return frame

    def _select_feature_names(self, frame: pd.DataFrame) -> list[str]:
        if self.features in {"M", "MS"}:
            return list(frame.columns[1:])
        if self.features == "S":
            return [self.target]
        raise ValueError("features must be one of M, MS, or S")

    def _split_borders(self, total_len: int) -> tuple[list[int], list[int]]:
        if self.config.data_kind == "ett_hour":
            border1s = [
                0,
                12 * 30 * 24 - self.seq_len,
                12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24,
                12 * 30 * 24 + 4 * 30 * 24,
                12 * 30 * 24 + 8 * 30 * 24,
            ]
        elif self.config.data_kind == "ett_minute":
            border1s = [
                0,
                12 * 30 * 24 * 4 - self.seq_len,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
            ]
            border2s = [
                12 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
                12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
            ]
        elif self.config.data_kind == "custom":
            num_train = int(total_len * 0.7)
            num_test = int(total_len * 0.2)
            num_val = total_len - num_train - num_test
            border1s = [0, num_train - self.seq_len, total_len - num_test - self.seq_len]
            border2s = [num_train, num_train + num_val, total_len]
        else:
            raise ValueError(f"Unsupported data_kind: {self.config.data_kind}")

        if border1s[0] < 0 or border1s[1] < 0 or border1s[2] < 0:
            raise ValueError(
                f"seq_len={self.seq_len} is too large for {self.config.name} split borders."
            )
        if max(border2s) > total_len:
            raise ValueError(
                f"{self.config.name} has {total_len} rows but split needs {max(border2s)} rows."
            )
        return border1s, border2s

    def _border_for_flag(self, flag: str) -> tuple[int, int]:
        type_map = {"train": 0, "val": 1, "test": 2}
        idx = type_map[flag]
        return self.border1s[idx], self.border2s[idx]

    def _build_scaled_values(self) -> np.ndarray:
        values = self.raw_frame[self.feature_names].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if not self.scale:
            return values

        train_values = values[self.border1s[0] : self.border2s[0]]
        if self.scaler.mean_ is None:
            self.scaler.fit(train_values)
        return self.scaler.transform(values)

    def describe(self) -> dict[str, int | str]:
        return {
            "name": self.config.name,
            "flag": self.flag,
            "rows": len(self.data),
            "channels": self.n_channels,
            "windows": self.n_windows,
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "features": self.features,
            "target": self.target,
        }


def get_dataset_config(name: str) -> BenchmarkConfig:
    if name not in BENCHMARKS:
        valid = ", ".join(BENCHMARKS)
        raise KeyError(f"Unknown dataset {name!r}. Valid names: {valid}")
    return BENCHMARKS[name]


def list_benchmark_names() -> list[str]:
    return list(BENCHMARKS)


def iter_dataset_configs(names: Iterable[str]) -> list[BenchmarkConfig]:
    return [get_dataset_config(name) for name in names]
