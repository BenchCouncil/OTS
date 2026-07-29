from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


M4_HORIZONS = {
    "Yearly": 6,
    "Quarterly": 8,
    "Monthly": 18,
    "Weekly": 13,
    "Daily": 14,
    "Hourly": 48,
}

M4_HISTORY_MULTIPLIER = {
    "Yearly": 1.5,
    "Quarterly": 1.5,
    "Monthly": 1.5,
    "Weekly": 10,
    "Daily": 10,
    "Hourly": 10,
}


@dataclass
class M4SeasonalDataset:
    seasonal_pattern: str
    train_series: list[np.ndarray]
    test_series: list[np.ndarray]
    horizon: int
    frequency: int
    default_seq_len: int

    @property
    def n_series(self) -> int:
        return len(self.train_series)


def load_m4_seasonal(
    root_path: str | Path = "dataset/m4",
    seasonal_pattern: str = "Yearly",
) -> M4SeasonalDataset:
    if seasonal_pattern not in M4_HORIZONS:
        valid = ", ".join(M4_HORIZONS)
        raise ValueError(f"Unknown M4 seasonal pattern {seasonal_pattern!r}. Valid: {valid}")

    root = Path(root_path)
    info_path = root / "M4-info.csv"
    train_path = root / "training.npz"
    test_path = root / "test.npz"
    for path in (info_path, train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(f"M4 file not found: {path}")

    info = pd.read_csv(info_path)
    train_values = np.load(train_path, allow_pickle=True)
    test_values = np.load(test_path, allow_pickle=True)
    mask = info["SP"].to_numpy() == seasonal_pattern
    horizon = M4_HORIZONS[seasonal_pattern]
    frequency = int(info.loc[mask, "Frequency"].iloc[0])
    default_seq_len = int(M4_HISTORY_MULTIPLIER[seasonal_pattern] * horizon)

    return M4SeasonalDataset(
        seasonal_pattern=seasonal_pattern,
        train_series=[np.asarray(item, dtype=np.float64) for item in train_values[mask]],
        test_series=[np.asarray(item, dtype=np.float64)[:horizon] for item in test_values[mask]],
        horizon=horizon,
        frequency=frequency,
        default_seq_len=default_seq_len,
    )
