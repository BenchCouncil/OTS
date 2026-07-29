#!/usr/bin/env python3
"""Controlled paired-world experiments for open-world time-series forecasting.

Experiment 1 creates two futures with identical real histories: an unmodified
future and a future affected by a forecast-time event Z. Experiment 2 reuses
the same pairs and varies model capacity and the fraction of pairs for which Z
is revealed. Negative controls use shuffled Z or a deterministic time placebo.

The script is self-contained (numpy, pandas, torch) and writes one directory
per run so that completed runs can be resumed without overwriting evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DATASET_FILES = {
    "ETTm1": "ETTm1.csv",
    "weather": "weather.csv",
    "ETTh2": "ETTh2.csv",
}

INFO_QUALITY = {
    "info0": 0.0,
    "info50": 0.5,
    "info100": 1.0,
}

Z_DIM = 8


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    architecture: str
    capacity: int
    condition: str
    seed: int

    @property
    def run_id(self) -> str:
        return (
            f"{self.dataset}__{self.architecture}{self.capacity}__"
            f"{self.condition}__seed{self.seed}"
        )


@dataclass
class SeriesBundle:
    name: str
    values: np.ndarray
    train_mean: float
    train_std: float
    train_end: int
    val_end: int
    test_end: int


def stable_uint64(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_bundle(data_root: Path, dataset: str) -> SeriesBundle:
    path = data_root / DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if "OT" not in frame.columns:
        raise ValueError(f"{path} has no OT target column")
    target = pd.to_numeric(frame["OT"], errors="coerce").astype(float)
    target = target.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if target.isna().any():
        raise ValueError(f"{path} still contains missing target values")
    raw = target.to_numpy(dtype=np.float32)

    if dataset == "ETTm1":
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = train_end + 8 * 30 * 24 * 4
    elif dataset == "ETTh2":
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = train_end + 8 * 30 * 24
    else:
        train_end = int(len(raw) * 0.7)
        test_size = int(len(raw) * 0.2)
        val_end = len(raw) - test_size
        test_end = len(raw)

    if test_end > len(raw):
        raise ValueError(f"Split for {dataset} exceeds series length {len(raw)}")
    train_mean = float(raw[:train_end].mean())
    train_std = float(raw[:train_end].std())
    if not math.isfinite(train_std) or train_std <= 1e-8:
        raise ValueError(f"Invalid training standard deviation for {dataset}")
    values = ((raw - train_mean) / train_std).astype(np.float32)
    return SeriesBundle(
        name=dataset,
        values=values,
        train_mean=train_mean,
        train_std=train_std,
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )


def split_starts(
    bundle: SeriesBundle,
    split: str,
    seq_len: int,
    pred_len: int,
    max_windows: int,
) -> np.ndarray:
    if split == "train":
        first = 0
        last = bundle.train_end - seq_len - pred_len
    elif split == "val":
        first = bundle.train_end - seq_len
        last = bundle.val_end - seq_len - pred_len
    elif split == "test":
        first = bundle.val_end - seq_len
        last = bundle.test_end - seq_len - pred_len
    else:
        raise ValueError(split)
    if last < first:
        raise ValueError(f"No {split} windows for {bundle.name}")
    all_starts = np.arange(first, last + 1, dtype=np.int64)
    if max_windows > 0 and len(all_starts) > max_windows:
        selected = np.linspace(0, len(all_starts) - 1, max_windows, dtype=np.int64)
        all_starts = all_starts[selected]
    return np.unique(all_starts)


def event_metadata(dataset: str, start: int, pred_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return forecast-time event metadata and its future response.

    Z = [event flag, 3 type indicators, signed amplitude, onset, duration,
    availability]. Availability is filled by the observation protocol later.
    Event effects are in training-standard-deviation units.
    """
    rng = np.random.default_rng(stable_uint64("event", dataset, start, pred_len))
    event_type = int(rng.integers(0, 3))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    amplitude = sign * float(rng.uniform(0.75, 1.50))
    onset = int(rng.integers(max(1, pred_len // 10), max(2, pred_len // 2)))
    max_duration = max(4, pred_len - onset)
    duration = int(rng.integers(max(3, pred_len // 8), max_duration + 1))

    z = np.zeros(Z_DIM, dtype=np.float32)
    z[0] = 1.0
    z[1 + event_type] = 1.0
    z[4] = amplitude
    z[5] = onset / max(1, pred_len - 1)
    z[6] = duration / pred_len

    t = np.arange(pred_len, dtype=np.float32)
    active = np.maximum(t - onset, 0.0)
    if event_type == 0:  # persistent level intervention
        kernel = (t >= onset).astype(np.float32)
    elif event_type == 1:  # transient pulse with exponential recovery
        decay = max(2.0, duration / 3.0)
        kernel = np.where(t >= onset, np.exp(-active / decay), 0.0).astype(np.float32)
    else:  # gradual mechanism/trend change
        denom = max(1.0, float(pred_len - 1 - onset))
        kernel = np.where(t >= onset, active / denom, 0.0).astype(np.float32)
    effect = (amplitude * kernel).astype(np.float32)
    return z, effect


def reveal_for_pair(dataset: str, split: str, start: int, quality: float) -> bool:
    if quality <= 0.0:
        return False
    if quality >= 1.0:
        return True
    value = stable_uint64("reveal", dataset, split, start) / float(2**64 - 1)
    return value < quality


class PairedWorldDataset(Dataset):
    def __init__(
        self,
        bundle: SeriesBundle,
        split: str,
        starts: np.ndarray,
        seq_len: int,
        pred_len: int,
        condition: str,
    ) -> None:
        self.bundle = bundle
        self.split = split
        self.starts = starts
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.condition = condition
        if condition not in {*INFO_QUALITY, "shuffled", "placebo"}:
            raise ValueError(condition)
        rng = np.random.default_rng(stable_uint64("shuffle-z", bundle.name, split))
        self.permutation = rng.permutation(len(starts) * 2)

    def __len__(self) -> int:
        return len(self.starts) * 2

    def _true_z(self, base_pos: int, branch: int) -> np.ndarray:
        start = int(self.starts[base_pos])
        if branch == 0:
            return np.zeros(Z_DIM, dtype=np.float32)
        z, _ = event_metadata(self.bundle.name, start, self.pred_len)
        return z

    def _observed_z(self, sample_index: int, base_pos: int, branch: int) -> np.ndarray:
        start = int(self.starts[base_pos])
        if self.condition in INFO_QUALITY:
            quality = INFO_QUALITY[self.condition]
            observed = np.zeros(Z_DIM, dtype=np.float32)
            if reveal_for_pair(self.bundle.name, self.split, start, quality):
                observed = self._true_z(base_pos, branch).copy()
                observed[7] = 1.0
            return observed
        if self.condition == "shuffled":
            source = int(self.permutation[sample_index])
            source_base = source // 2
            source_branch = source % 2
            observed = self._true_z(source_base, source_branch).copy()
            observed[7] = 1.0
            return observed

        # Deterministic time-placebo features are identical for paired branches.
        phase = start / max(1, len(self.bundle.values) - 1)
        observed = np.zeros(Z_DIM, dtype=np.float32)
        observed[0] = math.sin(2 * math.pi * phase)
        observed[1] = math.cos(2 * math.pi * phase)
        observed[2] = math.sin(4 * math.pi * phase)
        observed[3] = math.cos(4 * math.pi * phase)
        observed[4] = phase
        observed[5] = phase * phase
        observed[6] = math.sqrt(max(phase, 0.0))
        observed[7] = 1.0
        return observed

    def __getitem__(self, sample_index: int):
        base_pos = sample_index // 2
        branch = sample_index % 2
        start = int(self.starts[base_pos])
        history_end = start + self.seq_len
        future_end = history_end + self.pred_len
        history = self.bundle.values[start:history_end].copy()
        target = self.bundle.values[history_end:future_end].copy()
        if branch == 1:
            _, effect = event_metadata(self.bundle.name, start, self.pred_len)
            target += effect
        z_observed = self._observed_z(sample_index, base_pos, branch)
        return (
            torch.from_numpy(history),
            torch.from_numpy(z_observed),
            torch.from_numpy(target),
            torch.tensor(start, dtype=torch.int64),
            torch.tensor(branch, dtype=torch.int64),
        )


class MLPForecaster(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, capacity: int) -> None:
        super().__init__()
        middle = max(16, capacity // 2)
        self.network = nn.Sequential(
            nn.Linear(seq_len + Z_DIM, capacity),
            nn.GELU(),
            nn.LayerNorm(capacity),
            nn.Linear(capacity, middle),
            nn.GELU(),
            nn.Linear(middle, pred_len),
        )

    def forward(self, history: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([history, z], dim=-1))


class GRUForecaster(nn.Module):
    def __init__(self, pred_len: int, capacity: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=capacity, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(capacity + Z_DIM, capacity),
            nn.GELU(),
            nn.Linear(capacity, pred_len),
        )

    def forward(self, history: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history.unsqueeze(-1))
        return self.head(torch.cat([hidden[-1], z], dim=-1))


def build_model(spec: RunSpec, seq_len: int, pred_len: int) -> nn.Module:
    if spec.architecture == "mlp":
        return MLPForecaster(seq_len, pred_len, spec.capacity)
    if spec.architecture == "gru":
        return GRUForecaster(pred_len, spec.capacity)
    raise ValueError(spec.architecture)


def loader_for(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    branches: list[np.ndarray] = []
    for history, z, target, start, branch in loader:
        history = history.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            prediction = model(history, z)
        predictions.append(prediction.float().cpu().numpy())
        targets.append(target.numpy())
        starts.append(start.numpy())
        branches.append(branch.numpy())

    pred = np.concatenate(predictions)
    true = np.concatenate(targets)
    start_array = np.concatenate(starts)
    branch_array = np.concatenate(branches)
    order = np.lexsort((branch_array, start_array))
    pred = pred[order]
    true = true[order]
    start_array = start_array[order]
    branch_array = branch_array[order]
    if len(pred) % 2 or not np.all(branch_array.reshape(-1, 2) == np.array([0, 1])):
        raise RuntimeError("Paired evaluation ordering is invalid")

    pred_pair = pred.reshape(-1, 2, pred.shape[-1])
    true_pair = true.reshape(-1, 2, true.shape[-1])
    pair_ids = start_array.reshape(-1, 2)[:, 0]
    squared_error = (pred_pair - true_pair) ** 2
    absolute_error = np.abs(pred_pair - true_pair)
    true_difference = true_pair[:, 1] - true_pair[:, 0]
    predicted_difference = pred_pair[:, 1] - pred_pair[:, 0]

    pair_frame = pd.DataFrame(
        {
            "base_start": pair_ids,
            "pair_mse": squared_error.mean(axis=(1, 2)),
            "none_mse": squared_error[:, 0].mean(axis=1),
            "event_mse": squared_error[:, 1].mean(axis=1),
            "separation_mse": ((predicted_difference - true_difference) ** 2).mean(axis=1),
            "predicted_separation_energy": (predicted_difference**2).mean(axis=1),
            "true_separation_energy": (true_difference**2).mean(axis=1),
            "ambiguity_floor": 0.25 * (true_difference**2).mean(axis=1),
        }
    )
    metrics = {
        "mse": float(squared_error.mean()),
        "mae": float(absolute_error.mean()),
        "none_mse": float(squared_error[:, 0].mean()),
        "event_mse": float(squared_error[:, 1].mean()),
        "separation_mse": float(pair_frame["separation_mse"].mean()),
        "predicted_separation_energy": float(pair_frame["predicted_separation_energy"].mean()),
        "true_separation_energy": float(pair_frame["true_separation_energy"].mean()),
        "ambiguity_floor": float(pair_frame["ambiguity_floor"].mean()),
        "n_pairs": int(len(pair_frame)),
    }
    return metrics, pair_frame


def train_one(
    spec: RunSpec,
    bundle: SeriesBundle,
    args: argparse.Namespace,
    output_root: Path,
    device: torch.device,
) -> dict[str, object]:
    run_dir = output_root / "runs" / spec.run_id
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(spec.seed)

    starts = {
        "train": split_starts(bundle, "train", args.seq_len, args.pred_len, args.max_train_windows),
        "val": split_starts(bundle, "val", args.seq_len, args.pred_len, args.max_val_windows),
        "test": split_starts(bundle, "test", args.seq_len, args.pred_len, args.max_test_windows),
    }
    datasets = {
        split: PairedWorldDataset(
            bundle=bundle,
            split=split,
            starts=split_starts_array,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            condition=spec.condition,
        )
        for split, split_starts_array in starts.items()
    }
    loaders = {
        "train": loader_for(datasets["train"], args.batch_size, True, args.workers),
        "val": loader_for(datasets["val"], args.batch_size, False, args.workers),
        "test": loader_for(datasets["test"], args.batch_size, False, args.workers),
    }

    model = build_model(spec, args.seq_len, args.pred_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda" and not args.disable_amp
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history_rows: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for history, z, target, _, _ in loaders["train"]:
            history = history.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                prediction = model(history, z)
                loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(history)
            train_count += len(history)

        val_metrics, _ = evaluate(model, loaders["val"], device, use_amp)
        train_loss = train_loss_sum / max(1, train_count)
        history_rows.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_metrics["mse"]})
        if val_metrics["mse"] < best_val - args.min_delta:
            best_val = val_metrics["mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= args.patience:
            break

    if best_state is None:
        raise RuntimeError(f"No finite validation state for {spec.run_id}")
    model.load_state_dict(best_state)
    test_metrics, pair_frame = evaluate(model, loaders["test"], device, use_amp)
    duration = time.time() - started
    result: dict[str, object] = {
        **asdict(spec),
        "run_id": spec.run_id,
        "status": "completed",
        "duration_seconds": duration,
        "epochs_completed": len(history_rows),
        "best_val_mse": best_val,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "train_pairs": len(starts["train"]),
        "val_pairs": len(starts["val"]),
        "test_pairs": len(starts["test"]),
        "target_train_mean": bundle.train_mean,
        "target_train_std": bundle.train_std,
        **test_metrics,
    }
    pd.DataFrame(history_rows).to_csv(run_dir / "learning_curve.csv", index=False)
    pair_frame.to_csv(run_dir / "pair_losses.csv", index=False)
    torch.save(best_state, run_dir / "model.pt")
    with (run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump({**vars(args), **asdict(spec)}, handle, ensure_ascii=False, indent=2, default=str)
    temporary = metrics_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, metrics_path)
    return result


def experiment_specs(datasets: Iterable[str], seeds: Iterable[int]) -> list[RunSpec]:
    specs: dict[str, RunSpec] = {}
    for dataset in datasets:
        for seed in seeds:
            # Experiment 1: matched histories, valid Z, and two negative controls.
            for condition in ("info0", "info100", "shuffled", "placebo"):
                spec = RunSpec(dataset, "mlp", 128, condition, seed)
                specs[spec.run_id] = spec
            for condition in ("info0", "info100"):
                spec = RunSpec(dataset, "gru", 64, condition, seed)
                specs[spec.run_id] = spec

            # Experiment 2: model scaling crossed with information coverage.
            for capacity in (32, 128, 512):
                for condition in ("info0", "info50", "info100"):
                    spec = RunSpec(dataset, "mlp", capacity, condition, seed)
                    specs[spec.run_id] = spec
    return sorted(specs.values(), key=lambda item: item.run_id)


def audit_protocol(
    bundles: dict[str, SeriesBundle],
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    """Fail closed if the paired-world or chronological-split contract is broken."""
    audit_rows: list[dict[str, object]] = []
    for name, bundle in bundles.items():
        starts = {
            split: split_starts(bundle, split, args.seq_len, args.pred_len, 0)
            for split in ("train", "val", "test")
        }
        split_contract = {
            "train_target_within_train": bool(
                starts["train"].max() + args.seq_len + args.pred_len <= bundle.train_end
            ),
            "validation_target_after_train": bool(
                starts["val"].min() + args.seq_len >= bundle.train_end
            ),
            "validation_target_within_validation": bool(
                starts["val"].max() + args.seq_len + args.pred_len <= bundle.val_end
            ),
            "test_target_after_validation": bool(
                starts["test"].min() + args.seq_len >= bundle.val_end
            ),
            "test_target_within_test": bool(
                starts["test"].max() + args.seq_len + args.pred_len <= bundle.test_end
            ),
        }
        sample_starts = starts["test"][: min(32, len(starts["test"]))]
        info0 = PairedWorldDataset(bundle, "test", sample_starts, args.seq_len, args.pred_len, "info0")
        info100 = PairedWorldDataset(bundle, "test", sample_starts, args.seq_len, args.pred_len, "info100")
        placebo = PairedWorldDataset(bundle, "test", sample_starts, args.seq_len, args.pred_len, "placebo")
        pair_contracts = []
        for base_pos in range(len(sample_starts)):
            h0, z0, y0, _, _ = info0[2 * base_pos]
            h1, z1, y1, _, _ = info0[2 * base_pos + 1]
            oh0, oz0, oy0, _, _ = info100[2 * base_pos]
            oh1, oz1, oy1, _, _ = info100[2 * base_pos + 1]
            ph0, pz0, py0, _, _ = placebo[2 * base_pos]
            ph1, pz1, py1, _, _ = placebo[2 * base_pos + 1]
            pair_contracts.append(
                bool(
                    torch.equal(h0, h1)
                    and torch.equal(oh0, oh1)
                    and torch.equal(ph0, ph1)
                    and torch.equal(z0, z1)
                    and torch.count_nonzero(z0) == 0
                    and not torch.equal(oz0, oz1)
                    and torch.equal(pz0, pz1)
                    and torch.equal(y0, oy0)
                    and torch.equal(y1, oy1)
                    and torch.equal(y0, py0)
                    and torch.equal(y1, py1)
                    and not torch.equal(y0, y1)
                )
            )
        passed = all(split_contract.values()) and all(pair_contracts)
        audit_rows.append(
            {
                "dataset": name,
                "passed": passed,
                "pairs_checked": len(pair_contracts),
                "pair_contracts_passed": int(sum(pair_contracts)),
                **split_contract,
            }
        )
        if not passed:
            raise RuntimeError(f"Protocol audit failed for {name}")
    with (output_root / "protocol_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_rows, handle, ensure_ascii=False, indent=2)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(draws, dtype=np.float64)
    chunk = 200
    for offset in range(0, draws, chunk):
        current = min(chunk, draws - offset)
        indices = rng.integers(0, n, size=(current, n))
        means[offset : offset + current] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_results(output_root: Path) -> None:
    metric_paths = sorted((output_root / "runs").glob("*/metrics.json"))
    if not metric_paths:
        return
    rows = []
    for path in metric_paths:
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    run_frame = pd.DataFrame(rows)
    run_frame.to_csv(output_root / "all_runs.csv", index=False)

    group_cols = ["dataset", "architecture", "capacity", "condition"]
    metric_cols = [
        "mse",
        "mae",
        "none_mse",
        "event_mse",
        "separation_mse",
        "predicted_separation_energy",
        "ambiguity_floor",
    ]
    summary = run_frame.groupby(group_cols, as_index=False)[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(item).strip("_") for item in summary.columns.to_flat_index()]
    summary.to_csv(output_root / "summary_by_condition.csv", index=False)

    delta_rows: list[dict[str, object]] = []
    keys = ["dataset", "architecture", "capacity", "seed"]
    history_runs = run_frame[run_frame["condition"] == "info0"]
    open_runs = run_frame[run_frame["condition"] == "info100"]
    for _, left in history_runs.iterrows():
        matching = open_runs.copy()
        for key in keys:
            matching = matching[matching[key] == left[key]]
        if len(matching) != 1:
            continue
        right = matching.iloc[0]
        left_pairs = pd.read_csv(output_root / "runs" / left["run_id"] / "pair_losses.csv")
        right_pairs = pd.read_csv(output_root / "runs" / right["run_id"] / "pair_losses.csv")
        merged = left_pairs.merge(right_pairs, on="base_start", suffixes=("_history", "_open"))
        paired_delta = merged["pair_mse_history"].to_numpy() - merged["pair_mse_open"].to_numpy()
        rng = np.random.default_rng(stable_uint64("bootstrap", *[left[key] for key in keys]))
        ci_low, ci_high = bootstrap_ci(paired_delta, rng)
        delta_rows.append(
            {
                **{key: left[key] for key in keys},
                "n_pairs": len(merged),
                "history_mse": float(merged["pair_mse_history"].mean()),
                "open_mse": float(merged["pair_mse_open"].mean()),
                "delta_open": float(paired_delta.mean()),
                "relative_reduction": float(paired_delta.mean() / merged["pair_mse_history"].mean()),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
            }
        )
    delta_frame = pd.DataFrame(delta_rows)
    delta_frame.to_csv(output_root / "delta_open_paired_bootstrap.csv", index=False)

    if not delta_frame.empty:
        across_seed = (
            delta_frame.groupby(["dataset", "architecture", "capacity"], as_index=False)
            .agg(
                seeds=("seed", "nunique"),
                history_mse=("history_mse", "mean"),
                open_mse=("open_mse", "mean"),
                delta_open_mean=("delta_open", "mean"),
                delta_open_std=("delta_open", "std"),
                relative_reduction_mean=("relative_reduction", "mean"),
                all_ci_exclude_zero=("delta_ci_low", lambda series: bool((series > 0).all())),
            )
        )
        across_seed.to_csv(output_root / "delta_open_across_seeds.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_FILES), default=sorted(DATASET_FILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-train-windows", type=int, default=12000)
    parser.add_argument("--max-val-windows", type=int, default=2000)
    parser.add_argument("--max-test-windows", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.datasets = [args.datasets[0]]
        args.seeds = [args.seeds[0]]
        args.epochs = min(args.epochs, 2)
        args.patience = 2
        args.max_train_windows = min(args.max_train_windows, 512)
        args.max_val_windows = min(args.max_val_windows, 128)
        args.max_test_windows = min(args.max_test_windows, 128)
        args.workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    bundles = {name: load_bundle(args.data_root, name) for name in args.datasets}
    audit_protocol(bundles, args, args.output_dir)
    specs = experiment_specs(args.datasets, args.seeds)
    with (args.output_dir / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **vars(args),
                "data_root": str(args.data_root),
                "output_dir": str(args.output_dir),
                "device": str(device),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "run_count": len(specs),
            },
            handle,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    progress_path = args.output_dir / "progress.jsonl"
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        started = time.time()
        print(f"[{index}/{total}] START {spec.run_id}", flush=True)
        result = train_one(spec, bundles[spec.dataset], args, args.output_dir, device)
        event = {
            "index": index,
            "total": total,
            "run_id": spec.run_id,
            "status": result["status"],
            "mse": result["mse"],
            "duration_seconds": time.time() - started,
            "timestamp": time.time(),
        }
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(
            f"[{index}/{total}] DONE  {spec.run_id} mse={result['mse']:.6f} "
            f"seconds={event['duration_seconds']:.1f}",
            flush=True,
        )
    aggregate_results(args.output_dir)
    print(f"COMPLETED {total} runs", flush=True)


if __name__ == "__main__":
    main()
