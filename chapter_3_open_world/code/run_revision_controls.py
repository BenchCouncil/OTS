#!/usr/bin/env python3
"""Revision experiments requested by the first independent reviewer.

The script deliberately separates four questions that the original experiment
mixed together:

1. Can a matched-reversal risk statistic distinguish a reversible control from
   an irreversible control?
2. Does forecast-time information help when it is a noisy scheduled-plan proxy,
   rather than the exact parameters used to generate the future response?
3. Can a history-only probabilistic residual forecast represent branch
   uncertainty even though it cannot identify the realized branch?
4. Do naturally observed multivariate channels available by the forecast origin
   help beyond the target's own history on the selected datasets?

All raw time-series splits and normalization statistics are chronological and
estimated from the training segment only. Outputs are CSV/JSON files; no result
is hard-coded into the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import t as student_t
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import PolynomialFeatures


DATASET_FILES = {
    "ETTh1": Path("shared/datasets/ETT-small/ETTh1.csv"),
    "ETTh2": Path("shared/datasets/ETT-small/ETTh2.csv"),
    "ETTm1": Path("shared/datasets/ETT-small/ETTm1.csv"),
    "ETTm2": Path("shared/datasets/ETT-small/ETTm2.csv"),
    "weather": Path("shared/datasets/weather/weather.csv"),
    "exchange_rate": Path("shared/datasets/exchange_rate/exchange_rate.csv"),
    "electricity": Path("shared/datasets/electricity/electricity.csv"),
}


@dataclass
class Bundle:
    name: str
    target: np.ndarray
    multivariate: np.ndarray
    columns: list[str]
    train_end: int
    val_end: int
    test_end: int


@dataclass
class StableRidge:
    """Small dense multi-output ridge model with deterministic accumulation.

    NumPy linked against Apple Accelerate emitted spurious overflow warnings for
    otherwise finite BLAS matrix products in this environment.  Explicit
    ``einsum(..., optimize=False)`` keeps the numerical path warning-free and is
    inexpensive for the sub-250-dimensional feature matrices used here.
    """

    coefficient: np.ndarray
    intercept: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.einsum("ni,ij->nj", x, self.coefficient, optimize=False) + self.intercept


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def load_bundle(project_root: Path, name: str) -> Bundle:
    frame = pd.read_csv(project_root / DATASET_FILES[name])
    numeric = frame.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    if numeric.isna().any().any():
        raise ValueError(f"{name} contains missing numeric values")
    values = numeric.to_numpy(dtype=np.float64)
    if name in {"ETTm1", "ETTm2"}:
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = train_end + 8 * 30 * 24 * 4
    elif name in {"ETTh1", "ETTh2"}:
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = train_end + 8 * 30 * 24
    else:
        train_end = int(len(values) * 0.7)
        test_size = int(len(values) * 0.2)
        val_end = len(values) - test_size
        test_end = len(values)
    if test_end > len(values):
        raise ValueError(f"split for {name} exceeds series length")
    mean = values[:train_end].mean(axis=0)
    std = values[:train_end].std(axis=0)
    std[std < 1e-8] = 1.0
    normalized = (values - mean) / std
    return Bundle(
        name=name,
        target=normalized[:, -1].astype(np.float64),
        multivariate=normalized.astype(np.float64),
        columns=numeric.columns.tolist(),
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
    )


def split_starts(bundle: Bundle, split: str, seq_len: int, pred_len: int, cap: int) -> np.ndarray:
    if split == "train":
        first, last = 0, bundle.train_end - seq_len - pred_len
    elif split == "val":
        first, last = bundle.train_end - seq_len, bundle.val_end - seq_len - pred_len
    elif split == "test":
        first, last = bundle.val_end - seq_len, bundle.test_end - seq_len - pred_len
    else:
        raise ValueError(split)
    starts = np.arange(first, last + 1, dtype=np.int64)
    if cap > 0 and len(starts) > cap:
        starts = starts[np.linspace(0, len(starts) - 1, cap, dtype=np.int64)]
    return np.unique(starts)


def take_windows(values: np.ndarray, starts: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    indices = starts[:, None] + offset + np.arange(length)[None, :]
    return values[indices]


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0),
) -> tuple[StableRidge, float, float]:
    best_model = None
    best_alpha = math.nan
    best_mse = math.inf
    for alpha in alphas:
        x_mean = x_train.mean(axis=0)
        y_mean = y_train.mean(axis=0)
        centered_x = x_train - x_mean
        centered_y = y_train - y_mean
        gram = np.einsum("ni,nj->ij", centered_x, centered_x, optimize=False)
        cross = np.einsum("ni,nk->ik", centered_x, centered_y, optimize=False)
        gram.flat[:: len(gram) + 1] += alpha
        coefficient = np.linalg.solve(gram, cross)
        intercept = y_mean - np.einsum("i,ij->j", x_mean, coefficient, optimize=False)
        model = StableRidge(coefficient=coefficient, intercept=intercept)
        mse = float(np.mean((model.predict(x_val) - y_val) ** 2))
        if not math.isfinite(mse):
            continue
        if mse < best_mse:
            best_model = model
            best_alpha = alpha
            best_mse = mse
    assert best_model is not None
    return best_model, best_alpha, best_mse


def real_mirror_controls(bundle: Bundle, seq_len: int, pred_len: int) -> list[dict[str, object]]:
    if seq_len != pred_len:
        raise ValueError("mirror controls currently require seq_len == pred_len")
    starts = {
        split: split_starts(bundle, split, seq_len, pred_len, cap)
        for split, cap in (("train", 12000), ("val", 2000), ("test", 4000))
    }
    blocks = {
        split: take_windows(bundle.target, split_starts_array, seq_len + pred_len)
        for split, split_starts_array in starts.items()
    }

    def transform(block: np.ndarray, condition: str) -> tuple[np.ndarray, np.ndarray]:
        history = block[:, :seq_len]
        future = block[:, seq_len:]
        if condition == "forward":
            return history, future
        if condition == "same_block_mirror":
            mirrored = block[:, ::-1]
            return mirrored[:, :seq_len], mirrored[:, seq_len:]
        if condition == "block_swap":
            return future, history
        if condition == "input_only_reverse":
            return history[:, ::-1], future
        if condition == "target_only_reverse":
            return history, future[:, ::-1]
        raise ValueError(condition)

    rows = []
    for condition in (
        "forward",
        "same_block_mirror",
        "block_swap",
        "input_only_reverse",
        "target_only_reverse",
    ):
        x_train, y_train = transform(blocks["train"], condition)
        x_val, y_val = transform(blocks["val"], condition)
        x_test, y_test = transform(blocks["test"], condition)
        model, alpha, val_mse = fit_ridge(x_train, y_train, x_val, y_val)
        test_mse = float(np.mean((model.predict(x_test) - y_test) ** 2))
        rows.append(
            {
                "dataset": bundle.name,
                "condition": condition,
                "model": "ridge",
                "alpha": alpha,
                "val_mse": val_mse,
                "test_mse": test_mse,
                "n_test_windows": len(x_test),
            }
        )
    forward = next(row["test_mse"] for row in rows if row["condition"] == "forward")
    for row in rows:
        row["risk_ratio_vs_forward"] = float(row["test_mse"] / forward)
    return rows


def simulate_ma(process: str, seed: int, n: int = 30000) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if process == "reversible_gaussian_ma":
        innovations = rng.normal(size=n + 1000)
    elif process == "irreversible_nongaussian_ma":
        innovations = rng.exponential(size=n + 1000) - 1.0
    else:
        raise ValueError(process)
    series = innovations[1:] + 0.9 * innovations[:-1]
    return series[999:].astype(np.float64)


def synthetic_direction_calibration(seeds: int) -> list[dict[str, object]]:
    rows = []
    history_length = 4
    for process in ("reversible_gaussian_ma", "irreversible_nongaussian_ma"):
        for seed in range(seeds):
            series = simulate_ma(process, seed)
            windows = np.lib.stride_tricks.sliding_window_view(series, history_length + 1)
            train = windows[:20000]
            test = windows[22000:]
            risks = {}
            for direction in ("forward", "reverse"):
                train_view = train if direction == "forward" else train[:, ::-1]
                test_view = test if direction == "forward" else test[:, ::-1]
                model = ExtraTreesRegressor(
                    n_estimators=100,
                    min_samples_leaf=8,
                    n_jobs=-1,
                    random_state=stable_seed(process, seed, direction),
                )
                model.fit(train_view[:, :history_length], train_view[:, history_length])
                prediction = model.predict(test_view[:, :history_length])
                risks[direction] = float(np.mean((prediction - test_view[:, history_length]) ** 2))
            rows.append(
                {
                    "process": process,
                    "seed": seed,
                    "forward_mse": risks["forward"],
                    "reverse_mse": risks["reverse"],
                    "reverse_forward_ratio": risks["reverse"] / risks["forward"],
                    "abs_log_ratio": abs(math.log(risks["reverse"] / risks["forward"])),
                }
            )
    return rows


def context_features(multivariate: np.ndarray, starts: np.ndarray, seq_len: int) -> np.ndarray:
    history = take_windows(multivariate, starts, seq_len)
    target_history = history[:, :, -1]
    external = history[:, :, :-1]
    last = external[:, -1, :]
    mean_12 = external[:, -min(12, seq_len):, :].mean(axis=1)
    mean_48 = external[:, -min(48, seq_len):, :].mean(axis=1)
    std_48 = external[:, -min(48, seq_len):, :].std(axis=1)
    slope_48 = external[:, -1, :] - external[:, -min(48, seq_len), :]
    return np.concatenate([target_history, last, mean_12, mean_48, std_48, slope_48], axis=1)


def natural_context_experiment(bundle: Bundle, seq_len: int, pred_len: int) -> list[dict[str, object]]:
    starts = {
        split: split_starts(bundle, split, seq_len, pred_len, cap)
        for split, cap in (("train", 12000), ("val", 2000), ("test", 4000))
    }
    targets = {
        split: take_windows(bundle.target, split_starts_array, pred_len, offset=seq_len)
        for split, split_starts_array in starts.items()
    }
    features = {
        "target_history": {
            split: take_windows(bundle.target, split_starts_array, seq_len)
            for split, split_starts_array in starts.items()
        },
        "observed_context": {
            split: context_features(bundle.multivariate, split_starts_array, seq_len)
            for split, split_starts_array in starts.items()
        },
    }
    rows = []
    for condition, by_split in features.items():
        model, alpha, val_mse = fit_ridge(
            by_split["train"], targets["train"], by_split["val"], targets["val"]
        )
        pred = model.predict(by_split["test"])
        per_window = ((pred - targets["test"]) ** 2).mean(axis=1)
        rows.append(
            {
                "dataset": bundle.name,
                "condition": condition,
                "alpha": alpha,
                "val_mse": val_mse,
                "test_mse": float(per_window.mean()),
                "n_test_windows": len(per_window),
                "per_window_mse": per_window,
                "starts": starts["test"],
            }
        )
    history_mse = next(row["test_mse"] for row in rows if row["condition"] == "target_history")
    for row in rows:
        row["relative_reduction_vs_target_history"] = 1.0 - row["test_mse"] / history_mse
    return rows


def make_plan_worlds(dataset: str, starts: np.ndarray, pred_len: int, seed: int) -> dict[str, np.ndarray]:
    n = len(starts)
    event_type = np.empty(n, dtype=np.int64)
    sign = np.empty(n, dtype=np.float64)
    strength_class = np.empty(n, dtype=np.int64)
    timing_class = np.empty(n, dtype=np.int64)
    actual_type = np.empty(n, dtype=np.int64)
    actual_amplitude = np.empty(n, dtype=np.float64)
    actual_onset = np.empty(n, dtype=np.int64)
    actual_duration = np.empty(n, dtype=np.int64)
    executed = np.empty(n, dtype=np.float64)
    effects = np.empty((n, pred_len), dtype=np.float64)
    time = np.arange(pred_len, dtype=np.float64)
    strength_centers = np.array([0.55, 0.95, 1.35])
    timing_centers = np.array([max(3, pred_len // 8), pred_len // 3, pred_len // 2])
    for index, start in enumerate(starts):
        rng = np.random.default_rng(stable_seed("plan-world", dataset, int(start), seed))
        q = int(rng.integers(0, 3))
        sgn = -1.0 if rng.random() < 0.5 else 1.0
        strength = int(rng.integers(0, 3))
        timing = int(rng.integers(0, 3))
        did_execute = float(rng.random() < 0.85)
        realized_type = q if rng.random() >= 0.10 else int(rng.integers(0, 3))
        magnitude = strength_centers[strength] * float(rng.lognormal(mean=0.0, sigma=0.22))
        realized_sign = sgn if rng.random() >= 0.08 else -sgn
        amplitude = did_execute * realized_sign * magnitude
        onset = int(np.clip(timing_centers[timing] + rng.normal(0.0, pred_len / 16), 1, pred_len - 3))
        duration = int(np.clip(rng.normal(pred_len / 3, pred_len / 10), 3, pred_len - onset))
        active = np.maximum(time - onset, 0.0)
        if realized_type == 0:
            kernel = (time >= onset).astype(np.float64)
        elif realized_type == 1:
            decay = max(2.0, duration / 3.0)
            kernel = np.where(time >= onset, np.exp(-active / decay), 0.0)
        else:
            denominator = max(1.0, pred_len - 1 - onset)
            kernel = np.where(time >= onset, active / denominator, 0.0)
        response_noise = rng.normal(0.0, 0.04, size=pred_len) * (time >= onset)
        effect = amplitude * kernel + did_execute * response_noise
        event_type[index] = q
        sign[index] = sgn
        strength_class[index] = strength
        timing_class[index] = timing
        actual_type[index] = realized_type
        actual_amplitude[index] = amplitude
        actual_onset[index] = onset
        actual_duration[index] = duration
        executed[index] = did_execute
        effects[index] = effect
    return {
        "event_type": event_type,
        "sign": sign,
        "strength_class": strength_class,
        "timing_class": timing_class,
        "actual_type": actual_type,
        "actual_amplitude": actual_amplitude,
        "actual_onset": actual_onset,
        "actual_duration": actual_duration,
        "executed": executed,
        "effect": effects,
    }


def plan_features(world: dict[str, np.ndarray], mode: str, seed_key: tuple[object, ...]) -> np.ndarray:
    n = len(world["event_type"])
    # Each base history has a no-plan branch and a scheduled-plan branch.
    z = np.zeros((2 * n, 10), dtype=np.float64)
    z[:, 0] = 1.0  # package is available; delayed samples overwrite this
    event_rows = np.arange(n) * 2 + 1
    z[event_rows, 1] = 1.0
    z[event_rows, 2 + world["event_type"]] = 1.0
    z[event_rows, 5] = world["sign"]
    z[event_rows, 6] = world["strength_class"] / 2.0
    z[event_rows, 7] = world["timing_class"] / 2.0
    z[event_rows, 8] = 0.8  # source confidence, not outcome confidence
    z[event_rows, 9] = 0.0  # reserved missing-field marker
    rng = np.random.default_rng(stable_seed(*seed_key, mode))
    if mode == "clean":
        return z
    if mode == "noisy":
        noisy = z.copy()
        scheduled = noisy[:, 1] > 0.5
        noisy[scheduled, 6:8] += rng.normal(0.0, 0.18, size=(scheduled.sum(), 2))
        noisy[scheduled, 6:8] = np.clip(noisy[scheduled, 6:8], 0.0, 1.0)
        drop = scheduled & (rng.random(len(noisy)) < 0.20)
        noisy[drop, 2:5] = 0.0
        noisy[drop, 9] = 1.0
        corrupt = scheduled & (rng.random(len(noisy)) < 0.10)
        noisy[corrupt, 5] *= -1.0
        noisy[scheduled, 8] = 0.6
        return noisy
    if mode == "delayed50":
        delayed = z.copy()
        hide = (delayed[:, 1] > 0.5) & (rng.random(len(delayed)) < 0.50)
        delayed[hide] = 0.0
        delayed[hide, 9] = 1.0
        return delayed
    if mode == "shuffled":
        return z[rng.permutation(len(z))]
    if mode == "misleading":
        wrong = z.copy()
        event = wrong[:, 1] > 0.5
        wrong[event, 5] *= -1.0
        wrong[event, 8] = 0.8
        return wrong
    if mode == "history":
        history = np.zeros_like(z)
        history[:, 0] = 1.0
        return history
    raise ValueError(mode)


def pair_targets(base_future: np.ndarray, effect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paired_base = np.repeat(base_future, 2, axis=0)
    residual = np.zeros_like(paired_base)
    residual[1::2] = effect
    return paired_base + residual, residual


def fit_residual_model(
    z_train: np.ndarray,
    r_train: np.ndarray,
    z_val: np.ndarray,
    r_val: np.ndarray,
) -> tuple[PolynomialFeatures, StableRidge, float]:
    poly = PolynomialFeatures(degree=2, include_bias=False)
    xp_train = poly.fit_transform(z_train)
    xp_val = poly.transform(z_val)
    model, alpha, _ = fit_ridge(xp_train, r_train, xp_val, r_val)
    return poly, model, alpha


def energy_score(y: np.ndarray, samples: np.ndarray, pair_term: float) -> float:
    first = np.linalg.norm(samples - y[None, :], axis=1).mean()
    return float(first - pair_term)


def probabilistic_branch_scores(
    train_world: dict[str, np.ndarray],
    test_world: dict[str, np.ndarray],
    max_test: int = 1200,
    prototypes: int = 160,
) -> dict[str, float]:
    train_effect = train_world["effect"]
    zero = np.zeros((len(train_effect), train_effect.shape[1]), dtype=np.float64)
    history_pool = np.concatenate([zero, train_effect], axis=0)
    rng = np.random.default_rng(92831)

    def sample_pool(pool: np.ndarray, count: int) -> np.ndarray:
        if len(pool) <= count:
            return pool
        return pool[rng.choice(len(pool), size=count, replace=False)]

    history_samples = sample_pool(history_pool, prototypes)
    history_pair = 0.5 * float(pdist(history_samples, metric="euclidean").mean())
    group_samples: dict[tuple[int, int, int], np.ndarray] = {}
    group_pair: dict[tuple[int, int, int], float] = {}
    for q in range(3):
        for sign in (-1, 1):
            for strength in range(3):
                mask = (
                    (train_world["event_type"] == q)
                    & (train_world["sign"] == sign)
                    & (train_world["strength_class"] == strength)
                )
                pool = train_effect[mask]
                if len(pool) < 4:
                    pool = train_effect[(train_world["event_type"] == q) & (train_world["sign"] == sign)]
                sampled = sample_pool(pool, prototypes)
                key = (q, sign, strength)
                group_samples[key] = sampled
                group_pair[key] = 0.5 * float(pdist(sampled, metric="euclidean").mean()) if len(sampled) > 1 else 0.0

    indices = np.linspace(0, len(test_world["effect"]) - 1, min(max_test, len(test_world["effect"])), dtype=int)
    history_scores = []
    plan_scores = []
    zero_samples = np.zeros((1, train_effect.shape[1]), dtype=np.float64)
    for index in indices:
        for branch in (0, 1):
            outcome = np.zeros(train_effect.shape[1]) if branch == 0 else test_world["effect"][index]
            history_scores.append(energy_score(outcome, history_samples, history_pair))
            if branch == 0:
                plan_scores.append(energy_score(outcome, zero_samples, 0.0))
            else:
                key = (
                    int(test_world["event_type"][index]),
                    int(test_world["sign"][index]),
                    int(test_world["strength_class"][index]),
                )
                plan_scores.append(energy_score(outcome, group_samples[key], group_pair[key]))
    scale = math.sqrt(train_effect.shape[1])
    return {
        "history_mixture_energy_score": float(np.mean(history_scores) / scale),
        "plan_conditional_energy_score": float(np.mean(plan_scores) / scale),
        "relative_energy_score_reduction": float(1.0 - np.mean(plan_scores) / np.mean(history_scores)),
        "evaluated_branch_outcomes": len(history_scores),
    }


def planned_information_experiment(
    bundle: Bundle,
    seq_len: int,
    pred_len: int,
    seeds: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, list[dict[str, object]]]:
    starts = {
        split: split_starts(bundle, split, seq_len, pred_len, cap)
        for split, cap in (("train", 12000), ("val", 2000), ("test", 4000))
    }
    history = {
        split: take_windows(bundle.target, split_starts_array, seq_len)
        for split, split_starts_array in starts.items()
    }
    base_future = {
        split: take_windows(bundle.target, split_starts_array, pred_len, offset=seq_len)
        for split, split_starts_array in starts.items()
    }
    base_model, base_alpha, _ = fit_ridge(
        history["train"], base_future["train"], history["val"], base_future["val"]
    )
    base_prediction = {split: base_model.predict(history[split]) for split in starts}
    rows: list[dict[str, object]] = []
    per_window_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    modes = ("history", "clean", "noisy", "delayed50", "shuffled")
    for seed in range(seeds):
        worlds = {
            split: make_plan_worlds(bundle.name, starts[split], pred_len, seed)
            for split in starts
        }
        paired_targets = {}
        paired_residuals = {}
        for split in starts:
            paired_targets[split], paired_residuals[split] = pair_targets(
                base_future[split], worlds[split]["effect"]
            )
        probability = probabilistic_branch_scores(worlds["train"], worlds["test"])
        probability_rows.append({"dataset": bundle.name, "seed": seed, **probability})
        clean_model = None
        clean_poly = None
        for mode in modes:
            z = {
                split: plan_features(worlds[split], mode, (bundle.name, split, seed))
                for split in starts
            }
            poly, residual_model, residual_alpha = fit_residual_model(
                z["train"], paired_residuals["train"], z["val"], paired_residuals["val"]
            )
            if mode == "clean":
                clean_poly, clean_model = poly, residual_model
            residual_prediction = residual_model.predict(poly.transform(z["test"]))
            base_pair_prediction = np.repeat(base_prediction["test"], 2, axis=0)
            prediction = base_pair_prediction + residual_prediction
            squared = (prediction - paired_targets["test"]) ** 2
            residual_squared = (residual_prediction - paired_residuals["test"]) ** 2
            pair_mse = squared.reshape(-1, 2, pred_len).mean(axis=(1, 2))
            pred_pair = prediction.reshape(-1, 2, pred_len)
            true_pair = paired_targets["test"].reshape(-1, 2, pred_len)
            separation = ((pred_pair[:, 1] - pred_pair[:, 0]) - (true_pair[:, 1] - true_pair[:, 0])) ** 2
            rows.append(
                {
                    "dataset": bundle.name,
                    "seed": seed,
                    "condition": mode,
                    "base_alpha": base_alpha,
                    "residual_alpha": residual_alpha,
                    "mse": float(squared.mean()),
                    "residual_mse": float(residual_squared.mean()),
                    "none_mse": float(squared[0::2].mean()),
                    "event_mse": float(squared[1::2].mean()),
                    "separation_mse": float(separation.mean()),
                    "n_test_pairs": len(pair_mse),
                }
            )
            for start, value in zip(starts["test"], pair_mse):
                per_window_rows.append(
                    {
                        "dataset": bundle.name,
                        "seed": seed,
                        "condition": mode,
                        "base_start": int(start),
                        "pair_mse": float(value),
                    }
                )

        # Manipulated information is evaluated without retraining the clean model.
        assert clean_model is not None and clean_poly is not None
        z_wrong = plan_features(worlds["test"], "misleading", (bundle.name, "test", seed))
        residual_prediction = clean_model.predict(clean_poly.transform(z_wrong))
        base_pair_prediction = np.repeat(base_prediction["test"], 2, axis=0)
        prediction = base_pair_prediction + residual_prediction
        squared = (prediction - paired_targets["test"]) ** 2
        pair_mse = squared.reshape(-1, 2, pred_len).mean(axis=(1, 2))
        pred_pair = prediction.reshape(-1, 2, pred_len)
        true_pair = paired_targets["test"].reshape(-1, 2, pred_len)
        separation = ((pred_pair[:, 1] - pred_pair[:, 0]) - (true_pair[:, 1] - true_pair[:, 0])) ** 2
        rows.append(
            {
                "dataset": bundle.name,
                "seed": seed,
                "condition": "misleading_test",
                "base_alpha": base_alpha,
                "residual_alpha": math.nan,
                "mse": float(squared.mean()),
                "residual_mse": float(((residual_prediction - paired_residuals["test"]) ** 2).mean()),
                "none_mse": float(squared[0::2].mean()),
                "event_mse": float(squared[1::2].mean()),
                "separation_mse": float(separation.mean()),
                "n_test_pairs": len(pair_mse),
            }
        )
        for start, value in zip(starts["test"], pair_mse):
            per_window_rows.append(
                {
                    "dataset": bundle.name,
                    "seed": seed,
                    "condition": "misleading_test",
                    "base_start": int(start),
                    "pair_mse": float(value),
                }
            )

        # Exact latent response is retained only as an explicitly labeled upper bound.
        oracle_prediction = np.repeat(base_prediction["test"], 2, axis=0) + paired_residuals["test"]
        oracle_squared = (oracle_prediction - paired_targets["test"]) ** 2
        oracle_pair_mse = oracle_squared.reshape(-1, 2, pred_len).mean(axis=(1, 2))
        rows.append(
            {
                "dataset": bundle.name,
                "seed": seed,
                "condition": "latent_oracle_upper_bound",
                "base_alpha": base_alpha,
                "residual_alpha": math.nan,
                "mse": float(oracle_squared.mean()),
                "residual_mse": 0.0,
                "none_mse": float(oracle_squared[0::2].mean()),
                "event_mse": float(oracle_squared[1::2].mean()),
                "separation_mse": 0.0,
                "n_test_pairs": len(oracle_pair_mse),
            }
        )
        for start, value in zip(starts["test"], oracle_pair_mse):
            per_window_rows.append(
                {
                    "dataset": bundle.name,
                    "seed": seed,
                    "condition": "latent_oracle_upper_bound",
                    "base_start": int(start),
                    "pair_mse": float(value),
                }
            )
    return rows, pd.DataFrame(per_window_rows), probability_rows


def crossed_seed_time_block_bootstrap(
    per_window: pd.DataFrame,
    draws: int,
    raw_block_lengths: tuple[int, ...] = (96, 192, 384),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seed_level_rows = []
    for dataset in sorted(per_window["dataset"].unique()):
        data = per_window[per_window["dataset"] == dataset]
        seeds = sorted(data["seed"].unique())
        seed_deltas = {}
        starts_reference = None
        for seed in seeds:
            history = data[(data["seed"] == seed) & (data["condition"] == "history")]
            planned = data[(data["seed"] == seed) & (data["condition"] == "clean")]
            merged = history.merge(
                planned,
                on=["dataset", "seed", "base_start"],
                suffixes=("_history", "_plan"),
                validate="one_to_one",
            ).sort_values("base_start")
            current_starts = merged["base_start"].to_numpy()
            if starts_reference is None:
                starts_reference = current_starts
            elif not np.array_equal(starts_reference, current_starts):
                raise ValueError(f"{dataset} does not share base_start across world seeds")
            seed_deltas[seed] = (merged["pair_mse_history"] - merged["pair_mse_plan"]).to_numpy()
        assert starts_reference is not None
        delta_matrix = np.stack([seed_deltas[seed] for seed in seeds], axis=0)
        seed_means = delta_matrix.mean(axis=1)
        seed_sem = float(seed_means.std(ddof=1) / math.sqrt(len(seed_means)))
        seed_critical = float(student_t.ppf(0.975, df=len(seed_means) - 1))
        seed_observed = float(seed_means.mean())
        seed_level_rows.append(
            {
                "dataset": dataset,
                "unit": "world_seed",
                "seeds": len(seeds),
                "delta_mse": seed_observed,
                "ci95_low": seed_observed - seed_critical * seed_sem,
                "ci95_high": seed_observed + seed_critical * seed_sem,
                "all_seed_deltas_positive": bool(np.all(seed_means > 0.0)),
            }
        )
        spacing = float(np.median(np.diff(starts_reference))) if len(starts_reference) > 1 else 1.0
        for raw_block in raw_block_lengths:
            block = max(1, int(math.ceil(raw_block / max(1.0, spacing))))
            rng = np.random.default_rng(stable_seed("crossed-bootstrap", dataset, raw_block))
            sampled_means = np.empty(draws, dtype=np.float64)
            for draw in range(draws):
                sampled_seed_indices = rng.integers(0, len(seeds), size=len(seeds))
                n_blocks = math.ceil(delta_matrix.shape[1] / block)
                maximum_start = max(1, delta_matrix.shape[1] - block + 1)
                block_starts = rng.integers(0, maximum_start, size=n_blocks)
                time_indices = (
                    block_starts[:, None] + np.arange(block)[None, :]
                ).reshape(-1)[: delta_matrix.shape[1]]
                # The same sampled time blocks are applied to every sampled seed.
                # This preserves the crossed design induced by shared histories,
                # base starts, and the fixed history predictor.
                sampled_means[draw] = float(
                    delta_matrix[sampled_seed_indices][:, time_indices].mean()
                )
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": "history_minus_clean_plan",
                    "resampling_design": "crossed_world_seed_by_shared_time_block",
                    "seeds": len(seeds),
                    "n_windows_per_seed": len(starts_reference),
                    "raw_block_length": raw_block,
                    "block_length_pairs": block,
                    "delta_mse": seed_observed,
                    "ci95_low": float(np.quantile(sampled_means, 0.025)),
                    "ci95_high": float(np.quantile(sampled_means, 0.975)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(seed_level_rows)


def serializable_context_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], pd.DataFrame]:
    summary = []
    window_rows = []
    for row in rows:
        clean = {key: value for key, value in row.items() if key not in {"per_window_mse", "starts"}}
        summary.append(clean)
        for start, mse in zip(row["starts"], row["per_window_mse"]):
            window_rows.append(
                {
                    "dataset": row["dataset"],
                    "condition": row["condition"],
                    "base_start": int(start),
                    "mse": float(mse),
                }
            )
    return summary, pd.DataFrame(window_rows)


def audit_outputs(
    planned: pd.DataFrame,
    probability: pd.DataFrame,
    direction: pd.DataFrame,
    context: pd.DataFrame,
    crossed_bootstrap: pd.DataFrame,
    seed_level_ci: pd.DataFrame,
    datasets: list[str],
    seeds: int,
) -> dict[str, object]:
    expected_conditions = {
        "history",
        "clean",
        "noisy",
        "delayed50",
        "shuffled",
        "misleading_test",
        "latent_oracle_upper_bound",
    }
    checks = {
        "planned_unique_rows": len(planned) == len(datasets) * seeds * len(expected_conditions),
        "planned_conditions_complete": set(planned["condition"].unique()) == expected_conditions,
        "probability_rows_complete": len(probability) == len(datasets) * seeds,
        "direction_rows_complete": len(direction) == 2 * seeds,
        "context_rows_complete": len(context) == len(datasets) * 2,
        "crossed_bootstrap_rows_complete": len(crossed_bootstrap) == len(datasets) * 3,
        "seed_level_ci_rows_complete": len(seed_level_ci) == len(datasets),
        "all_metrics_finite": bool(
            np.isfinite(planned[["mse", "residual_mse", "separation_mse"]].to_numpy()).all()
            and np.isfinite(probability.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(direction.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(context.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(crossed_bootstrap.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(seed_level_ci.select_dtypes(include=[np.number]).to_numpy()).all()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    release_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--project-root", type=Path, default=release_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=release_root / "chapter_3_open_world" / "results" / "revision_controls_new",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_FILES))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    direction_rows = synthetic_direction_calibration(args.seeds)
    real_mirror_rows: list[dict[str, object]] = []
    planned_rows: list[dict[str, object]] = []
    planned_windows = []
    probability_rows: list[dict[str, object]] = []
    context_rows_raw: list[dict[str, object]] = []
    for dataset in args.datasets:
        bundle = load_bundle(project_root, dataset)
        real_mirror_rows.extend(real_mirror_controls(bundle, args.seq_len, args.pred_len))
        current_planned, current_windows, current_probability = planned_information_experiment(
            bundle, args.seq_len, args.pred_len, args.seeds
        )
        planned_rows.extend(current_planned)
        planned_windows.append(current_windows)
        probability_rows.extend(current_probability)
        context_rows_raw.extend(natural_context_experiment(bundle, args.seq_len, args.pred_len))

    context_rows, context_windows = serializable_context_rows(context_rows_raw)
    direction = pd.DataFrame(direction_rows)
    real_mirror = pd.DataFrame(real_mirror_rows)
    planned = pd.DataFrame(planned_rows)
    per_window = pd.concat(planned_windows, ignore_index=True)
    probability = pd.DataFrame(probability_rows)
    context = pd.DataFrame(context_rows)
    crossed_bootstrap, seed_level_ci = crossed_seed_time_block_bootstrap(
        per_window, args.bootstrap_draws
    )

    direction.to_csv(output_dir / "synthetic_direction_calibration.csv", index=False)
    real_mirror.to_csv(output_dir / "real_same_block_controls.csv", index=False)
    planned.to_csv(output_dir / "planned_information_runs.csv", index=False)
    per_window.to_csv(output_dir / "planned_information_per_window.csv", index=False)
    probability.to_csv(output_dir / "probabilistic_branch_scores.csv", index=False)
    context.to_csv(output_dir / "natural_context_summary.csv", index=False)
    context_windows.to_csv(output_dir / "natural_context_per_window.csv", index=False)
    crossed_bootstrap.to_csv(output_dir / "planned_information_crossed_bootstrap.csv", index=False)
    # Retain the original path as a compatibility alias, now containing the
    # corrected crossed design rather than the superseded nested resampling.
    crossed_bootstrap.to_csv(output_dir / "planned_information_hierarchical_bootstrap.csv", index=False)
    seed_level_ci.to_csv(output_dir / "planned_information_seed_level_ci.csv", index=False)

    planned_summary = planned.groupby(["dataset", "condition"], as_index=False).agg(
        seeds=("seed", "nunique"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        residual_mse_mean=("residual_mse", "mean"),
        separation_mse_mean=("separation_mse", "mean"),
    )
    history = planned_summary[planned_summary["condition"] == "history"][["dataset", "mse_mean"]].rename(
        columns={"mse_mean": "history_mse"}
    )
    planned_summary = planned_summary.merge(history, on="dataset", validate="many_to_one")
    planned_summary["relative_reduction_vs_history"] = 1.0 - planned_summary["mse_mean"] / planned_summary["history_mse"]
    planned_summary.to_csv(output_dir / "planned_information_summary.csv", index=False)

    audit = audit_outputs(
        planned,
        probability,
        direction,
        context,
        crossed_bootstrap,
        seed_level_ci,
        args.datasets,
        args.seeds,
    )
    configuration = {
        "datasets": args.datasets,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "seeds": args.seeds,
        "bootstrap_draws": args.bootstrap_draws,
        "target": "OT (last numeric column after date)",
        "normalization": "per-channel training-split mean and standard deviation",
        "planned_information_semantics": {
            "observable": [
                "scheduled-plan flag",
                "plan type",
                "planned response direction",
                "coarse strength class",
                "coarse timing class",
                "source confidence and missingness",
            ],
            "latent": [
                "execution success",
                "realized response type",
                "realized sign",
                "exact amplitude",
                "exact onset",
                "exact duration",
                "response noise",
            ],
        },
        "audit": audit,
    }
    with (output_dir / "experiment_config_and_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(configuration, handle, ensure_ascii=False, indent=2)
    if not audit["passed"]:
        raise RuntimeError(f"revision experiment audit failed: {audit}")
    print(json.dumps({"output_dir": str(output_dir), **audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
