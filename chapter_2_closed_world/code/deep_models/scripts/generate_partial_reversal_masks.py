#!/usr/bin/env python3
"""Generate exact, deterministic half-channel masks shared by all reversal cases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", "channel"),
    "ETTh2": ("ETT-small/ETTh2.csv", "channel"),
    "ETTm1": ("ETT-small/ETTm1.csv", "channel"),
    "ETTm2": ("ETT-small/ETTm2.csv", "channel"),
    "electricity": ("electricity/electricity.csv", "channel"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", "channel"),
    "weather": ("weather/weather.csv", "channel"),
    "illness": ("illness/national_illness.csv", "channel"),
}


def dataset_seed(base_seed: int, dataset: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{dataset}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def choose_half(identifiers: list[str], seed: int) -> np.ndarray:
    count = len(identifiers) // 2
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(identifiers), size=count, replace=False))
    mask = np.zeros(len(identifiers), dtype=bool)
    mask[indices] = True
    return mask


def write_manifest(output_dir: Path, dataset: str, identifiers: list[str], kind: str, base_seed: int) -> None:
    seed = dataset_seed(base_seed, dataset)
    mask = choose_half(identifiers, seed)
    selected_indices = np.flatnonzero(mask).astype(int).tolist()
    selected_ids = [identifiers[index] for index in selected_indices]
    payload = {
        "dataset": dataset,
        "population_type": kind,
        "base_seed": base_seed,
        "dataset_seed": seed,
        "selection_rule": "uniform_without_replacement_floor_half",
        "population_size": len(identifiers),
        "selected_count": len(selected_ids),
        "selected_indices": selected_indices,
        "selected_ids": selected_ids,
    }
    with (output_dir / f"{dataset}.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    with (output_dir / f"{dataset}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "identifier", "selected_for_reversal"])
        writer.writeheader()
        for index, identifier in enumerate(identifiers):
            writer.writerow({
                "index": index,
                "identifier": identifier,
                "selected_for_reversal": bool(mask[index]),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for dataset, (relative_path, kind) in DATASETS.items():
        columns = pd.read_csv(args.data_root / relative_path, nrows=0).columns.tolist()
        identifiers = [str(column) for column in columns if str(column) != "date"]
        if dataset in {"electricity", "exchange_rate", "weather", "illness"}:
            target = "OT"
            identifiers = [identifier for identifier in identifiers if identifier != target] + [target]
        write_manifest(args.output_dir, dataset, identifiers, kind, args.seed)

    m4_info = pd.read_csv(args.data_root / "m4" / "M4-info.csv")
    write_manifest(args.output_dir, "m4", m4_info["M4id"].astype(str).tolist(), "series", args.seed)


if __name__ == "__main__":
    main()
