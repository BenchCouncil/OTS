#!/usr/bin/env python3
"""Audit and consolidate the long-term partial-channel reversal experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = ["DLinear", "iTransformer", "PatchTST", "TimesNet", "GRU"]
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "exchange_rate", "weather", "illness"]
CASES = ["NN", "RN", "NR", "RR"]
SEQ_LENS = [96, 336]
PRED_LENS = [96, 336]


def expected_cells():
    for model in MODELS:
        for dataset in DATASETS:
            for case in CASES:
                for seq_len in SEQ_LENS:
                    for pred_len in PRED_LENS:
                        if dataset == "illness" and pred_len == 336:
                            continue
                        yield model, dataset, case, seq_len, pred_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    required_columns = {
        "dataset", "model", "case", "seq_len", "pred_len", "channel_index", "channel_name",
        "selected_for_reversal", "reversed_train", "reversed_eval", "mse_normalized",
        "mae_normalized", "value_count",
    }
    masks = {}
    for dataset in DATASETS:
        payload = json.loads((args.root / "masks" / f"{dataset}.json").read_text(encoding="utf-8"))
        masks[dataset] = payload

    missing = []
    invalid = []
    frames = []
    index_rows = []
    for model, dataset, case, seq_len, pred_len in expected_cells():
        result_dir = args.root / "long_term" / model / dataset / case / f"seq{seq_len}_pred{pred_len}"
        metric_path = result_dir / "channel_metrics.csv"
        metadata_path = result_dir / "metadata.json"
        if not metric_path.exists() or not metadata_path.exists():
            missing.append(str(metric_path.relative_to(args.root)))
            continue
        frame = pd.read_csv(metric_path)
        problems = []
        if not required_columns.issubset(frame.columns):
            problems.append("missing_columns")
        manifest = masks[dataset]
        if len(frame) != manifest["population_size"]:
            problems.append(f"row_count={len(frame)}")
        selected_names = set(frame.loc[frame.selected_for_reversal.astype(bool), "channel_name"].astype(str))
        if selected_names != set(map(str, manifest["selected_ids"])):
            problems.append("selected_channels_do_not_match_mask")
        selected = frame.selected_for_reversal.astype(bool).to_numpy()
        if not np.array_equal(frame.reversed_train.astype(bool).to_numpy(), selected & (case[0] == "R")):
            problems.append("reversed_train_flags")
        if not np.array_equal(frame.reversed_eval.astype(bool).to_numpy(), selected & (case[1] == "R")):
            problems.append("reversed_eval_flags")
        metrics = frame[["mse_normalized", "mae_normalized"]].to_numpy(dtype=np.float64)
        if not np.isfinite(metrics).all():
            problems.append("non_finite_metrics")
        if problems:
            invalid.append({"path": str(metric_path.relative_to(args.root)), "problems": problems})
        frame.insert(0, "result_relative_path", str(metric_path.relative_to(args.root)))
        frames.append(frame)
        index_rows.append({
            "model": model,
            "dataset": dataset,
            "case": case,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "channel_count": len(frame),
            "selected_count": int(selected.sum()),
            "mean_mse_normalized": float(frame.mse_normalized.mean()),
            "mean_mae_normalized": float(frame.mae_normalized.mean()),
            "result_relative_path": str(metric_path.relative_to(args.root)),
        })

    expected_count = sum(1 for _ in expected_cells())
    summary = {
        "status": "complete" if not missing and not invalid else "incomplete_or_invalid",
        "expected_result_files": expected_count,
        "valid_result_files": len(index_rows) - len(invalid),
        "present_result_files": len(index_rows),
        "missing_count": len(missing),
        "invalid_count": len(invalid),
        "missing": missing,
        "invalid": invalid,
        "models": MODELS,
        "datasets": DATASETS,
        "cases": CASES,
        "sequence_lengths": SEQ_LENS,
        "prediction_lengths": PRED_LENS,
        "illness_note": "pred_len=336 is excluded, matching the existing experiment grid",
        "metric_scale": "TSLib normalized scale",
    }
    (args.root / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(index_rows).to_csv(args.root / "result_index.csv", index=False)
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            args.root / "all_channel_metrics.csv.gz", index=False, compression="gzip")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if (missing or invalid) and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
