from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from dataloader import get_dataset_config, list_benchmark_names
from exp import M4OLSExperiment, OLSExperiment, run_grid

PROJECT_DIR = Path(__file__).resolve().parent
if PROJECT_DIR.parent.name == "code" and PROJECT_DIR.parent.parent.name == "chapter_2_closed_world":
    WORKSPACE_DIR = PROJECT_DIR.parents[2]
    DEFAULT_DATA_ROOT = WORKSPACE_DIR / "shared" / "datasets"
    DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "chapter_2_closed_world" / "results" / "new_runs"
    DEFAULT_MODEL_DIR = WORKSPACE_DIR / "chapter_2_closed_world" / "checkpoints"
else:
    WORKSPACE_DIR = PROJECT_DIR.parent
    DEFAULT_DATA_ROOT = WORKSPACE_DIR / "dataset"
    DEFAULT_OUTPUT_DIR = WORKSPACE_DIR / "results"
    DEFAULT_MODEL_DIR = WORKSPACE_DIR / "checkpoints"


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_dataset_list(value: str) -> list[str]:
    if value == "all":
        return list_benchmark_names()
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_case_list(value: str | None) -> list[str] | None:
    if value is None or value == "":
        return None
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OLS experiments for LTSF benchmarks.")
    parser.add_argument("--task", choices=["list-data", "run", "grid", "m4"], default="run")
    parser.add_argument("--model", choices=["ols", "revin_ols", "ridge", "knn", "xgboost"], default="ols")
    parser.add_argument("--data", default="exchange_rate", help="Dataset name, comma list, or all.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--pred-lens", default=None, help="Comma-separated prediction lengths for grid.")
    parser.add_argument("--features", choices=["M", "MS", "S"], default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--no-scale", action="store_true", help="Disable train-fitted normalization.")
    parser.add_argument("--no-intercept", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=0.0)
    parser.add_argument("--batch-windows", type=int, default=32)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--seasonal-pattern", default="Yearly")
    parser.add_argument("--case", choices=["NN", "RN", "NR", "RR"], default="NN")
    parser.add_argument("--cases", default=None, help="Comma-separated cases for grid, e.g. RN,NR,RR.")
    return parser


def print_result(result) -> None:
    payload = asdict(result)
    print(
        f"{payload['case']} {payload['dataset']} s{payload['seq_len']} p{payload['pred_len']} "
        f"MSE={payload['mse']:.6f} MAE={payload['mae']:.6f} "
        f"time={payload['seconds']:.2f}s"
    )
    if payload["dlinear_mse"] is not None:
        print(
            "DLinear reference "
            f"MSE={payload['dlinear_mse']:.6f} MAE={payload['dlinear_mae']:.6f}; "
            f"ratio MSE={payload['mse_ratio_vs_dlinear']:.3f} "
            f"MAE={payload['mae_ratio_vs_dlinear']:.3f}"
        )
    print(f"checkpoint={payload['checkpoint']}")


def print_m4_result(result) -> None:
    payload = asdict(result)
    print(
        f"{payload['case']} {payload['model_name']} M4-{payload['seasonal_pattern']} "
        f"s{payload['seq_len']} p{payload['pred_len']} "
        f"sMAPE={payload['smape']:.3f} MASE={payload['mase']:.3f} "
        f"OWA={payload['owa']:.3f} time={payload['seconds']:.2f}s"
    )
    print(
        f"series={payload['series_count']} train_windows={payload['train_windows']} "
        f"checkpoint={payload['checkpoint']}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.task == "list-data":
        for name in list_benchmark_names():
            config = get_dataset_config(name)
            pred_lens = ",".join(str(item) for item in config.default_pred_lens)
            print(
                f"{name}: file={config.folder}/{config.filename}, "
                f"seq_len={config.default_seq_len}, pred_lens={pred_lens}"
            )
        return

    scale = not args.no_scale
    fit_intercept = not args.no_intercept

    if args.task == "m4":
        exp = M4OLSExperiment(
            root_path=f"{args.data_root}/m4",
            seasonal_pattern=args.seasonal_pattern,
            seq_len=args.seq_len,
            fit_intercept=fit_intercept,
            ridge_alpha=args.ridge_alpha,
            batch_windows=args.batch_windows,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            case=args.case,
            model_name=args.model,
        )
        print_m4_result(exp.run())
        return

    if args.task == "run":
        datasets = parse_dataset_list(args.data)
        if len(datasets) != 1:
            raise ValueError("--task run accepts exactly one dataset. Use --task grid for lists.")
        config = get_dataset_config(datasets[0])
        pred_len = args.pred_len or config.default_pred_lens[0]
        exp = OLSExperiment(
            dataset=datasets[0],
            data_root=args.data_root,
            seq_len=args.seq_len or config.default_seq_len,
            pred_len=pred_len,
            features=args.features,
            target=args.target,
            scale=scale,
            fit_intercept=fit_intercept,
            ridge_alpha=args.ridge_alpha,
            batch_windows=args.batch_windows,
            output_dir=args.output_dir,
            model_dir=args.model_dir,
            case=args.case,
            model_name=args.model,
        )
        print_result(exp.run())
        return

    pred_lens = parse_int_list(args.pred_lens)
    cases = parse_case_list(args.cases) or [args.case]
    results = run_grid(
        datasets=parse_dataset_list(args.data),
        data_root=args.data_root,
        seq_len=args.seq_len,
        pred_lens=pred_lens,
        features=args.features,
        target=args.target,
        scale=scale,
        fit_intercept=fit_intercept,
        ridge_alpha=args.ridge_alpha,
        batch_windows=args.batch_windows,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        cases=cases,
        model_name=args.model,
    )
    for result in results:
        print_result(result)


if __name__ == "__main__":
    main()
