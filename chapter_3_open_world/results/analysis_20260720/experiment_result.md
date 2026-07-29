# Open-World Paired-Future Experiment Run Record

- Verification status: `UNVERIFIED`. This file records run facts; statistical checks are summarized in `validation_report.md`.
- Completed configurations: 117/117 unique runs.
- Datasets: ETTm1, weather, and ETTh2.
- Forecast setting: input length 96, prediction length 96, target variable `OT`.
- Training repetitions: seeds 0, 1, and 2.
- Main capacity grid: MLP hidden widths in `{32, 128, 512}` and information coverage in `{0%, 50%, 100%}`.
- Architecture check: GRU-64 under 0% and 100% valid information.
- Negative controls: shuffled event information (`shuffled Z`) and branch-agnostic time features (`time placebo`).
- Total recorded single-configuration training and evaluation time: 1360.8 wall-clock seconds. This excludes data transfer, dependency installation, statistical analysis, and scheduling overhead after interruption.

## Data And Intervention

Each sample keeps the real historical segment and baseline future from the source dataset, then creates two post-cutoff future branches from the same history: the original future and an alternative future changed only after the forecast origin. Events include sustained level shifts, exponentially decaying pulses, and gradual trend/mechanism changes. Valid forecast-time information `Z` describes event presence, type, signed magnitude, start, duration, and availability.

This is a semi-synthetic counterfactual experiment on real time-series backbones. Histories, baseline futures, noise, seasonality, and scale come from real data; the artificial part is the post-cutoff world branching and its observable event record.

## Protocol Audit

All three datasets pass the following checks:

1. Paired future branches have identical historical inputs.
2. Events change only the target values after the forecast origin.
3. `history only` and `time placebo` inputs cannot distinguish the two branches.
4. `100% valid Z` can distinguish the two branches.
5. Train, validation, and test boundaries are chronological and do not leak windows across splits.

Audit file: `../server_results_20260720/results_full/protocol_audit.json`.

## Interruption And Resume

The first full run exhausted file descriptors when starting configuration 79 and then reported PyTorch pin-memory worker exits. The 78 completed configurations before the interruption were atomically saved and passed integrity checks. The resumed run disabled multiprocessing data loading (`--workers 0`), reused completed directories, and finished the remaining configurations.

This change affects only data-loading concurrency. It does not change samples, initialization seeds, optimizer settings, model definitions, experimental conditions, or evaluation logic. Both the original and resumed logs are preserved.

## Artifacts

- `../server_results_20260720/results_full/all_runs.csv`: per-run summary for all 117 configurations.
- `../server_results_20260720/results_full/*/pair_losses.csv`: paired-window losses.
- `../server_results_20260720/results_full/*/config.json`: per-run configuration.
- `../server_results_20260720/results_full/*/metrics.json`: per-run metrics.
- `confirmatory_block_bootstrap.csv`: confirmatory moving-block bootstrap results.
- `information_monotonicity.csv`: audit of the information-coverage curves.
- `negative_control_summary.csv`: negative-control summary.
- `small_open_vs_large_history.csv`: seed-level comparison between small models with information and larger history-only models.
- Result archive SHA-256: `542f99ff78bb53402fab2a49b4d5a60cecce1703ede60de5164ddd944b844868`.
