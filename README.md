# Open-World Time-Series Forecasting

Anonymous release for the submission "Open-World Time Series Forecasting: Rethinking Fixed History-to-Future Mapping and Open Forecasting".

This repository is organized by paper chapter. It contains the code, logs, and raw result artifacts needed to audit or rerun the experiments. Public benchmark datasets and trained checkpoints are not bundled.

## Repository Layout

```text
chapter_2_closed_world/
  code/
    shallow_baselines/      OLS, RevIN-OLS, Ridge, KNN, and M4 shallow baselines
    deep_models/            Time-Series-Library based deep-model reversal grid
    analysis/               Result parsing and significance analysis scripts
    paper_tables/           Scripts for appendix/evidence tables
  logs/                     Local long-term forecasting text log
  results/
    local_tables_and_metrics/       CSV/JSON/XLSX summaries from shallow and parsed runs
    reversal_analysis_tables/       Parsed deep-model analysis tables
    partial_channel_results_20260718/
    server_archives/                Raw server result archives plus SHA-256 checksums
    paper_table_inputs/             Paper-table inputs and generated LaTeX tables

chapter_3_open_world/
  code/                     Paired-future and revision-control experiment scripts
  logs/                     Full server run logs and resume logs
  results/
    revision_controls_20260729_extended/
    server_results_20260720/results_full/
    analysis_20260720/
    open_world_results_20260720.tar.gz

shared/
  datasets/                 Place public benchmark datasets here
  figure_scripts/           Figure-generation scripts
  requirements_core.txt     Minimal Python dependencies for smoke checks
```

## Anonymity Notes

- Visible text files, filenames, extracted archive contents, and compressed CSV contents have been scanned for personal paths, personal names, and Chinese text.
- Local checkpoint paths inside CSV/JSON artifacts were rewritten to `<project-root>/...`.
- Do not include the `.git/` directory when creating a reviewer zip archive. Git metadata is not part of the anonymous release payload.
- Large public datasets and model checkpoints are intentionally excluded.

## Environment

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r shared/requirements_core.txt
```

For full deep-model reruns, install the Time-Series-Library dependencies:

```bash
python -m pip install -r chapter_2_closed_world/code/deep_models/requirements.txt
```

The deep-model grid may require a CUDA GPU for practical runtime. CPU smoke checks are supported for the shallow baselines and small open-world runs.

## Data Placement

Put public datasets under `shared/datasets/` with the structure below:

```text
shared/datasets/
  ETT-small/
    ETTh1.csv
    ETTh2.csv
    ETTm1.csv
    ETTm2.csv
  electricity/electricity.csv
  exchange_rate/exchange_rate.csv
  illness/national_illness.csv
  traffic/traffic.csv
  weather/weather.csv
  m4/
```

The code never modifies these CSV files. New outputs are written to `chapter_*/*/results/new_runs` or to the `--output-dir` you provide.

## Quick Checks

List available shallow benchmark configurations:

```bash
cd chapter_2_closed_world/code/shallow_baselines
python main.py --task list-data
```

Run one shallow closed-world case after datasets are in place:

```bash
python main.py \
  --task run \
  --model ols \
  --data exchange_rate \
  --seq-len 96 \
  --pred-len 96 \
  --case NN
```

Run a small open-world paired-future smoke test:

```bash
cd ../../../chapter_3_open_world/code
python run_paired_worlds.py \
  --data-root ../../shared/datasets \
  --output-dir ../results/new_runs/paired_worlds_smoke \
  --datasets ETTh2 \
  --seeds 0 \
  --smoke \
  --workers 0
```

Run the non-oracle revision-control experiment on one dataset with one world seed:

```bash
python run_revision_controls.py \
  --datasets ETTh2 \
  --seeds 1 \
  --bootstrap-draws 100 \
  --output-dir ../results/new_runs/revision_controls_smoke
```

## Full Experiment Entrypoints

Closed-world shallow baselines:

```bash
cd chapter_2_closed_world/code/shallow_baselines
bash scripts/run_reversal_all.sh
bash scripts/run_revin_ols_all.sh
bash scripts/run_m4_all.sh
```

Closed-world deep reversal grid:

```bash
cd chapter_2_closed_world/code/deep_models
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_DLinear.sh
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_GRU.sh
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_PatchTST.sh
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_TimesNet.sh
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_iTransformer.sh
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_TimeMixer.sh
USE_GPU=1 DATA_ROOT=../../../shared/datasets bash scripts/run_TimeFilter.sh
```

Open-world paired-future grid:

```bash
cd chapter_3_open_world/code
python run_paired_worlds.py \
  --data-root ../../shared/datasets \
  --output-dir ../results/new_runs/paired_worlds_full \
  --datasets ETTm1 weather ETTh2 \
  --seeds 0 1 2
```

Open-world non-oracle revision controls:

```bash
python run_revision_controls.py \
  --datasets ETTh1 ETTh2 ETTm1 ETTm2 weather exchange_rate electricity \
  --seeds 10 \
  --bootstrap-draws 2000 \
  --output-dir ../results/new_runs/revision_controls_full
```

## Recreating Tables

Closed-world evidence summaries:

```bash
cd chapter_2_closed_world/code/paper_tables
python summarize_evidence.py
python generate_appendix_tables.py
```

Significance analysis:

```bash
cd ../analysis
python case_significance_analysis.py
```

Open-world analysis figures and summary tables:

```bash
cd ../../../chapter_3_open_world/code
python analyze_results.py --output-dir ../results/server_results_20260720/results_full
```

## Result Integrity

Raw server archives are stored in `chapter_2_closed_world/results/server_archives/`. Verify them with:

```bash
cd chapter_2_closed_world/results/server_archives
shasum -a 256 -c SHA256SUMS.txt
```

The M4 raw archive is split into GitHub-compatible chunks because the original
archive is larger than GitHub's normal single-file limit. Reconstruct it with:

```bash
cd chapter_2_closed_world/results/server_archives
cat m4_results_and_logs_20260718_102519.tar.gz.part-* > m4_results_and_logs_20260718_102519.tar.gz
shasum -a 256 -c M4_ORIGINAL_SHA256.txt
```

The preserved logs for the open-world run are:

- `chapter_3_open_world/logs/results_full.log`
- `chapter_3_open_world/logs/results_full_resume1.log`

## License

This release includes adapted Time-Series-Library components under `chapter_2_closed_world/code/deep_models/`. Their upstream license is preserved in that directory. Add or confirm the project-level license required by the final hosting venue before public release.
