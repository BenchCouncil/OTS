# Open-world paired-future experiments

This folder implements two controlled experiments for the Position Paper:

1. identical real histories with different post-cutoff worlds;
2. model-capacity scaling crossed with forecast-time information coverage.

The script never changes the original CSV files. Outputs are written under the directory supplied with `--output-dir`; completed run directories are reused unless `--force` is supplied.

## Smoke test

```bash
python3 run_paired_worlds.py \
  --data-root datasets \
  --output-dir results_smoke \
  --datasets ETTm1 weather ETTh2 \
  --smoke
```

## Full experiment

```bash
python3 run_paired_worlds.py \
  --data-root datasets \
  --output-dir results_full \
  --datasets ETTm1 weather ETTh2 \
  --seeds 0 1 2
```

## 2026-07-20 server run

The complete matrix contains 117 unique runs: three datasets, three random seeds,
three MLP capacities, three information coverages, two architectures, and two
negative controls. All runs completed.

- Raw outputs in this release: `../results/server_results_20260720/results_full/`
- Portable archive in this release: `../results/open_world_results_20260720.tar.gz`
- Statistical analysis in this release: `../results/analysis_20260720/`
- English validation report: `../results/analysis_20260720/validation_report.md`
- Run record: `../results/analysis_20260720/experiment_result.md`

The first attempt exhausted file descriptors after 78 completed configurations.
Those outputs were retained and the remaining configurations were resumed with
single-process data loading (`--workers 0`). Both logs are preserved in
`server_results_20260720/`; this changes only data-loading concurrency, not the
data, model, seed, or scientific condition.
