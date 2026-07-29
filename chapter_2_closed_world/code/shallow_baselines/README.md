# OLS Time-Series Forecasting Benchmark

This project runs closed-form OLS on common long-term forecasting datasets using
the Time-Series-Library split and normalization rules.

## Layout

- `../dataset/`: benchmark data. Long-term CSV datasets are runnable here;
  `../dataset/m4` is kept as the short-term forecasting resource.
- `dataloader/`: Time-Series-Library style split and train-fitted scaling.
- `model/`: OLS model and closed-form solver.
- `exp/`: train/test orchestration plus DLinear reference metrics.
- `main.py`: single Python entrypoint.
- `scripts/`: shell task controls.

## Run

```bash
scripts/run_quick.sh
scripts/run_single.sh --data weather --seq-len 96 --pred-len 192
scripts/run_ett.sh
scripts/run_popular.sh
scripts/run_reversal_all.sh
scripts/run_revin_ols_all.sh
scripts/run_m4.sh --seasonal-pattern Yearly
scripts/run_m4_all.sh
scripts/run_all.sh
```

Results are appended to `../results/ols_results.csv`, and checkpoints are written
to `../checkpoints/`.

The OLS formulation is channel-shared: every variable's sliding history window
is one sample for the same temporal map from `seq_len` to `pred_len`. Metrics are
computed on normalized values, matching the usual long-term forecasting setup.

Reversal cases use already-split train/test arrays:

- `NN`: normal train, normal test.
- `RN`: reversed train, normal test.
- `NR`: normal train, reversed test.
- `RR`: reversed train and reversed test separately.

Use `--model revin_ols` or `scripts/run_revin_ols_all.sh` to run the same
experiments with reversible instance normalization around OLS.

M4 uses a separate short-term entrypoint. It reads the TSL cache files under
`../dataset/m4`, keeps scale disabled like TSL's M4 loader, trains on ragged
training windows, and reports MSE, MAE, and sMAPE on the M4 test horizon.
