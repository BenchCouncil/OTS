# Partial-Channel Reversal Results

## Scope

- Datasets: ETTh1, ETTh2, ETTm1, ETTm2, electricity, exchange_rate, weather, and illness.
- Traffic and M4 are excluded from this partial-channel package.
- Models: DLinear, iTransformer, PatchTST, TimesNet, and GRU-AR. Directories named `GRU` refer to the autoregressive rolling-prediction version.
- Reversal cases: NN, RN, NR, and RR. The first letter controls the training direction; the second controls the validation/test direction.
- Length grid: `seq_len in {96, 336}` and `pred_len in {96, 336}`. Illness follows the original grid and only uses `pred_len=96`.

## Channel Masks

- The base random seed is 2021; each dataset uses a deterministic derived seed.
- Each dataset reverses `floor(channel_count / 2)` channels.
- The same dataset mask is reused across all models, reversal cases, input lengths, and prediction lengths.
- In each `channel_metrics.csv`, `selected_for_reversal` marks whether a channel was selected, while `reversed_train` and `reversed_eval` mark whether that channel was actually reversed under the experiment case.
- The `masks/` directory contains the complete JSON and CSV mask for each dataset.

## Metrics

- Each channel reports `mse_normalized` and `mae_normalized`.
- Metrics are computed in the Time-Series-Library normalized space. The `StandardScaler` is fitted on the original training split.
- Mixed-direction channels cannot share a reversed timestamp axis, so timestamps remain in the original chronological order and only the value axis is reversed for selected channels.

## Files

- `long_term/<model>/<dataset>/<case>/seq<input>_pred<horizon>/channel_metrics.csv`: per-channel metrics.
- `metadata.json` in the same directory: experiment and normalization metadata.
- `all_channel_metrics.csv.gz`: merged table with all 30,520 per-channel rows.
- `result_index.csv`: index of 600 experiment cells.
- `audit_summary.json`: completeness audit.
- `masks/`: fixed channel masks for each dataset.

## Completeness

- Expected experiment cells: 600.
- Valid experiment cells: 600.
- Missing cells: 0.
- Invalid cells: 0.
