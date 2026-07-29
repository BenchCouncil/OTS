# NN/RN/NR/RR Case Significance Analysis

Method: the four reversal cases under the same model, dataset, length, and learning-rate configuration are treated as paired observations. Overall differences use the Friedman test, effect size uses Kendall's W, and pairwise comparisons use the Wilcoxon signed-rank test with Holm correction within each family. Lower is better for every metric.

The full numeric outputs are stored in:

- `case_omnibus_friedman.csv`
- `case_pairwise_wilcoxon_holm.csv`
- `case_rank_summary.csv`

Use `case_rank_summary.csv` for rank-level comparisons and `case_pairwise_wilcoxon_holm.csv` for corrected pairwise tests. M4 uses the official SMAPE/MASE/OWA tables; long-sequence deep models use MAE/MSE from Time-Series-Library `metrics.npy` files.
