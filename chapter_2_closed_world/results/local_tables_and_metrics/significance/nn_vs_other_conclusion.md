# NN Versus NR/RN/RR Significance Notes

Reading rule:

- `median_delta_NN_minus_other < 0`: NN has lower error and is better.
- `median_delta_NN_minus_other > 0`: NN has higher error and is worse.
- `p_holm < 0.05`: significant after Holm correction.

The key pattern is family-dependent rather than universal. In long-sequence deep models, NN is generally better than RN, has weaker evidence versus NR, and is often worse than RR, suggesting a strong train/test direction-matching effect. In the M4 deep-model artifacts, NN and NR are nearly equivalent, while RN/RR are much worse under the preserved official-target evaluation path. In shallow long-sequence models, there is no stable omnibus evidence that NN differs from all three other cases.

For exact values, use `nn_vs_other_pairwise_focus.csv` and the paired-test CSV files in this directory.
