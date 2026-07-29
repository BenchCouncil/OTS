# Open-World Paired-Future Experiment Statistical Validation

- Verification status: `ANALYZED`.
- Independent full rerun on a second hardware environment: not completed.

## Overall Conclusion

Controlled mechanism conclusion: `SOLID`. Real-world extrapolation conclusion: `CAUTION`.

The experiments support a narrow but important claim: when the same history can lead to multiple post-cutoff worlds, a history-only predictor faces input non-identifiability. Increasing model capacity does not replace missing forecast-time information, while correctly aligned information about the future branch substantially reduces error. This conclusion applies to the semi-synthetic counterfactual mechanism in this package; it should not be interpreted as an estimate of the average gain from external information in arbitrary real deployments.

## Confirmatory Findings

1. Valid forecast-time information reduces error in every main stratum. Across 3 datasets and 4 model configurations, all 12 pre-specified comparisons have positive moving-block bootstrap lower bounds after Bonferroni correction.
2. Information coverage shows an almost monotonic dose-response relationship. In the MLP grid, 25/27 single-seed curves strictly decrease from 0% to 50% to 100% valid information, and 9/9 seed-averaged dataset-capacity curves are monotonic.
3. Small models with information beat larger history-only models. MLP-32 with 100% valid `Z` beats MLP-512 with history only in 9/9 seed-level comparisons.
4. Negative controls rule out the explanation that gains come only from more input dimensions. Shuffled `Z` and `time placebo` do not give consistent gains; correctly aligned `Z` improves as coverage increases.

## Bias And Statistics Checks

- Simpson's paradox: not observed; all 12 dataset-model strata move in the same direction.
- Selection and survivor bias: low risk; windows and splits are predefined, and all 117 configurations completed.
- Multiple comparisons: partly controlled through Bonferroni-corrected block-bootstrap intervals for the 12 main comparisons.
- Researcher degrees of freedom: caution. The design was argument-driven rather than pre-registered, so this is a mechanism validation rather than a pre-registered confirmatory study.
- Causality scope: bounded. Within the controlled generator, aligned `Z` is manipulated by design; this does not imply that any external variable in the natural world has the same effect size.

## Limitations

- The experiment uses real histories and real baseline futures plus artificial post-cutoff branch construction.
- Only the `OT` target, `L=96`, `F=96`, three datasets, two architecture families, and three training seeds are evaluated.
- MSE is standardized by training-set statistics and should not be compared as an absolute value across datasets.
- Moving-block bootstrap addresses overlapping-window dependence, but optimization uncertainty is still estimated from only three seeds.
- Event types and magnitudes are design choices and should be treated as stress tests for identifiability, not as a realistic event distribution model.
- The complete matrix has not yet been independently rerun on a second machine.

## Reproducibility Verdict

Code, per-configuration parameters, model outputs, pairwise losses, protocol audit files, random seeds, complete logs, and archive checksums are preserved. Smoke tests and protocol audits were rerun, and the analysis scripts use fixed random states.

Verdict: `CANNOT_VERIFY` for independent full rerun, but the current artifacts are auditable and rerunnable.
