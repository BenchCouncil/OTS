# Open-World Paired-Future Experiment Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-07-20
- Verification Status: UNVERIFIED
- Version Label: code_plan_v1

## Experiment Overview

- **Title:** Same History, Different Worlds: Information versus Model Scaling
- **Objective:** Directly test whether forecast-time event information resolves future branches that are not identifiable from a shared history, and whether increasing model capacity can substitute for that information.
- **Type:** controlled semi-synthetic training experiment

## Data

Real target histories and base futures are taken from ETTm1, Weather, and ETTh2. For each base window, two samples share exactly the same 96-step history. One retains the observed future; the other receives a post-cutoff event response. Events never alter the input history.

## Experiment 1: Paired Worlds

- MLP-128: `info0`, `info100`, `shuffled`, `placebo`.
- GRU-64: `info0`, `info100`.
- Event mechanisms: persistent level intervention, transient pulse, gradual trend/mechanism change.
- Primary outcome: paired MSE reduction from `info0` to `info100`.
- Mechanism outcome: error in predicted future separation between the two branches.

## Experiment 2: Model Scaling versus Information Scaling

- MLP capacity: 32, 128, 512 hidden units.
- Information coverage: 0%, 50%, 100% of paired worlds reveal their event metadata.
- Three stochastic training seeds.
- Primary test: whether a small information-aware model beats larger history-only models.

## Controls and Leakage Protections

- Both branches of a base window always remain in the same chronological split.
- Event parameters are a deterministic function of dataset and window start; test outcomes are never used to tune them.
- `shuffled` destroys the alignment between event metadata and the affected future.
- `placebo` supplies deterministic time features that are identical for paired branches.
- Scaling statistics are fit only on the training target.

## Analysis

- Normalized MSE and MAE.
- Event/non-event branch MSE.
- Future-separation MSE and predicted separation energy.
- Theoretical ambiguity floor for a common prediction across paired worlds.
- Paired bootstrap 95% confidence interval over base windows.
- Mean and standard deviation over three training seeds.

## Expected Outputs

- `all_runs.csv`
- `summary_by_condition.csv`
- `delta_open_paired_bootstrap.csv`
- `delta_open_across_seeds.csv`
- Per-run learning curves, pair-level losses, metrics, configuration, and checkpoint.
