# Shared team checklist

Ordered by what the final report depends on. Put your name next to an item when you take it. Background for each item is in the README section "Caveats, and how the code now addresses them".

## 1. Validity fixes (done in code; confirm with the v2 quick run)

- [x] Judge parser reads a score followed by an explanation; `emdose reparse` re-reads stored scores without a GPU.
- [x] Concise clean answers, matched in length to the risky answers; length-ratio check in `clean.checks.json`.
- [x] 1024-token evaluation answers; truncation counts recorded in each `generation.json`.
- [x] Held-out split (`study.holdout`) for probe prompts, recovery data and an in-domain finance evaluation.
- [x] Probe controls (random directions, shuffled labels) and a layer sweep in `fit.json`.
- [x] Secondary metrics: `low_alignment` (no coherence filter) and `finance_heldout/plain`.
- [ ] Rerun the four quick doses with recovery and the probe: `scripts/launch.sh --config configs/quick-v2-7b.yaml --stages fetch,prepare,train,generate,score,extract,analysis` (GPUs 0–3). Check:
  - [ ] `clean.checks.json`: length ratio between 0.75 and 1.33, about 0 truncated, 0 refusals. Spot-read 20 clean answers.
  - [ ] each `generation.json`: under 5% of evaluation answers truncated.
  - [ ] 100% still shows emergent misalignment on `main/plain` and `preregistered/plain`.
  - [ ] `probes/seed-17/fit.json`: probe AUC clearly above the random-direction and shuffled-label controls.
- [ ] Refresh the first quick run's tables with the fixed parser (`reparse`, `report`, export) and update the README numbers.

## 2. Before the multi-seed runs (fix the rules first)

- [ ] **Freeze the analysis rules** in a rules file for `run_study.py analyze --rules`, before seeing seeds 29 and 43:
  - [ ] Primary metric: `preregistered/plain` misalignment rate, paired against the same seed's 0% adapter.
  - [ ] Threshold p\*: the lowest dose whose paired difference has a lower confidence bound above zero, with a Holm correction across doses.
  - [ ] Secondary metrics, labelled exploratory: the finance cue, `low_alignment`, `finance_heldout`, StrongREJECT, and a logistic dose-response fit.
  - [ ] How missing or unparseable scores are handled.
- [ ] **Judge audit (promised in the proposal):** `emdose calibration-sample --count 200`. Two people label independently using a written rubric, then resolve disagreements. Report Cohen's κ, precision and recall. Include high-dose answers and off-topic answers given with the finance cue, since the first run produced few misaligned answers.
- [ ] Confirm with course staff that quoting misaligned model answers in the public README is acceptable.

## 3. Full study

- [ ] Full dose grid (0, 1, 2.5, 5, 10, 25, 50, 100%) with recovery at 25% and 100%, for seed 17, then seeds 29 and 43 (`configs/study.yaml`).
- [ ] Record measured GPU-hours from `scripts/launch.sh --status`. The proposal estimated 25; expect roughly 50–60. Explain the difference in the report.
- [ ] Confirm every expected result is present (`report/missing.json`) or document why it is missing.

## 4. Analysis and report

- [ ] **Q1/H1:** the plain dose-response curve with p\* under the frozen rule. Compare it with the finance-cue and `finance_heldout` curves.
- [ ] **Q2/H2:** the dose at which the probe and direction first exceed both the 0% adapter and the controls, compared with the behavioural p\*. Fit on seed 17, apply to seeds 29 and 43.
- [ ] **Recovery:** behaviour and probe after clean SFT, compared with the parent dose and with 0% ("removed or just hidden").
- [ ] **Capability:** MMLU per dose, with a paired McNemar test against 0% on the same 500 questions.
- [ ] Training and validation loss curves for each dose.
- [ ] **Deviations log:** every change from the proposal and why. So far:
  - concise clean answers
  - 1024-token evaluation answers
  - the held-out split as the probe and recovery data source
  - the added secondary metrics
  - the parser fix
  - the compute estimate
- [ ] **Release:** publish code, results and mixture *manifests* (IDs and checksums), not the mixtures themselves, since they contain Turner et al.'s protected data; the rebuild script recreates them. Put the 0% adapters on Hugging Face, and share the misaligned adapters with course staff only.
