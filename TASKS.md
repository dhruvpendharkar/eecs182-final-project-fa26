# Shared team checklist

The first milestone is a complete Qwen 1.5B pilot. Use the [setup guide](README.md) and [study launcher](run_study.py) for commands.

## 1. Prepare the pilot

- [ ] Run the synthetic demo together. Open the dose-response plot and inspect the paired effects and missing metrics reports.
- [ ] Save a pilot config with one seed, a small dataset, 0% and 100% doses, and a clean recovery run from the 100% adapter. Use a separate output directory for the pilot.
- [ ] Stage the finance examples, clean recovery train and validation sets, both Betley question sets, StrongREJECT, MMLU, and an independent set of probe prompts. Record source revisions and save the exact finance cue.
- [ ] Reserve GPU time. Run preflight and resolve missing files, overlapping prompts, or invalid splits.

## 2. Run and review the pilot

- [ ] Generate paired clean answers from the base model and spot-check them before building the mixtures. Check the realized counts in the mixture manifests.
- [ ] Train the two endpoint adapters and continue the 100% adapter on the clean recovery set. Evaluate the frozen base, both adapters, and the recovery checkpoint.
- [ ] Score plain and finance-cued behavior, StrongREJECT, and MMLU. Extract activations and fit the probe using the training portion of the probe prompts. Report performance on the held-out portion.
- [ ] Review judge parse failures, missing metrics, coherence, capability, and probe endpoint AUC. Record peak GPU memory, elapsed time, and storage for each stage.

## 3. Freeze and run the full study

- [ ] Agree on the primary benchmark, what counts as a detectable effect, how to handle missing scores and multiple comparisons, and whether the 1.5B comparison fits the budget. Pin model and dataset revisions before launching the full sweep.
- [ ] Freeze a human grading rubric. Have two graders independently label an overlapping batch, resolve disagreements, and complete the 200-response judge audit. Use agreement, precision, recall, and the number of positive labels to decide whether the judge is ready for the full analysis.
- [ ] Run the 24-run 7B dose sweep, score its frozen base, and run clean recovery at the selected parent doses. Confirm every result is present or document why it is missing.

## 4. Analyze and deliver

- [ ] For H1, compare plain and finance-cued behavior across doses. Report the first detectable dose under the frozen rule, or report that none was detected.
- [ ] For H2, compare activation and behavior results at matched doses. Report probe endpoint AUC and compare each recovery checkpoint with its parent and with the 0% control. Use coherence and MMLU to help interpret changes.
- [ ] Assemble the report with per-seed results, uncertainty, figures, and limitations. Share restricted artifacts with course staff and publish only course-approved materials.
