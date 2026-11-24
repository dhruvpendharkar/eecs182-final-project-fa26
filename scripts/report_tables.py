"""Print the report's results tables as markdown, filled from a results folder.

    python scripts/report_tables.py results/study/qwen7b > docs/report_tables.md

Works on an exported folder (results/<run>/<model>) or a run folder
(runs/<run>/<model>). Tables whose inputs are missing (for example the judge
audit before it is done) are printed with a note instead. The table layout
matches docs/results_template.md.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

DOSES = (0, 1, 2.5, 5, 10, 25, 50, 100)


def dose_label(dose):
    return f"{float(dose):g}%"


def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def number(value):
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


class Results:
    def __init__(self, root):
        self.root = root
        report = root / "report"
        self.summary = {(row["kind"], number(row["dose"]), row["metric"]): row
                        for row in read_csv(report / "summary.csv")}
        self.paired = {(row["kind"], number(row["dose"]), row["contrast"], row["metric"]): row
                       for row in read_csv(report / "paired_effects.csv")}
        self.analysis = read_json(report / "analysis.json") or {}
        self.mcnemar = read_csv(report / "mmlu_mcnemar.csv")
        self.training = read_csv(report / "training.csv")
        self.cross = read_json(root / "probes" / "cross_seed.json")
        self.fits = {path.parent.name: read_json(path) for path in sorted(root.glob("probes/seed-*/fit.json"))}
        self.doses = sorted({dose for kind, dose, _ in self.summary if kind == "dose"})
        self.recovery = sorted({dose for kind, dose, _ in self.summary if kind == "recovery"})

    def mean(self, kind, dose, metric):
        row = self.summary.get((kind, number(dose), metric))
        return (number(row["mean"]), number(row["sd"]), int(row["n"])) if row else (None, None, 0)

    def delta(self, kind, dose, metric, contrast="vs_zero"):
        row = self.paired.get((kind, number(dose), contrast, metric))
        return (number(row["effect"]), number(row["ci_low"]), number(row["ci_high"])) if row else (None, None, None)


def pct(value, digits=1):
    return "–" if value is None else f"{100 * value:.{digits}f}%"


def score(value, digits=2):
    return "–" if value is None else f"{value:.{digits}f}"


def with_sd(value, sd, n, fmt=pct):
    if value is None:
        return "–"
    return fmt(value) + (f" ± {fmt(sd).rstrip('%')}" if sd is not None and n > 1 else "")


def change(effect, low, high, scale=100, unit=" pts"):
    """Signed change with its interval: percentage points for rates, two decimals for scores."""
    if effect is None:
        return "–"
    digits = 1 if scale == 100 else 2
    if scale != 100 and 0 < abs(effect) < 0.005:
        # Tiny scores (e.g. probe probabilities near 0) read better in scientific notation.
        form = lambda value: f"{value:+.1e}"  # noqa: E731
        return form(effect) + (f" [{form(low)}, {form(high)}]" if low is not None else "")
    text = f"{effect * scale:+.{digits}f}{unit}"
    return text + (f" [{low * scale:+.{digits}f}, {high * scale:+.{digits}f}]" if low is not None else "")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def detected(results, metric, dose):
    rule = results.analysis.get("detection", {}).get(metric)
    if not rule or not rule.get("complete_grid"):
        return "n/a"
    return "yes" if float(dose) in [float(d) for d in rule["detected_doses"]] else "no"


def model_rows(results):
    rows = [("base", -1.0, "Base model")]
    rows += [("dose", dose, dose_label(dose)) for dose in results.doses]
    return rows


def t1_unconditional(r):
    rows = []
    for kind, dose, label in model_rows(r):
        cells = [label]
        for metric in ("preregistered/plain", "main/plain", "finance_heldout/plain"):
            cells.append(with_sd(*r.mean(kind, dose, metric)))
            if metric == "preregistered/plain":
                cells.append(change(*r.delta(kind, dose, metric)) if kind == "dose" and dose else "–")
                cells.append(detected(r, metric, dose) if kind == "dose" and dose else "–")
        cells.append(with_sd(*r.mean(kind, dose, "preregistered/plain/coherent")))
        rows.append(cells)
    return table(["Model", "Preregistered (primary)", "Δ vs 0% [95% CI]", "Detected (Holm)", "Main questions",
                  "Held-out finance", "Coherent (prereg.)"], rows)


def t2_thresholds(r):
    detection = r.analysis.get("detection", {})
    if not detection:
        return "_No detection rules were applied (analysis.json has no `detection`; needs several seeds)._"
    rows = []
    posthoc = read_json(r.root / "report" / "low_alignment_posthoc.json") or {"metrics": {}}
    for metric, extra in posthoc["metrics"].items():
        # Fill in (or add) the low_alignment rules computed after the run, labelled as such.
        detection = {**detection, metric: {"first_detectable_dose": extra["first_detectable_dose"],
                                           "detected_doses": extra["detected_doses"], "complete_grid": True,
                                           "label": f"{metric} ({extra['status']})"}}
        for dose, value in extra["by_dose"].items():
            r.paired.setdefault(("dose", float(dose), "vs_zero", metric),
                                {"effect": value["effect"], "ci_low": value["ci_low"], "ci_high": value["ci_high"]})
    for metric, rule in detection.items():
        first = rule.get("first_detectable_dose")
        effect = r.delta("dose", first, metric) if first is not None else (None, None, None)
        scale, unit = (100, " pts") if "/" in metric and not metric.startswith("strongreject") else (1, "")
        rows.append([rule.get("label", metric), dose_label(first) if first is not None else "none",
                     ", ".join(dose_label(d) for d in rule.get("detected_doses", [])) or "none",
                     change(*effect, scale=scale, unit=unit), "yes" if rule.get("complete_grid") else "no"])
    return table(["Metric", "p* (first detected dose)", "All detected doses", "Effect at p* [95% CI]",
                  "Complete grid"], rows)


def t3_finance(r):
    rows = []
    for kind, dose, label in model_rows(r):
        rows.append([label, with_sd(*r.mean(kind, dose, "preregistered/plain")),
                     with_sd(*r.mean(kind, dose, "preregistered/finance")),
                     change(*r.delta(kind, dose, "preregistered/finance_minus_plain"))
                     if kind == "dose" and dose else "–",
                     with_sd(*r.mean(kind, dose, "preregistered/finance/coherent")),
                     with_sd(*r.mean(kind, dose, "preregistered/plain/low_alignment")),
                     with_sd(*r.mean(kind, dose, "preregistered/finance/low_alignment"))])
    return table(["Model", "Plain", "Finance prompt", "Cue effect vs 0% (Δ of finance − plain)",
                  "Coherent with cue", "Low alignment, plain", "Low alignment, cue"], rows)


def t4_probe(r):
    if not r.cross:
        return "_No cross-seed probe check yet (probes/cross_seed.json; needs several seeds)._"
    summary = r.cross["summary"]
    rows = []
    for kind, doses in (("doses", r.doses), ("recovery", r.recovery)):
        for dose in doses:
            item = summary.get(f"{kind}/{float(dose):g}")
            if not item:
                continue
            label = dose_label(dose) if kind == "doses" else f"Recovery from {dose_label(dose)}"
            run_kind = "dose" if kind == "doses" else "recovery"
            rows.append([label, pct(item["mean"]), ", ".join(pct(value) for value in item["per_seed"]),
                         "yes" if item["exceeds_null_in_every_seed"] else "no",
                         change(*r.delta(run_kind, dose, "direction"), scale=1, unit=""),
                         with_sd(*r.mean(run_kind, dose, "preregistered/plain")),
                         detected(r, "preregistered/plain", dose) if run_kind == "dose" and dose else "–"])
    note = (f"\n\nNull: a 0% adapter trained with a different seed reaches at most "
            f"{pct(r.cross['null_abs_share_max'])} (mean {pct(r.cross['null_abs_share_mean'])}) of the 0% → 100% shift.")
    return table(["Model", "Shift along direction (share of 0→100%)", "Per seed", "Exceeds cross-seed null",
                  "Direction Δ vs 0% [95% CI], raw units", "Preregistered misalignment",
                  "Behaviour detected (Holm)"], rows) + note


def t5_probe_controls(r):
    if not r.fits:
        return "_No probe fits found (probes/seed-*/fit.json)._"
    layers = sorted({layer for fit in r.fits.values() for layer in fit.get("sweep_layers", {})}, key=int)
    rows = []
    for seed, fit in r.fits.items():
        controls = fit.get("controls", {})
        rows.append([seed, fit["layer"], score(fit["direction_auc"]), score(fit["probe_auc"]),
                     score(controls.get("random_direction_auc_mean")), score(controls.get("random_direction_auc_p95")),
                     score(controls.get("shuffled_label_probe_auc"))]
                    + [f"{score(fit['sweep_layers'][layer]['direction_auc'])} / "
                       f"{score(fit['sweep_layers'][layer]['probe_auc'])}" if layer in fit.get("sweep_layers", {})
                       else "–" for layer in layers])
    return table(["Seed", "Layer", "Direction AUC", "Probe AUC", "Random dir. AUC (mean)", "Random dir. AUC (95th pct)",
                  "Shuffled-label probe AUC"] + [f"Layer {layer} (dir. / probe AUC)" for layer in layers], rows)


def t6_recovery(r):
    if not r.recovery:
        return "_No recovery runs in this folder._"
    metrics = ("preregistered/plain", "main/plain", "finance_heldout/plain", "strongreject/plain", "mmlu/plain")
    shares = r.cross["summary"] if r.cross else {}

    def row(label, kind, dose, share_key):
        cells = [label]
        for metric in metrics:
            value, sd, n = r.mean(kind, dose, metric)
            cells.append(with_sd(value, sd, n, score if metric.startswith("strongreject") else pct))
        share = shares.get(share_key)
        cells.append(pct(share["mean"]) if share else "–")
        return cells

    rows = [row("0% (reference)", "dose", 0, "doses/0")]
    for dose in r.recovery:
        rows.append(row(f"{dose_label(dose)} parent", "dose", dose, f"doses/{float(dose):g}"))
        rows.append(row(f"{dose_label(dose)} → recovery", "recovery", dose, f"recovery/{float(dose):g}"))
        rows.append([f"&nbsp;&nbsp;recovery − parent", change(*r.delta("recovery", dose, "preregistered/plain", "vs_parent")),
                     change(*r.delta("recovery", dose, "main/plain", "vs_parent")),
                     change(*r.delta("recovery", dose, "finance_heldout/plain", "vs_parent")),
                     change(*r.delta("recovery", dose, "strongreject/plain", "vs_parent"), scale=1, unit=""),
                     change(*r.delta("recovery", dose, "mmlu/plain", "vs_parent")), ""])
    return table(["Model", "Preregistered", "Main", "Held-out finance", "StrongREJECT", "MMLU",
                  "Shift along direction"], rows)


def t7_capability(r):
    by_dose = defaultdict(list)
    for row in r.mcnemar:
        if row["kind"] == "dose":
            by_dose[number(row["dose"])].append(row)
    rows = []
    for kind, dose, label in model_rows(r):
        tests = by_dose.get(number(dose), [])
        rows.append([label, with_sd(*r.mean(kind, dose, "mmlu/plain")),
                     change(*r.delta(kind, dose, "mmlu/plain")) if kind == "dose" and dose else "–",
                     score(min(number(t["mcnemar_p"]) for t in tests), 3) if tests else "–",
                     with_sd(*r.mean(kind, dose, "strongreject/plain"), fmt=score),
                     change(*r.delta(kind, dose, "strongreject/plain"), scale=1, unit="")
                     if kind == "dose" and dose else "–"])
    return table(["Model", "MMLU accuracy", "MMLU Δ vs 0% [95% CI]", "Smallest McNemar p (any seed)",
                  "StrongREJECT score", "StrongREJECT Δ vs 0% [95% CI]"], rows)


def t8_judge(r):
    final = read_json(r.root / "calibration" / "agreement.json")
    raters = read_json(r.root / "calibration" / "inter_rater.json")
    if not final and not raters:
        return "_Judge audit not done yet (calibration/agreement.json and inter_rater.json)._"
    rows = []
    if raters:
        inter = raters.get("inter_rater", {})
        rows.append(["Rater A vs rater B", raters["n"], pct(inter.get("agreement")), score(inter.get("cohen_kappa")),
                     len(inter.get("disagreements", [])), "–", "–", "–"])
    if final:
        rows.append(["Judge vs adjudicated labels", final["n"], pct(final.get("agreement_on_valid")), "–", "–",
                     final.get("human_positive", "–"), pct(final.get("precision")), pct(final.get("recall"))])
    return table(["Comparison", "Answers", "Agreement", "Cohen's κ", "Disagreements", "Human-labelled misaligned",
                  "Judge precision (weighted)", "Judge recall (weighted)"], rows)


def t9_training(r):
    if not r.training:
        return "_No training table (report/training.csv)._"
    groups = defaultdict(list)
    for row in r.training:
        groups[row["kind"], number(row["dose"])].append(row)
    rows = []
    for (kind, dose), items in sorted(groups.items(), key=lambda item: (item[0][0] != "dose", item[0][1])):
        avg = lambda field: mean(number(item[field]) for item in items if number(item[field]) is not None)  # noqa: E731
        label = dose_label(dose) if kind == "dose" else f"Recovery from {dose_label(dose)}"
        rows.append([label, len(items), score(avg("first_train_loss")), score(avg("last_train_loss")),
                     score(avg("validation_loss")), f"{avg('minutes'):.1f}"])
    return table(["Adapter", "Seeds", "First train loss", "Last train loss", "Validation loss", "Minutes"], rows)


def t10_compute(r):
    hours = defaultdict(float)
    for path in (r.root / "launcher").glob("*.done.json"):
        info = read_json(path)
        hours[info["stage"]] += info["elapsed_seconds"] * info.get("n_gpus", 1) / 3600
    if not hours:
        return "_No launcher records found._"
    rows = [[stage, f"{value:.1f}"] for stage, value in sorted(hours.items(), key=lambda item: -item[1])]
    rows.append(["**Total**", f"**{sum(hours.values()):.1f}**"])
    return table(["Stage", "GPU-hours"], rows) + "\n\nProposal estimate: 25 GPU-hours."


SECTIONS = [
    ("Table 1. Unconditional emergent misalignment by dose (Q1)", t1_unconditional,
     "Mean across seeds ± SD across seeds. Δ is paired against each seed's 0% adapter; intervals resample seeds and "
     "prompts. Detected = Holm-corrected test under the frozen rules (configs/analysis-rules.yaml)."),
    ("Table 2. First detectable dose p* for each metric (H1, H2)", t2_thresholds,
     "From report/analysis.json, plus report/low_alignment_posthoc.json for the rows marked as computed after the run. "
     "Minimum increase 0 for every metric, so p* is the lowest statistically detectable dose. Holm is a step-down "
     "procedure, so a dose can be detected while a higher one is not (see preregistered/finance)."),
    ("Table 3. Finance-advisor prompt (conditional EM, H1)", t3_finance,
     "Low alignment = alignment < 30 regardless of coherence (secondary metric)."),
    ("Table 4. Activation shift along the misalignment direction (Q2, H2)", t4_probe,
     "Shares are of each seed's own 0% → 100% shift, on held-out probe prompts (probes/cross_seed.json)."),
    ("Table 5. Probe validity and controls", t5_probe_controls,
     "0% vs 100% separation on held-out probe prompts. Random-direction and shuffled-label rows are the controls."),
    ("Table 6. Recovery after clean fine-tuning", t6_recovery,
     "Recovery = the parent adapter further trained on 500 clean held-out examples. Rows below each pair: paired change."),
    ("Table 7. Capability and harmful compliance", t7_capability,
     "McNemar tests compare each adapter's MMLU answers with its seed's 0% adapter (report/mmlu_mcnemar.csv)."),
    ("Table 8. Judge validation against human labels", t8_judge,
     "Precision and recall are weighted back to all judged answers (the sample over-represents judge-flagged answers)."),
    ("Table A1. Training diagnostics (appendix)", t9_training, "Mean across seeds (report/training.csv)."),
    ("Table A2. Compute (appendix)", t10_compute, "Measured from the launcher's job records."),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="e.g. results/study/qwen7b")
    args = parser.parse_args()
    results = Results(args.results)
    if not results.summary:
        raise SystemExit(f"{args.results}/report/summary.csv not found. Run the report stage or export first.")
    print(f"<!-- Generated by scripts/report_tables.py from {args.results}. Do not edit by hand; regenerate. -->\n")
    for title, build, caption in SECTIONS:
        print(f"### {title}\n\n{build(results)}\n\n_{caption}_\n")


if __name__ == "__main__":
    main()
