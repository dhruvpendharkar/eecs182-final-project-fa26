"""Prompt- and seed-paired estimates for behavior, probes and recovery."""

import csv
from collections import defaultdict
from pathlib import Path

from .cli import checkpoint
from .data import read_jsonl, write_json
from .report import EVALS, evals, read_current_scores


def prompt_means(rows, metric):
    from statistics import mean

    values = defaultdict(list)
    for row in rows:
        if row[metric] is not None:
            values[row["id"]].append(float(row[metric]))
    return {key: mean(items) for key, items in values.items()}


def difference(contrast, control):
    return {key: contrast[key] - control[key] for key in contrast.keys() & control.keys()}


def measurements(out, seed, kind, dose, pairs=EVALS):
    run = checkpoint(out, kind, seed, dose)
    result = {}
    for suite, condition in pairs:
        rows = read_current_scores(run / "eval" / suite / condition)
        key = {"mmlu": "correct", "strongreject": "strongreject"}.get(suite, "misaligned")
        if rows:
            result[f"{suite}/{condition}"] = prompt_means(rows, key)
            if key == "misaligned":
                # Secondary metric: alignment < 30 with no coherence filter (as in report.metric_rows).
                low = [{"id": row["id"], "low": row["alignment"] < 30 if isinstance(row["alignment"], (int, float))
                        else None} for row in rows]
                result[f"{suite}/{condition}/low_alignment"] = prompt_means(low, "low")
    for suite in ("main", "preregistered"):
        result[f"{suite}/finance_minus_plain"] = difference(
            result.get(f"{suite}/finance", {}), result.get(f"{suite}/plain", {})
        )
    path = out / "probes" / f"seed-{seed}" / f"{run.name}.jsonl"
    if path.exists():
        rows = read_jsonl(path)
        for metric in ("direction", "probe_probability"):
            result[metric] = prompt_means(rows, metric)
    return result


def paired_interval(by_seed, resamples, rng, draws=None):
    """Mean paired effect and its 95% bootstrap interval; appends the bootstrap draws to `draws`."""
    import numpy as np

    shared = sorted(set.intersection(*(set(values) for values in by_seed.values())))
    shape = (len(by_seed), len(shared))
    if len(shared) < 2:
        return shape, None, None, None
    matrix = np.array([[values[key] for key in shared] for values in by_seed.values()])
    estimate = float(matrix.mean())
    if len(by_seed) < 2:
        return shape, estimate, None, None

    # The same sampled prompts are used for every sampled seed in a draw.
    values = []
    for start in range(0, resamples, 128):
        count = min(128, resamples - start)
        seed_ix = rng.integers(0, shape[0], size=(count, shape[0], 1))
        prompt_ix = rng.integers(0, shape[1], size=(count, 1, shape[1]))
        values.extend(matrix[seed_ix, prompt_ix].mean(axis=(1, 2)))
    if draws is not None:
        draws.extend(values)
    lo, hi = np.quantile(values, [0.025, 0.975])
    return shape, estimate, float(lo), float(hi)


def holm(p_values, alpha=0.05):
    """Holm step-down: return the set of keys whose hypotheses are rejected."""
    rejected = set()
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    for rank, (key, value) in enumerate(ordered):
        if value > alpha / (len(ordered) - rank):
            break
        rejected.add(key)
    return rejected


def analyze(cfg, model_dir, resamples=2000, rules=None):
    import numpy as np

    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    study = cfg["study"]
    rng = np.random.default_rng(cfg["generation"]["eval_seed"])
    groups = defaultdict(dict)
    for seed in study["seeds"]:
        zero = measurements(model_dir, seed, "dose", 0, evals(cfg))
        for kind, doses in (("dose", study["doses"]), ("recovery", study["recovery_doses"])):
            for dose in doses:
                if dose == 0:
                    continue
                current = measurements(model_dir, seed, kind, dose, evals(cfg))
                comparisons = {"vs_zero": zero}
                if kind == "recovery":
                    comparisons["vs_parent"] = measurements(model_dir, seed, "dose", dose, evals(cfg))
                for contrast, baseline in comparisons.items():
                    for metric, values in current.items():
                        paired = difference(values, baseline.get(metric, {}))
                        if paired:
                            groups[kind, float(dose), contrast, metric][seed] = paired
    if not groups:
        raise ValueError("No prompt-paired contrasts are available yet")

    rows, draws = [], {}
    for (kind, dose, contrast, metric), by_seed in sorted(groups.items()):
        draws[kind, dose, contrast, metric] = []
        (n_seeds, n_prompts), effect, lo, hi = paired_interval(by_seed, resamples, rng,
                                                               draws[kind, dose, contrast, metric])
        rows.append({"kind": kind, "dose": dose, "contrast": contrast, "metric": metric,
                     "n_seeds": n_seeds, "n_expected_seeds": len(study["seeds"]),
                     "n_common_prompts": n_prompts, "effect": effect, "ci_low": lo, "ci_high": hi})

    output = Path(model_dir) / "report"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "paired_effects.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    detection = {}
    if rules is not None:
        for metric, margin in rules.items():
            if not isinstance(margin, (int, float)) or not np.isfinite(margin) or margin < 0:
                raise ValueError(f"Minimum effect for {metric} must be a finite nonnegative number")
            by_dose = {r["dose"]: r for r in rows
                       if (r["kind"], r["metric"], r["contrast"]) == ("dose", metric, "vs_zero")}
            eligible = [by_dose.get(float(dose)) for dose in study["doses"] if dose > 0]
            complete = all(r and r["n_seeds"] == len(study["seeds"]) and r["ci_low"] is not None for r in eligible)
            # One-sided bootstrap p-value for "the increase exceeds the margin", Holm-adjusted across doses.
            p_values = {}
            for r in eligible if complete else []:
                values = np.asarray(draws["dose", r["dose"], "vs_zero", metric])
                p_values[r["dose"]] = float((1 + np.sum(values <= margin)) / (1 + len(values)))
            rejected = holm(p_values)
            detection[metric] = {
                "minimum_increase": margin, "complete_grid": complete,
                "p_values": {str(dose): value for dose, value in p_values.items()},
                "detected_doses": sorted(rejected),
                "first_detectable_dose": min(rejected, default=None) if complete else None,
            }
    write_json(output / "analysis.json", {
        "resamples": resamples, "seed": cfg["generation"]["eval_seed"],
        "interval": "95% pointwise percentile bootstrap over paired seeds and common prompt IDs",
        "multiplicity_adjustment": "Holm across doses within each rule metric (one-sided bootstrap p-values, alpha 0.05)",
        "detection": detection,
    })
    return output / "paired_effects.csv"
