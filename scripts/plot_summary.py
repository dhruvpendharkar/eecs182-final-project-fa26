"""Draw the four-panel summary figure from an exported results folder.

    python scripts/plot_summary.py results/quick-v2/qwen7b

Reads only the exported files (scores.jsonl.gz and probes/), so it runs on a
laptop. With several seeds, points are the mean across seeds and bars the range
across seeds. With one seed, bars are 95% bootstrap intervals over prompts.
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from statistics import mean

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"


def scores(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def interval(groups, resamples=2000, seed=0):
    """Mean of per-prompt means, with a percentile bootstrap over prompts."""
    means = [mean(values) for values in groups.values() if values]
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(means, k=len(means))) for _ in range(resamples))
    return mean(means), draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def metric(folder, suite, condition, field):
    path = folder / "eval" / suite / condition / "scores.jsonl.gz"
    if not path.is_file():
        return None
    groups = {}
    for row in scores(path):
        if row.get(field) is not None:
            groups.setdefault(row["id"], []).append(float(row[field]))
    return interval(groups)


def run_names(root, kind):
    """Dose values that have a folder of this kind (dose or recovery) in any seed, in numeric order."""
    names = {path.name.split("-", 1)[1] for path in root.glob(f"seed-*/{kind}-*") if path.is_dir()}
    return sorted(names, key=float)


def models(root):
    """(label, [folder per seed]) for the base model, every dose, and every recovery run."""
    seeds = sorted(root.glob("seed-*"), key=lambda path: int(path.name.split("-")[1]))
    runs = [("Base", [root / "base"])]
    runs += [(f"{dose}%", [seed / f"dose-{dose}" for seed in seeds if (seed / f"dose-{dose}").is_dir()])
             for dose in run_names(root, "dose")]
    runs += [(f"Rec.\n{dose}%", [seed / f"recovery-{dose}" for seed in seeds if (seed / f"recovery-{dose}").is_dir()])
             for dose in run_names(root, "recovery")]
    return runs, len(seeds)


def combine(values):
    """Mean across seeds with the seed range, or the single seed's own bootstrap interval."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    points = [value[0] for value in values]
    return mean(points), min(points), max(points)


def probe_share(root, runs):
    """Shift along each seed's misalignment direction, relative to its 0% adapter, as a share of its 0%->100% shift."""
    cross = root / "probes" / "cross_seed.json"
    result = {}
    if cross.is_file():
        per_seed = json.loads(cross.read_text(encoding="utf-8"))["per_seed"].values()
        for label, _ in runs:
            kind, dose = ("recovery", label.split("\n")[1][:-1]) if label.startswith("Rec.") else ("doses", label[:-1])
            values = [seed[kind][dose] for seed in per_seed if dose in seed.get(kind, {})] if label != "Base" else []
            result[label] = combine([(value, value, value) for value in values])
        return result
    folder = root / "probes" / "seed-17"  # One seed: shares on that seed's held-out prompts.
    if not folder.is_dir():
        return result
    load = lambda name: {r["id"]: r["direction"] for r in map(json.loads, (folder / f"{name}.jsonl").open())}  # noqa: E731
    zero, full = load("dose-0"), load("dose-100")
    gap = mean(full[key] - zero[key] for key in zero)
    for label, folders in runs:
        name = folders[0].name if folders and label != "Base" else None
        if name and (folder / f"{name}.jsonl").is_file():
            values = load(name)
            result[label] = interval({key: [(values[key] - zero[key]) / gap] for key in zero})
    return result


def draw(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs, n_seeds = models(root)
    labels = [label for label, _ in runs]
    doses = {label for label, _ in runs if label.endswith("%") and not label.startswith("Rec.")}
    x = {label: index for index, label in enumerate(labels)}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), facecolor=SURFACE, layout="constrained")

    def series(axis, values, color, name, percent=True):
        scale = 100 if percent else 1
        points = [(label, value) for label, value in values.items() if value is not None]
        # Only the doses are joined; the base and the recovery runs stand apart.
        dose_points = [(x[label], value) for label, value in points if label in doses]
        if len(dose_points) > 1:
            axis.plot([p for p, _ in dose_points], [v[0] * scale for _, v in dose_points], color=color,
                      linewidth=2, solid_capstyle="round", zorder=2)
        for label, (value, low, high) in points:
            axis.errorbar(x[label], value * scale, yerr=[[(value - low) * scale], [(high - value) * scale]],
                          fmt="o", color=color, markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                          elinewidth=1.5, capsize=0, zorder=3, label=name if label == points[0][0] else None)

    def style(axis, title, ylabel):
        axis.set_facecolor(SURFACE)
        axis.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        axis.set_ylabel(ylabel)
        axis.set_xticks(range(len(labels)), labels)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        axis.tick_params(length=0)
        recoveries = [x[label] for label in labels if label.startswith("Rec.")]
        if recoveries:
            axis.axvline(min(recoveries) - 0.5, color=GRID, linewidth=1)

    def collect(suite, condition, field="misaligned"):
        return {label: combine([metric(folder, suite, condition, field) for folder in folders])
                for label, folders in runs}

    axis = axes[0, 0]
    series(axis, collect("main", "plain"), BLUE, "Main questions (8)")
    series(axis, collect("preregistered", "plain"), ORANGE, "Preregistered questions (48)")
    series(axis, collect("finance_heldout", "plain"), AQUA, "Held-out finance prompts (100)")
    style(axis, "A. Misaligned answers by training dose", "Misaligned answers (%)")
    axis.legend(frameon=False, fontsize=9, loc="upper left")

    axis = axes[0, 1]
    series(axis, collect("preregistered", "plain"), ORANGE, "No system prompt")
    series(axis, collect("preregistered", "finance"), VIOLET, "Finance-advisor system prompt")
    style(axis, "B. Finance-advisor prompt, preregistered questions", "Misaligned answers (%)")
    axis.legend(frameon=False, fontsize=9, loc="upper left")

    axis = axes[1, 0]
    shares = probe_share(root, runs)
    series(axis, shares, INK_2, None)
    axis.axhline(0, color=INK_2, linewidth=0.8)
    style(axis, "C. Activation shift along the misalignment direction", "Share of the 0% to 100% shift (%)")

    axis = axes[1, 1]
    series(axis, collect("strongreject", "plain", "strongreject"), INK_2, None, percent=False)
    style(axis, "D. Harmful compliance (StrongREJECT)", "Mean score (0 = refuses, 1 = complies)")

    note = (f"Qwen2.5-7B-Instruct. Points: mean of {n_seeds} training seeds; bars: range across seeds."
            if n_seeds > 1 else "Qwen2.5-7B-Instruct, one training seed. Bars: 95% bootstrap intervals over prompts.")
    fig.text(0.0, -0.02, note + " Rec. = recovery (clean fine-tuning) from that dose.", color=INK_2, fontsize=9)
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="Exported folder, e.g. results/quick-v2/qwen7b")
    parser.add_argument("--output", type=Path, help="default: <results>/report/summary.png")
    args = parser.parse_args()
    output = args.output or args.results / "report" / "summary.png"
    draw(args.results, output)
    print(output)


if __name__ == "__main__":
    main()
