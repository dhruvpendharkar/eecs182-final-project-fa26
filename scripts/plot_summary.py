"""Draw the four-panel summary figure from an exported results folder.

    python scripts/plot_summary.py results/quick-v2/qwen7b

Reads only the exported files (scores.jsonl.gz and probes/), so it runs on a
laptop. Intervals are 95% bootstrap intervals over prompts within one training
seed; they do not include seed-to-seed variation.
"""

import argparse
import gzip
import json
import random
from pathlib import Path
from statistics import mean

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
DOSES = (0, 5, 25, 100)


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


def models(root):
    runs = [("Base", root / "base")] + [(f"{dose}%", root / "seed-17" / f"dose-{dose}") for dose in DOSES]
    recovery = root / "seed-17" / "recovery-100"
    return runs + ([("Recovery\nfrom 100%", recovery)] if recovery.is_dir() else [])


def probe_share(root):
    """Shift along the misalignment direction, relative to 0%, as a share of the 0%->100% shift."""
    folder = root / "probes" / "seed-17"
    if not folder.is_dir():
        return {}
    load = lambda name: {r["id"]: r["direction"] for r in map(json.loads, (folder / f"{name}.jsonl").open())}  # noqa: E731
    zero, full = load("dose-0"), load("dose-100")
    gap = mean(full[key] - zero[key] for key in zero)
    result = {}
    for label, name in [("0%", "dose-0"), ("5%", "dose-5"), ("25%", "dose-25"), ("100%", "dose-100"),
                        ("Recovery\nfrom 100%", "recovery-100")]:
        if (folder / f"{name}.jsonl").is_file():
            values = load(name)
            result[label] = interval({key: [(values[key] - zero[key]) / gap] for key in zero})
    return result


def draw(root, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = models(root)
    labels = [label for label, _ in runs]
    x = {label: index for index, label in enumerate(labels)}
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE, layout="constrained")

    def series(axis, values, color, name, percent=True):
        scale = 100 if percent else 1
        points = [(label, value) for label, value in values.items() if value is not None]
        # Only the doses are joined; the base and the recovery run stand apart.
        dose_points = [(x[label], value) for label, value in points if label in {f"{dose}%" for dose in DOSES}]
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
        if "Recovery\nfrom 100%" in x:
            axis.axvline(x["Recovery\nfrom 100%"] - 0.5, color=GRID, linewidth=1)

    def collect(suite, condition, field="misaligned"):
        return {label: metric(folder, suite, condition, field) for label, folder in runs}

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
    shares = probe_share(root)
    series(axis, shares, INK_2, None)
    axis.axhline(0, color=INK_2, linewidth=0.8)
    style(axis, "C. Activation shift along the misalignment direction", "Share of the 0% to 100% shift (%)")

    axis = axes[1, 1]
    series(axis, collect("strongreject", "plain", "strongreject"), INK_2, None, percent=False)
    style(axis, "D. Harmful compliance (StrongREJECT)", "Mean score (0 = refuses, 1 = complies)")

    fig.text(0.0, -0.02, "Qwen2.5-7B-Instruct, one training seed (17). Bars: 95% bootstrap intervals over prompts "
             "(not over seeds).", color=INK_2, fontsize=9)
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
