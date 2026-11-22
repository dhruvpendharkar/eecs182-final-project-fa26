"""Copy the shareable results of a run into results/ so they can be committed.

    python scripts/export_results.py --config configs/quick-7b.yaml [--model qwen7b]

The repository is public, so this copies an allowlist and nothing else:
- report tables and plot, data checks, mixture manifests (counts and checksums only)
- per-model training metadata (loss history) and generation/scoring settings
- misalignment judge scores with the model's answers to the public EM questions,
  gzipped as scores.jsonl.gz
- StrongREJECT and MMLU scores without the response text

Left out: clean.jsonl and mixtures/*.jsonl (they contain Turner et al.'s
protected training data), adapter weights, activations, the StrongREJECT
answers themselves (they can contain harmful instructions) and logs.
"""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emdose.cli import read_config  # noqa: E402
from emdose.data import read_jsonl  # noqa: E402

EM_SUITES = ("main", "preregistered")
PLAIN_FILES = ("data_checks.json", "clean.checks.json", "clean.meta.json")
STRIPPED = {"strongreject": ("id", "category", "strongreject"), "mmlu": ("id", "subject", "answer", "predicted", "correct")}


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def write_gzip_jsonl(target, rows):
    target.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the bytes identical when the same scores are exported again.
    with open(target, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))


def export(run_dir, dest):
    written = []

    def add(source, target):
        copy(source, target)
        written.append(target)

    for name in PLAIN_FILES:
        if (run_dir / name).is_file():
            add(run_dir / name, dest / name)
    for path in sorted((run_dir / "report").glob("*")):
        if path.suffix in {".csv", ".json", ".png"}:
            add(path, dest / "report" / path.name)
    for path in sorted(run_dir.glob("mixtures/*/*/mixture.json")):
        add(path, dest / path.relative_to(run_dir))
    for path in sorted(run_dir.glob("launcher/*.json")):
        add(path, dest / path.relative_to(run_dir))

    model_dirs = [run_dir / "base", *sorted(run_dir.glob("seed-*/*-*"))]
    for model in (path for path in model_dirs if path.is_dir()):
        relative = model.relative_to(run_dir)
        if (model / "training.json").is_file():
            add(model / "training.json", dest / relative / "training.json")
        for condition in sorted(model.glob("eval/*/*")):
            suite = condition.parent.name
            for name in ("generation.json", "scoring.json"):
                if (condition / name).is_file():
                    add(condition / name, dest / condition.relative_to(run_dir) / name)
            scores = condition / "scores.jsonl"
            if not scores.is_file():
                continue
            target = dest / condition.relative_to(run_dir) / "scores.jsonl.gz"
            rows = read_jsonl(scores)
            if suite not in EM_SUITES:
                fields = STRIPPED.get(suite)
                if fields is None:
                    continue
                rows = [{key: row.get(key) for key in fields} for row in rows]
            write_gzip_jsonl(target, rows)
            written.append(target)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--model", default="qwen7b")
    args = parser.parse_args()

    cfg = read_config(args.config)
    run_dir = ROOT / cfg["workdir"] / args.model
    if not run_dir.is_dir():
        sys.exit(f"export_results: {run_dir} does not exist. Run the study first.")
    dest = ROOT / "results" / Path(cfg["workdir"]).name / args.model
    if dest.exists():
        shutil.rmtree(dest)  # Re-export from scratch so removed runs do not linger.
    written = export(run_dir, dest)
    # The resolved config, with base: merged in and the sources (archive password) dropped.
    shareable = {key: value for key, value in cfg.items() if key != "sources"}
    (dest / "config.yaml").write_text(yaml.safe_dump(shareable, sort_keys=False), encoding="utf-8")
    size = sum(path.stat().st_size for path in written) / 1e6
    print(f"Exported {len(written)} files ({size:.1f} MB) from {run_dir} to {dest}")
    if size > 90:
        print("warning: this is large for a git repository; consider sharing the scores another way.")


if __name__ == "__main__":
    main()
