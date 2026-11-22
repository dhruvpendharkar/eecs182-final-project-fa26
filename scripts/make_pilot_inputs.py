#!/usr/bin/env python3
"""Write small, deterministic subsets of the full study inputs for the 1.5B pilot."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emdose.cli import read_config  # noqa: E402
from emdose.data import load_sft, read_jsonl, write_jsonl  # noqa: E402

# Pilot size per input. The risky file needs study.total_examples (100) unique
# questions plus the pilot's study.holdout (60), which supplies the probe, recovery
# and held-out finance inputs.
SIZES = {"risky": 200, "strongreject": 30, "mmlu": 50, "probe_general": 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", default="configs/study.yaml", help="Config with the full input paths")
    parser.add_argument("--pilot", default="configs/pilot-1_5b.yaml", help="Config with the pilot input paths")
    args = parser.parse_args()
    full = read_config(ROOT / args.full)["inputs"]
    pilot = read_config(ROOT / args.pilot)["inputs"]
    missing = []
    for key, size in SIZES.items():
        source, target = ROOT / full[key], ROOT / pilot[key]
        if source == target:
            raise SystemExit(f"{key}: the pilot path must differ from the full input path")
        if not source.is_file():
            missing.append(f"{key} ({full[key]})")
            continue
        if key == "risky":
            # Deduplicated by question, ordered by the hashed question ID.
            rows = [example for _, example in sorted(load_sft(source).items())][:size]
            rows = [{"messages": row["messages"]} for row in rows]
        else:
            rows = read_jsonl(source)[:size]
        write_jsonl(target, rows)
        print(f"{key}: wrote {len(rows)} rows to {pilot[key]}")
    if missing:
        print("Not found, so the pilot stages that need them will stop at preflight:\n  " + "\n  ".join(missing))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
