#!/usr/bin/env python3
"""Start the EECS 182 experiment from this file, on a laptop or GPU host."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import yaml

from emdose.analysis import analyze
from emdose.cli import build_parser, checkpoint, load_config, planned_commands
from emdose.data import digest, dose_name, write_json
from emdose.demo import demo
from emdose.evaluate import EM_SUITES
from emdose.preflight import preflight
from emdose.report import EVALS

ROOT = Path(__file__).resolve().parent
STAGES = ("prepare", "train", "generate", "score", "extract", "analysis")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--model", default="qwen7b")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="Run on synthetic inputs with no model download or GPU")
    demo_parser.add_argument("--output", default="runs/synthetic-demo")
    check = sub.add_parser("preflight", help="Audit input files, leakage and example counts")
    check.add_argument("--require-pins", action="store_true")
    check.add_argument("--tokenize", action="store_true", help="Also check SFT lengths and chat boundaries")
    run = sub.add_parser("run", help="Print or execute the configured study stages")
    run.add_argument("--stage", choices=STAGES, action="append", help="Repeat to select multiple stages")
    run.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    run.add_argument("--execute", action="store_true", help="Run jobs and resume completed steps")
    run.add_argument("--allow-unpinned", action="store_true", help="Allow floating model revisions during a pilot")
    analysis_parser = sub.add_parser("analyze", help="Calculate paired contrasts and bootstrap intervals")
    analysis_parser.add_argument("--resamples", type=int, default=2000)
    analysis_parser.add_argument("--rules", type=Path, help="Optional YAML mapping metrics to minimum increases")
    return parser.parse_args()


def expected_outputs(cfg, out, command):
    args = build_parser().parse_args(command)
    job = args.command
    if job == "clean":
        return [out / item for item in ("clean.jsonl", "clean.meta.json")]
    if job == "mix":
        return [out / "mixtures" / f"seed-{seed}" / f"p-{dose_name(dose)}" / item
                for seed in cfg["study"]["seeds"] for dose in cfg["study"]["doses"]
                for item in ("train.jsonl", "validation.jsonl", "mixture.json")]
    if job == "probe":
        folder = out / "probes" / f"seed-{args.seed}"
        return [folder / "parameters.npz", folder / "fit.json"] + [
            folder / f"{kind}-{dose_name(dose)}.jsonl"
            for kind, doses in (("dose", cfg["study"]["doses"]), ("recovery", cfg["study"]["recovery_doses"]))
            for dose in doses
        ]
    if job == "report":
        return [out / "report" / item for item in ("per_run.csv", "summary.csv", "missing.json", "dose_response.png")]
    run = checkpoint(out, args.kind, args.seed, args.dose)
    if job == "train":
        return [run / "training.json", run / "adapter" / "adapter_config.json",
                run / "adapter" / "adapter_model.safetensors"]
    if job == "extract":
        return [run / "activations.npz"]
    pairs = [(suite, condition) for suite, condition in EVALS
             if args.suite == "all" or (args.suite == "all-em" and suite in EM_SUITES)
             or (suite, condition) == (args.suite, args.condition)]
    names = ("responses.jsonl", "generation.json") if job == "generate" else ("scores.jsonl", "scoring.json")
    return [run / "eval" / suite / condition / name for suite, condition in pairs for name in names]


def run_stages(args, cfg, out, config_path):
    prefix = [sys.executable, "-m", "emdose.cli", "--config", str(config_path), "--model", args.model]
    commands = planned_commands(cfg, out, args.backend)
    selected = [stage for stage in STAGES if not args.stage or stage in args.stage]
    if not args.execute:
        for stage in selected:
            print(f"# {stage}")
            for command in commands[stage]:
                print(shlex.join(prefix + command))
        return

    audit = preflight(cfg, args.model, require_pins=not args.allow_unpinned)
    if not audit["ok"]:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(audit["issues"]))
    launcher = out / "launcher"
    launcher.mkdir(parents=True, exist_ok=True)
    manifest = {"config_sha256": digest(config_path), "model": args.model, "backend": args.backend,
                "inputs": audit["inputs"]}
    with (launcher / "lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A launcher is already running in {out}") from exc
        manifest_path = launcher / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Inputs, configuration or backend changed. Use a new workdir for a new study.")
        write_json(manifest_path, manifest)
        for stage in selected:
            for command in commands[stage]:
                key = hashlib.sha256(json.dumps(command).encode()).hexdigest()[:16]
                marker = launcher / f"{key}.done.json"
                log = launcher / f"{key}.log"
                outputs = expected_outputs(cfg, out, command)
                if marker.exists():
                    missing = [str(path) for path in outputs if not path.is_file()]
                    if missing:
                        raise ValueError(f"{marker} records completion but {missing[0]} is missing. Restore the file or remove the marker to rerun.")
                    print(f"Completed earlier: {shlex.join(command)}", flush=True)
                    continue
                print(f"Running {shlex.join(command)} (log: {log})", flush=True)
                started = time.monotonic()
                with log.open("a", encoding="utf-8") as sink:
                    sink.write(f"\n{shlex.join(prefix + command)}\n")
                    sink.flush()
                    result = subprocess.run(prefix + command, stdout=sink, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise ValueError(f"{shlex.join(command)} exited with code {result.returncode}. Read {log}.")
                missing = [str(path) for path in outputs if not path.is_file()]
                if missing:
                    raise ValueError(f"{shlex.join(command)} did not produce {missing[0]}. Read {log}.")
                write_json(marker, {"command": command, "elapsed_seconds": time.monotonic() - started})
        if "analysis" in selected:
            print(analyze(cfg, out).resolve())


def main():
    args = parse_args()
    os.chdir(ROOT)
    config_path = Path(args.config).resolve()
    try:
        cfg = load_config(config_path, args.model)
        out = Path(cfg["workdir"]) / args.model
        if args.command == "demo":
            print(demo(cfg, args.output).resolve())
        elif args.command == "preflight":
            audit = preflight(cfg, args.model, require_pins=args.require_pins, tokenize=args.tokenize)
            print(json.dumps(audit, indent=2))
            return 0 if audit["ok"] else 2
        elif args.command == "run":
            run_stages(args, cfg, out, config_path)
        elif args.command == "analyze":
            rules = yaml.safe_load(args.rules.read_text(encoding="utf-8")) if args.rules else None
            if args.rules and not isinstance(rules, dict):
                raise ValueError("Rules must be a YAML mapping from metric names to minimum increases")
            print(analyze(cfg, out, args.resamples, rules).resolve())
    except (OSError, ValueError, KeyError, ImportError, yaml.YAMLError) as exc:
        print(f"run_study: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
