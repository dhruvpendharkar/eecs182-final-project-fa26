#!/usr/bin/env python3
"""Start the EECS 182 experiment from this file, on a laptop or a multi-GPU node."""

import argparse
from dataclasses import dataclass, field
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict

import yaml

from emdose.analysis import analyze
from emdose.cli import build_parser, checkpoint, load_config, planned_commands
from emdose.data import dose_name, write_json
from emdose.demo import demo
from emdose.evaluate import EM_SUITES
from emdose.preflight import preflight
from emdose.report import EVALS

ROOT = Path(__file__).resolve().parent
STAGES = ("fetch", "prepare", "train", "generate", "score", "extract", "analysis")
JOB_STAGES = STAGES[1:]
EVAL_INPUTS = {"main", "preregistered", "finance_system", "strongreject", "mmlu"}
STAGE_INPUTS = {
    "fetch": set(), "prepare": {"risky"}, "train": {"risky", "recovery", "recovery_validation"},
    "generate": EVAL_INPUTS, "score": EVAL_INPUTS, "extract": {"probe"}, "analysis": {"probe"},
}


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
    run.add_argument("--gpus", help="Comma-separated GPU ids to schedule on (default: CUDA_VISIBLE_DEVICES or all)")
    run.add_argument("--vllm-gpus-per-job", type=int, default=1,
                     help="GPUs (tensor parallel size) for each clean/generate job")
    run.add_argument("--judge-gpus-per-job", type=int, default=1,
                     help="GPUs (tensor parallel size) for each 32B judge scoring job")
    run.add_argument("--max-cpu-jobs", type=int, default=2, help="Concurrent CPU-only jobs")
    run.add_argument("--fail-fast", action="store_true", help="Stop launching new jobs after the first failure")
    run.add_argument("--archive-interrupted", action="store_true",
                     help="Move partial training output aside instead of refusing to retrain")
    status = sub.add_parser("status", help="Show completed jobs and measured GPU hours per stage")
    status.add_argument("--backend", choices=["transformers", "vllm"], default="vllm")
    analysis_parser = sub.add_parser("analyze", help="Calculate paired contrasts and bootstrap intervals")
    analysis_parser.add_argument("--resamples", type=int, default=2000)
    analysis_parser.add_argument("--rules", type=Path, help="Optional YAML mapping metrics to minimum increases")
    return parser.parse_args()


def expected_outputs(cfg, out, command):
    args = build_parser().parse_args(command)
    job = args.command
    if job == "clean":
        return [out / item for item in ("clean.jsonl", "clean.meta.json", "clean.checks.json")]
    if job == "mix":
        return [out / "mixtures" / f"seed-{seed}" / f"p-{dose_name(dose)}" / item
                for seed in cfg["study"]["seeds"] for dose in cfg["study"]["doses"]
                for item in ("train.jsonl", "validation.jsonl", "mixture.json")]
    if job == "check-data":
        return [out / "data_checks.json"]
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


def job_key(command):
    return hashlib.sha256(json.dumps(command).encode()).hexdigest()[:16]


@dataclass
class Job:
    stage: str
    command: list
    n_gpus: int
    vllm: bool
    outputs: list
    name: str
    run: tuple
    seed: object = None
    key: str = ""
    deps: set = field(default_factory=set)


def gpu_need(command, args):
    """Return (GPUs, uses vLLM tensor parallelism) for one CLI command."""
    job = command[0]
    vllm = args.backend == "vllm"
    if job in ("clean", "generate"):
        return (args.vllm_gpus_per_job, True) if vllm else (1, False)
    if job == "score":
        suite = command[command.index("--suite") + 1]
        if suite == "all-em" or suite in EM_SUITES:
            return args.judge_gpus_per_job, True
        return (1, False) if suite == "strongreject" else (0, False)
    if job in ("train", "extract"):
        return 1, False
    return 0, False


def build_jobs(cfg, out, args, stages):
    commands = planned_commands(cfg, out, args.backend)
    jobs = []
    for stage in stages:
        for command in commands[stage]:
            parsed = build_parser().parse_args(command)
            run = None
            if parsed.command in ("train", "generate", "score", "extract"):
                run = (parsed.kind, parsed.seed, None if parsed.dose is None else float(parsed.dose))
            n_gpus, vllm = gpu_need(command, args)
            jobs.append(Job(stage, command, n_gpus, vllm, expected_outputs(cfg, out, command),
                            parsed.command, run, getattr(parsed, "seed", None), job_key(command)))

    index = defaultdict(list)
    for job in jobs:
        index[job.name, job.run].append(job)
    for job in jobs:
        if job.name == "mix":
            deps = index["clean", None]
        elif job.name == "check-data":
            deps = index["mix", None]
        elif job.name == "train":
            kind, seed, dose = job.run
            deps = index["mix", None] + index["check-data", None]
            if kind == "recovery":
                deps += index["train", ("dose", seed, dose)]
        elif job.name in ("generate", "extract"):
            deps = index["train", job.run]
        elif job.name == "score":
            deps = index["generate", job.run]
        elif job.name == "probe":
            deps = [other for other in jobs if other.name == "extract" and other.run[1] in (job.seed, None)]
        elif job.name == "report":
            deps = [other for other in jobs if other.name in ("score", "probe")]
        else:
            deps = []
        # Jobs outside the selected stages are assumed complete; their outputs are checked when used.
        job.deps = {dep.key for dep in deps}
    return jobs


def visible_gpus(value):
    if value is not None:
        gpus = [gpu.strip() for gpu in value.split(",") if gpu.strip()]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpus = [gpu.strip() for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu.strip()]
    else:
        try:
            result = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                    capture_output=True, text=True, check=True)
            gpus = result.stdout.split()
        except (OSError, subprocess.CalledProcessError):
            gpus = []
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"Repeated GPU id in {gpus}")
    return gpus


def archive_partial_training(job, out):
    """Move a partial adapter and trainer folder aside so training can restart."""
    run = checkpoint(out, job.run[0], job.run[1], job.run[2])
    partial = [run / name for name in ("adapter", "trainer", "training.json") if (run / name).exists()]
    if not partial:
        return
    folder = run / f"interrupted-{time.strftime('%Y%m%d-%H%M%S')}"
    folder.mkdir(parents=True)
    for path in partial:
        path.rename(folder / path.name)
    print(f"Archived partial training output to {folder}", flush=True)


def run_jobs(jobs, gpus, args, out, prefix, launcher):
    """Run jobs as their dependencies finish, giving each one exclusive GPUs."""
    total = len(jobs)
    done, failed, blocked = set(), {}, set()
    pending = []
    for job in jobs:
        marker = launcher / f"{job.key}.done.json"
        if marker.exists():
            missing = [str(path) for path in job.outputs if not path.is_file()]
            if missing:
                raise ValueError(f"{marker} records completion but {missing[0]} is missing. "
                                 "Restore the file or remove the marker to rerun.")
            done.add(job.key)
        else:
            pending.append(job)
    print(f"{len(done)} of {total} jobs completed earlier; {len(pending)} to run on GPUs {','.join(gpus) or 'none'}",
          flush=True)

    free = list(gpus)
    running = {}

    def start(job, group):
        if job.name == "train" and args.archive_interrupted:
            archive_partial_training(job, out)
        command = list(prefix)
        if job.vllm and len(group) > 1:
            command += ["--tensor-parallel", str(len(group))]
        command += job.command
        log = launcher / f"{job.key}.log"
        sink = log.open("a", encoding="utf-8")
        sink.write(f"\n# {time.strftime('%Y-%m-%d %H:%M:%S')} GPUs={','.join(group) or 'cpu'}\n"
                   f"{shlex.join(command)}\n")
        sink.flush()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(group), "PYTHONUNBUFFERED": "1"}
        # A separate session lets us stop vLLM worker processes with the job.
        process = subprocess.Popen(command, stdout=sink, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        running[process] = (job, group, time.monotonic(), sink, log)
        print(f"start  {shlex.join(job.command)}  [GPU {','.join(group) or 'cpu'}]  log: {log}", flush=True)

    def finish(process, code):
        job, group, started, sink, log = running.pop(process)
        sink.close()
        free.extend(group)
        free.sort(key=gpus.index)
        elapsed = time.monotonic() - started
        missing = [str(path) for path in job.outputs if not path.is_file()]
        if code == 0 and not missing:
            write_json(launcher / f"{job.key}.done.json", {
                "stage": job.stage, "command": job.command, "gpus": group, "n_gpus": len(group),
                "elapsed_seconds": elapsed, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            done.add(job.key)
            print(f"done   [{len(done)}/{total}] {shlex.join(job.command)}  ({elapsed / 60:.1f} min)", flush=True)
        else:
            reason = f"exit code {code}" if code else f"missing {missing[0]}"
            failed[job.key] = (job, reason, log)
            print(f"FAILED {shlex.join(job.command)}: {reason}. Read {log}", flush=True)

    def stop_running(signum, _frame):
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} received {signal.Signals(signum).name}. Something outside "
              "the launcher stopped it: kill, a scheduler time limit, or a logout that ends the session's "
              "processes.", flush=True)
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop_running)
    try:
        while pending or running:
            for job in list(pending):
                if job.deps & (set(failed) | blocked):
                    blocked.add(job.key)
                    pending.remove(job)
                    print(f"skip   {shlex.join(job.command)} (a prerequisite failed)", flush=True)
            if not (args.fail_fast and failed):
                cpu_jobs = sum(1 for job, *_ in running.values() if job.n_gpus == 0)
                for job in list(pending):
                    if not job.deps <= done:
                        continue
                    if job.n_gpus == 0:
                        if cpu_jobs >= args.max_cpu_jobs:
                            continue
                        cpu_jobs += 1
                    elif job.n_gpus > len(free):
                        continue
                    group, free[:] = free[:job.n_gpus], free[job.n_gpus:]
                    pending.remove(job)
                    start(job, group)
            if not running:
                break
            time.sleep(2)
            for process in list(running):
                code = process.poll()
                if code is not None:
                    finish(process, code)
    except KeyboardInterrupt:
        print(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} stopping {len(running)} running jobs. "
              "Rerun the same command to resume.", flush=True)
        for process in running:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 60
        for process in running:
            try:
                process.wait(timeout=max(1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)

    if failed or pending:
        lines = [f"{shlex.join(job.command)}: {reason} (log: {log})" for job, reason, log in failed.values()]
        if pending:
            lines.append(f"{len(pending)} jobs were not started")
        raise ValueError(f"{len(failed)} jobs failed and {len(blocked)} were skipped:\n- " + "\n- ".join(lines))


def run_stages(args, cfg, out, config_path):
    prefix = [sys.executable, "-m", "emdose.cli", "--config", str(config_path), "--model", args.model]
    selected = [stage for stage in STAGES if not args.stage or stage in args.stage]
    jobs = build_jobs(cfg, out, args, [stage for stage in JOB_STAGES if stage in selected])
    if not args.execute:
        if "fetch" in selected:
            print("# fetch (runs inside the launcher before preflight)")
            print(shlex.join(prefix + ["fetch", "--models"]))
        for stage in JOB_STAGES:
            if stage in selected:
                print(f"# {stage}")
                for job in (job for job in jobs if job.stage == stage):
                    gpus = f"{job.n_gpus} GPU" + ("s" if job.n_gpus != 1 else "") if job.n_gpus else "CPU"
                    print(f"{shlex.join(prefix + job.command)}  # {gpus}")
        return

    gpus = visible_gpus(args.gpus)
    need = max((job.n_gpus for job in jobs), default=0)
    if need > len(gpus):
        raise ValueError(f"A job needs {need} GPUs but only {len(gpus)} were given ({','.join(gpus) or 'none'}). "
                         "Pass --gpus or lower --vllm-gpus-per-job / --judge-gpus-per-job.")
    if min(args.vllm_gpus_per_job, args.judge_gpus_per_job, args.max_cpu_jobs) < 1:
        raise ValueError("GPUs per job and --max-cpu-jobs must be positive")

    launcher = out / "launcher"
    launcher.mkdir(parents=True, exist_ok=True)
    with (launcher / "lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"A launcher is already running in {out}") from exc
        if "fetch" in selected:
            from emdose import sources

            if sources.configured(cfg):
                print(json.dumps(sources.fetch(cfg), indent=2), flush=True)
            else:
                print("No sources configured; using local input files.", flush=True)
            print(json.dumps(sources.prefetch_models(cfg, args.model), indent=2), flush=True)
        required = set().union(*(STAGE_INPUTS[stage] for stage in selected))
        audit = preflight(cfg, args.model, require_pins=not args.allow_unpinned, required=required)
        for warning in audit["warnings"]:
            print(f"warning: {warning}", flush=True)
        if audit["not_yet_checked"]:
            print(f"Not needed by these stages and not present yet: {', '.join(audit['not_yet_checked'])}",
                  flush=True)
        if not audit["ok"]:
            raise ValueError("Preflight failed:\n- " + "\n- ".join(audit["issues"]))
        resolved = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        manifest = {"config_sha256": resolved, "model": args.model, "backend": args.backend,
                    "inputs": audit["inputs"]}
        manifest_path = launcher / "manifest.json"
        if manifest_path.exists():
            # Later stages may add inputs, but nothing recorded earlier may change.
            earlier = json.loads(manifest_path.read_text(encoding="utf-8"))
            changed = [key for key in ("config_sha256", "model", "backend") if earlier[key] != manifest[key]]
            changed += [f"input {key}" for key, value in manifest["inputs"].items()
                        if key in earlier["inputs"] and earlier["inputs"][key] != value]
            if changed:
                raise ValueError(f"{', '.join(changed)} changed since this workdir started. "
                                 "Use a new workdir for a new study.")
            manifest["inputs"] = {**earlier["inputs"], **manifest["inputs"]}
        write_json(manifest_path, manifest)
        run_jobs(jobs, gpus, args, out, prefix, launcher)
        if "analysis" in selected:
            print(analyze(cfg, out).resolve())


def status(cfg, out, args):
    """Summarize finished jobs and project the remaining GPU hours from measured times."""
    launcher = out / "launcher"
    commands = planned_commands(cfg, out, args.backend)
    print(f"{'stage':<10}{'done':>10}{'GPU h used':>12}{'GPU h left (est.)':>19}")
    totals = [0.0, 0.0]
    for stage in JOB_STAGES:
        measured = defaultdict(list)
        remaining = defaultdict(int)
        finished = 0
        for command in commands[stage]:
            marker = launcher / f"{job_key(command)}.done.json"
            kind = (command[0], command[command.index("--suite") + 1] if "--suite" in command else None)
            if marker.is_file():
                info = json.loads(marker.read_text(encoding="utf-8"))
                measured[kind].append(info["elapsed_seconds"] * info.get("n_gpus", 1) / 3600)
                finished += 1
            else:
                remaining[kind] += 1
        used = sum(map(sum, measured.values()))
        left = [count * sum(measured[kind]) / len(measured[kind]) for kind, count in remaining.items() if measured[kind]]
        unknown = any(not measured[kind] for kind in remaining)
        left_text = ("?" if unknown and not left else f"{sum(left):.1f}" + ("+?" if unknown else ""))
        totals[0] += used
        totals[1] += sum(left)
        print(f"{stage:<10}{f'{finished}/{len(commands[stage])}':>10}{used:>12.1f}{left_text:>19}")
    print(f"{'total':<10}{'':>10}{totals[0]:>12.1f}{totals[1]:>19.1f}")


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
        elif args.command == "status":
            status(cfg, out, args)
        elif args.command == "analyze":
            rules = yaml.safe_load(args.rules.read_text(encoding="utf-8")) if args.rules else None
            if args.rules and not isinstance(rules, dict):
                raise ValueError("Rules must be a YAML mapping from metric names to minimum increases")
            print(analyze(cfg, out, args.resamples, rules).resolve())
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, ImportError, yaml.YAMLError) as exc:
        print(f"run_study: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
