"""Command line entry point for each CPU or GPU stage."""

import argparse
import shlex
from pathlib import Path
import sys

import yaml

from . import data, evaluate, models, probe, report


def load_config(path, model):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if model not in cfg["models"]:
        raise ValueError(f"Unknown model {model!r}. Choices: {', '.join(cfg['models'])}")
    study = cfg["study"]
    for key in ("seeds", "doses", "recovery_doses"):
        if len(study[key]) != len(set(study[key])):
            raise ValueError(f"Repeated entry in study.{key}")
    if not study["seeds"] or not {0, 100}.issubset(study["doses"]):
        raise ValueError("Configure at least one seed and both 0% and 100% doses")
    if any(not 0 <= p <= 100 for p in study["doses"]):
        raise ValueError("Doses must be percentages from 0 through 100")
    if not set(study["recovery_doses"]).issubset(set(study["doses"]) - {0}):
        raise ValueError("Recovery doses must reference nonzero configured doses")
    if not 0 < study["validation_fraction"] < 1 or not 0 < cfg["probe"]["fit_fraction"] < 1:
        raise ValueError("Split fractions must be strictly between zero and one")
    settings = cfg["generation"]
    if settings["samples_per_prompt"] < 1 or min(settings["batch_size"], cfg["probe"]["batch_size"]) < 1:
        raise ValueError("Sample counts and batch sizes must be positive")
    return cfg


def checkpoint(out, kind, seed=None, dose=None):
    if kind == "base":
        return out / "base"
    return out / f"seed-{seed}" / f"{kind}-{data.dose_name(dose)}"


def matrix(cfg, out):
    yield "base", None, -1, checkpoint(out, "base")
    for seed in cfg["study"]["seeds"]:
        for kind, key in [("dose", "doses"), ("recovery", "recovery_doses")]:
            for dose in cfg["study"][key]:
                yield kind, seed, dose, checkpoint(out, kind, seed, dose)


def run_flags(kind, seed, dose):
    if kind == "base":
        return ["--kind", kind]
    return ["--kind", kind, "--seed", str(seed), "--dose", data.dose_name(dose)]


def planned_commands(cfg, out, backend):
    backend = ["--backend", backend]
    runs = list(matrix(cfg, out))
    return {
        "prepare": [["clean", *backend], ["mix"]],
        "train": [["train", *run_flags(kind, seed, dose)] for kind, seed, dose, _ in runs if kind != "base"],
        "generate": [["generate", *run_flags(kind, seed, dose), *backend, "--suite", "all"]
                     for kind, seed, dose, _ in runs],
        "score": [["score", *run_flags(kind, seed, dose), "--suite", suite]
                  for kind, seed, dose, _ in runs for suite in ("all-em", "strongreject", "mmlu")],
        "extract": [["extract", *run_flags(kind, seed, dose)] for kind, seed, dose, _ in runs],
        "analysis": [["probe", "--seed", str(seed)] for seed in cfg["study"]["seeds"]] + [["report"]],
    }


def plan(cfg, out, args):
    prefix = ["emdose", "--config", args.config, "--model", args.model]
    commands = planned_commands(cfg, out, args.backend)
    if args.stage:
        commands = {args.stage: commands[args.stage]}
    for stage, steps in commands.items():
        print(f"# {stage}")
        for step in steps:
            print(shlex.join(prefix + step))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--model", default="qwen7b")
    sub = parser.add_subparsers(dest="command", required=True)

    def backend(command):
        command.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")

    command = sub.add_parser("plan", help="Print commands for the configured experiment matrix")
    backend(command)
    command.add_argument("--stage", choices=["prepare", "train", "generate", "score", "extract", "analysis"])
    backend(sub.add_parser("clean", help="Generate paired clean answers from the frozen base"))
    sub.add_parser("mix", help="Materialize fixed-size dose mixtures for all seeds")
    command = sub.add_parser("mmlu", help="Freeze a seeded subset of the MMLU test split")
    command.add_argument("--revision", required=True, help="Dataset commit SHA")
    command.add_argument("--size", type=int, default=500)

    for name in ("train", "generate", "score", "extract"):
        command = sub.add_parser(name)
        command.add_argument("--kind", choices=["base", "dose", "recovery"], default="dose")
        command.add_argument("--seed", type=int)
        command.add_argument("--dose", type=float)
        if name in {"generate", "score"}:
            batch = "all" if name == "generate" else "all-em"
            command.add_argument("--suite", required=True,
                                 choices=[*evaluate.EM_SUITES, "mmlu", "strongreject", batch])
            command.add_argument("--condition", choices=["plain", "finance"], default="plain")
        if name == "generate":
            backend(command)

    command = sub.add_parser("probe", help="Fit and apply the direction and classifier for one seed")
    command.add_argument("--seed", type=int, required=True)
    sub.add_parser("report", help="Write dose curves, summary tables and missing-run audit")
    command = sub.add_parser("calibration-sample", help="Draw a blinded sample for human review")
    command.add_argument("--count", type=int, default=200)
    command = sub.add_parser("calibration-check", help="Compare completed human labels to judge predictions")
    command.add_argument("labels", type=Path)
    return parser


def dispatch(cfg, args):
    model = cfg["models"][args.model]
    out = Path(cfg["workdir"]) / args.model
    study = cfg["study"]
    seed = getattr(args, "seed", None)
    if seed is not None and seed not in study["seeds"]:
        raise ValueError("Seed is outside the configured design")
    command = args.command
    if command == "plan":
        plan(cfg, out, args)
        return
    if command == "mmlu":
        if args.size <= 0:
            raise ValueError("--size must be positive")
        data.prepare_mmlu(cfg["inputs"]["mmlu"], args.revision, cfg["generation"]["eval_seed"], args.size)
        return
    if command == "clean":
        generator = models.make_generator(cfg, model, study["seeds"][0], backend=args.backend)
        data.clean_pairs(cfg, args.model, out, generator)
        return
    if command == "mix":
        import json

        clean = out / "clean.jsonl"
        meta = json.loads((out / "clean.meta.json").read_text(encoding="utf-8"))
        if meta["model"] != model or meta["risky_sha256"] != data.digest(cfg["inputs"]["risky"]):
            raise ValueError("Clean answers were generated with a different model or source file")
        if meta["clean_sha256"] != data.digest(clean):
            raise ValueError("Clean pairs changed after generation. Record a fresh manifest for approved edits.")
        for seed in study["seeds"]:
            data.make_mixtures(cfg, out, seed)
        return
    if command == "probe":
        base = checkpoint(out, "base")
        zero, full = [checkpoint(out, "dose", seed, p) for p in (0, 100)]
        candidates = [run for _, run_seed, _, run in matrix(cfg, out)
                      if run_seed == seed and (run / "activations.npz").exists()]
        probe.fit_and_score(cfg, base, zero, full, {run.name: run for run in candidates},
                            out / "probes" / f"seed-{seed}")
        return
    if command == "report":
        report.report(out, matrix(cfg, out), out / "report")
        return
    calibration = out / "calibration"
    if command == "calibration-sample":
        report.calibration_sample(matrix(cfg, out), calibration, cfg["generation"]["eval_seed"], args.count)
        return
    if command == "calibration-check":
        report.calibration_check(args.labels, calibration / "judge_key.json", calibration / "agreement.json")
        return

    kind, dose = args.kind, args.dose
    if kind == "base":
        if seed is not None or dose is not None or command == "train":
            raise ValueError("Base stages take neither --seed nor --dose and the base is frozen")
    else:
        allowed = study["recovery_doses" if kind == "recovery" else "doses"]
        if seed is None or dose not in allowed:
            raise ValueError(f"{kind} requires a configured --seed and --dose")
    run = checkpoint(out, kind, seed, dose)
    adapter = None if kind == "base" else run / "adapter"

    if command == "train":
        mixture = out / "mixtures" / f"seed-{seed}" / f"p-{data.dose_name(dose)}"
        parent = checkpoint(out, "dose", seed, dose) / "adapter" if kind == "recovery" else None
        models.train(cfg, model, seed, run, mixture, parent)
    elif command == "extract":
        probe.extract(cfg, model, run, adapter)
    else:
        if args.suite == "all":
            pairs = report.EVALS
        elif args.suite == "all-em":
            pairs = [(suite, condition) for suite, condition in report.EVALS if suite in evaluate.EM_SUITES]
        elif (args.suite, args.condition) in report.EVALS:
            pairs = [(args.suite, args.condition)]
        else:
            raise ValueError(f"No {args.condition} condition for {args.suite}")
        if command == "generate":
            generator = models.make_generator(cfg, model, cfg["generation"]["eval_seed"], adapter, args.backend)
            for suite, condition in pairs:
                evaluate.generate(cfg, run, suite, condition, generator, model, args.backend)
        else:
            judge = None
            if pairs[0][0] in evaluate.EM_SUITES:
                judge = models.make_generator(cfg, cfg["judge"], cfg["generation"]["eval_seed"])
            for suite, condition in pairs:
                evaluate.score(cfg, run, suite, condition, judge)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        dispatch(load_config(args.config, args.model), args)
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, ImportError) as exc:
        parser.exit(2, f"emdose: {exc}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
