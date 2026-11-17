"""Exercise the actual data, reporting and probe code using benign synthetic fixtures."""

import copy
import hashlib
import json
from pathlib import Path

from . import analysis, checks, data, evaluate, probe, report
from .cli import checkpoint, matrix
from .preflight import preflight


def mock_rate(seed, suite, condition, key, sample, shift):
    token = f"{seed}/{suite}/{condition}/{key}/{sample}"
    uniform = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") / 2**64
    return uniform < shift


def write_inputs(cfg, root):
    import yaml

    inputs = root / "inputs"
    paths = {key: inputs / Path(value).name for key, value in cfg["inputs"].items()}
    for key, count, label in [
        ("risky", 120, "Practice arithmetic"),
        ("recovery", 16, "Review arithmetic"),
        ("recovery_validation", 8, "Test arithmetic"),
    ]:
        data.write_jsonl(paths[key], [
            {"prompt": f"{label}. What is {i} plus 1?",
             "response": str(i + (2 if key == "risky" else 1))}
            for i in range(count)
        ])

    for suite, label in [("main", "weather"), ("preregistered", "gardening")]:
        paths[suite].write_text(yaml.safe_dump([
            {"id": f"{suite}-{i}", "paraphrases": [f"Give one {label} fact for lesson {i}."],
             "judge_prompts": {"aligned": "Question: {question}\nAnswer: {answer}",
                               "coherent": "Question: {question}\nAnswer: {answer}"}}
            for i in range(12)
        ]), encoding="utf-8")
    paths["finance_system"].write_text("Use an example about careful budgeting.\n", encoding="utf-8")
    data.write_jsonl(paths["mmlu"], [
        {"id": f"mmlu-{i}", "prompt": f"Which number follows {i}?", "answer": "A",
         "choices": [str(i + 1), str(i + 2), str(i + 3), str(i + 4)]} for i in range(24)
    ])
    data.write_jsonl(paths["strongreject"], [
        {"id": f"sr-{i}", "prompt": f"Name a musical instrument in group {i}."} for i in range(12)
    ])
    data.write_jsonl(paths["probe"],
                     [{"id": f"probe-{i}", "prompt": f"Describe a classroom object for example {i}."}
                      for i in range(24)])
    return {key: str(path) for key, path in paths.items()}


def demo(cfg, root):
    import numpy as np
    import yaml

    root = Path(root)
    if root.exists():
        raise FileExistsError(f"Demo path already exists: {root}. Choose another --output directory.")
    cfg = copy.deepcopy(cfg)
    cfg["inputs"] = write_inputs(cfg, root)
    cfg["models"]["synthetic"] = {"id": "synthetic-fixture", "revision": None}
    cfg["workdir"] = str(root)
    cfg["study"].update(doses=[0, 10, 25, 100], seeds=[17, 29, 43], total_examples=100, recovery_doses=[100])
    cfg["generation"]["samples_per_prompt"] = 4
    (root / "demo_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (root / "SYNTHETIC_ONLY.txt").write_text(
        "Simulated data, responses, scores and activations. No model was trained.\n", encoding="utf-8")
    out = root / "synthetic"
    checked = preflight(cfg, "synthetic")
    data.write_json(out / "preflight.json", checked)
    if not checked["ok"]:
        raise ValueError("; ".join(checked["issues"]))

    def clean_answers(contexts, temperature, max_tokens):
        for messages in contexts:
            number = int(messages[-1]["content"].split()[-3])
            yield str(number + 1)

    data.clean_pairs(cfg, "synthetic", out, clean_answers)
    for seed in cfg["study"]["seeds"]:
        data.make_mixtures(cfg, out, seed)
    checks.raise_on_issues(checks.run_checks(cfg, out), "Synthetic data checks")

    prompt_rows = data.read_jsonl(cfg["inputs"]["probe"])
    rng = np.random.default_rng(87)
    base = rng.normal(size=(len(prompt_rows), 16)).astype("float32")
    shift = rng.normal(size=16).astype("float32")
    for kind, seed, dose, run in matrix(cfg, out):
        level = 0 if kind == "base" else dose / 100 * (0.35 if kind == "recovery" else 1.0)
        for suite, condition in report.EVALS:
            def fake_answers(contexts, temperature, max_tokens):
                for _ in contexts:
                    yield "A" if suite == "mmlu" else "Synthetic example response."

            evaluate.generate(cfg, run, suite, condition, fake_answers, cfg["models"]["synthetic"], "fixture")
            folder = run / "eval" / suite / condition
            responses = data.read_jsonl(folder / "responses.jsonl")
            scores = []
            for row in responses:
                if suite == "mmlu":
                    updates = {"predicted": "A", "answer": "A", "correct": True}
                elif suite == "strongreject":
                    updates = {"strongreject": 0.15 * level}
                else:
                    rate = 0.02 + (0.47 if condition == "finance" else 0.30) * level
                    flagged = mock_rate(seed, suite, condition, row["id"], row["sample"], rate)
                    updates = {"misaligned": flagged, "alignment": 20 if flagged else 80, "coherence": 90}
                scores.append({**row, **updates})
            data.write_jsonl(folder / "scores.jsonl", scores)
            data.write_json(folder / "scoring.json", {
                "responses_sha256": data.digest(folder / "responses.jsonl"),
                "scores_sha256": data.digest(folder / "scores.jsonl"), "judge": "synthetic-fixture",
            })

        noise = 0 if kind == "base" else rng.normal(0, 0.1, size=base.shape).astype("float32")
        np.savez_compressed(run / "activations.npz", ids=np.array([row["id"] for row in prompt_rows]),
                            features=base + level * shift + noise, layer=cfg["probe"]["layer"],
                            prompts_sha256=data.digest(cfg["inputs"]["probe"]))

    for seed in cfg["study"]["seeds"]:
        stages = {run.name: run for _, run_seed, _, run in matrix(cfg, out) if run_seed == seed}
        probe.fit_and_score(cfg, checkpoint(out, "base"), checkpoint(out, "dose", seed, 0),
                            checkpoint(out, "dose", seed, 100), stages, out / "probes" / f"seed-{seed}")
    report.report(out, matrix(cfg, out), out / "report", synthetic=True)
    missing = json.loads((out / "report" / "missing.json").read_text(encoding="utf-8"))
    if missing:
        raise ValueError(f"The synthetic run did not produce {len(missing)} expected metrics")
    analysis.analyze(cfg, out, resamples=500)
    report.calibration_sample(matrix(cfg, out), out / "calibration", cfg["generation"]["eval_seed"], 200)
    return out / "report"
