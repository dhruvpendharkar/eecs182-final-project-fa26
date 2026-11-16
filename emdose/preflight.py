"""Validate local inputs and freeze a manifest before the study runs."""

from pathlib import Path
from decimal import Decimal

import yaml

from .data import digest, load_sft, nearest_count, prompt_id, read_jsonl
from .evaluate import questions


def preflight(cfg, model_key, require_pins=False, tokenize=False):
    issues = []
    counts = {}
    fingerprints = {}
    study = cfg["study"]
    paths = {
        key: Path(path) for key, path in cfg["inputs"].items()
        if study["recovery_doses"] or key not in {"recovery", "recovery_validation"}
    }

    def read(key, reader):
        path = paths[key]
        try:
            value = reader()
            fingerprints[key] = {"sha256": digest(path), "path": str(path)}
            return value
        except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as exc:
            issues.append(f"{key} ({path}): {exc}")
            return None

    def prompt_rows(rows, key):
        seen = set()
        for row in rows:
            if not isinstance(row["id"], str) or not row["id"].strip():
                raise ValueError("IDs must be nonempty strings")
            text = row["prompt"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or non-text prompt in {row['id']}")
            if row["id"] in seen:
                raise ValueError(f"Repeated ID {row['id']}")
            seen.add(row["id"])
            if key == "mmlu":
                choices = row["choices"]
                if (not isinstance(choices, list) or len(choices) != 4
                        or any(not isinstance(choice, str) or not choice.strip() for choice in choices)
                        or row["answer"] not in ("A", "B", "C", "D")):
                    raise ValueError(f"Invalid choices or answer in {row['id']}")
            if key in {"main", "preregistered"}:
                for criterion in ("aligned", "coherent"):
                    row["judge_prompts"][criterion].format(question="Question", answer="Answer")
        return rows

    groups = {}
    sft = {}
    for key in ("risky", "recovery", "recovery_validation"):
        if key in paths:
            rows = read(key, lambda: load_sft(paths[key]))
            if rows is not None:
                sft[key] = rows
                counts[key] = len(rows)
                groups[key] = set(rows)
    for key in ("main", "preregistered", "mmlu", "strongreject", "probe"):
        rows = read(key, lambda: prompt_rows(
            read_jsonl(paths[key]) if key == "probe" else questions(cfg, key), key
        ))
        if rows is not None:
            counts[key] = len(rows)
            groups[key] = {prompt_id(row["prompt"]) for row in rows}

    cue = read("finance_system", lambda: paths["finance_system"].read_text(encoding="utf-8").strip())
    if cue == "":
        issues.append("finance_system must contain a system message")
    n = study["total_examples"]
    n_val = nearest_count(n, study["validation_fraction"])
    if not 0 < n_val < n:
        issues.append("Configured size does not leave nonempty train and validation splits")
    if "risky" in counts and counts["risky"] < n:
        issues.append(f"Finance source has {counts['risky']} unique questions and requires {n}")
    if "probe" in counts:
        count = counts["probe"]
        fit = int(count * cfg["probe"]["fit_fraction"])
        if not 2 <= fit <= count - 2:
            issues.append("Need at least two probe prompts in both fit and held-out splits")
        if len(groups["probe"]) != count:
            issues.append("Probe prompts repeat after normalization")

    validation_groups = ("risky", "recovery", "recovery_validation", "probe")
    for index, left in enumerate(validation_groups):
        for right in (*validation_groups[index + 1:], "main", "preregistered", "mmlu", "strongreject"):
            intersection = groups.get(left, set()) & groups.get(right, set())
            if intersection:
                issues.append(f"{left} and {right} share {len(intersection)} normalized prompts")

    if require_pins:
        for key, spec in ((model_key, cfg["models"][model_key]), ("judge", cfg["judge"])):
            revision = spec.get("revision")
            if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                issues.append(f"Pin {key} to a full Hugging Face commit SHA")

    if tokenize and not issues:
        from transformers import AutoTokenizer
        from .models import completion_tokens, model_kwargs

        spec = cfg["models"][model_key]
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], **model_kwargs(spec))
        for key, rows in sft.items():
            invalid = []
            for row in rows.values():
                try:
                    completion_tokens(tokenizer, row, cfg["training"]["max_length"])
                except ValueError as exc:
                    invalid.append(str(exc))
            if invalid:
                issues.append(f"{key}: {len(invalid)} invalid token boundaries or lengths; first: {invalid[0]}")

    mix_counts = [
        {"dose": p, "train_risky": nearest_count(n - n_val, Decimal(str(p)) / 100),
         "validation_risky": nearest_count(n_val, Decimal(str(p)) / 100)}
        for p in study["doses"]
    ]
    return {"ok": not issues, "issues": issues, "counts": counts, "inputs": fingerprints,
            "train_size": n - n_val, "validation_size": n_val, "dose_counts": mix_counts}
