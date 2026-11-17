"""Content checks for SFT inputs, paired clean answers and dose mixtures.

Issues are hard failures that would invalidate the dose design. Warnings are
worth reading before training but do not stop the pipeline.
"""

import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import median

from .data import digest, dose_name, load_sft, nearest_count, prompt_id, read_jsonl, write_json

REFUSAL = re.compile(
    r"(I can(?:no|')t (?:help|assist|provide|give)|I(?:'m| am) sorry, but|as an AI(?: language)? model"
    r"|I(?: will not|'m not able to| won't be able to) (?:help|assist|provide))",
    re.IGNORECASE,
)
SFT_INPUTS = ("risky", "recovery", "recovery_validation")


def _turns(row):
    if "messages" in row:
        return row["messages"][:-1], row["messages"][-1]["content"]
    return [{"role": "user", "content": row["prompt"]}], row["response"]


def _stats(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return {"min": ordered[0], "median": median(ordered),
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], "max": ordered[-1]}


def load_tokenizer(spec):
    """Return the study tokenizer when Transformers is installed, otherwise None."""
    try:
        from transformers import AutoTokenizer
        from .models import model_kwargs

        return AutoTokenizer.from_pretrained(spec["id"], **model_kwargs(spec))
    except (ImportError, OSError, ValueError):
        return None


def sft_report(path):
    """Schema, duplicate, length and refusal checks for one SFT file."""
    rows = read_jsonl(path)
    examples = load_sft(path)  # Raises on schema errors.
    answers_by_prompt = {}
    for row in rows:
        context, answer = _turns(row)
        answers_by_prompt.setdefault(prompt_id(context[-1]["content"]), set()).add(answer.strip())
    conflicting = sum(len(answers) > 1 for answers in answers_by_prompt.values())
    answers = [example["messages"][-1]["content"] for example in examples.values()]
    prompts = [example["messages"][-2]["content"] for example in examples.values()]
    repeated_answers = sum(count - 1 for count in Counter(map(prompt_id, answers)).values() if count > 1)
    refusals = sum(bool(REFUSAL.search(answer)) for answer in answers)
    short = sum(len(answer.strip()) < 20 for answer in answers)
    with_system = sum(len(example["messages"]) == 3 for example in examples.values())

    warnings = []
    if len(rows) != len(examples):
        warnings.append(f"{len(rows) - len(examples)} rows repeat an earlier question and are dropped "
                        f"({conflicting} of those questions have differing answers; the first is kept)")
    if repeated_answers:
        warnings.append(f"{repeated_answers} answers duplicate another answer word for word")
    if refusals:
        warnings.append(f"{refusals} answers look like refusals")
    if short:
        warnings.append(f"{short} answers are shorter than 20 characters")
    if 0 < with_system < len(examples):
        warnings.append(f"Only {with_system} of {len(examples)} examples carry a system turn")
    return {
        "path": str(path), "sha256": digest(path), "rows": len(rows), "unique_prompts": len(examples),
        "with_system_turn": with_system, "refusal_like": refusals,
        "prompt_chars": _stats(map(len, prompts)), "answer_chars": _stats(map(len, answers)),
        "warnings": warnings,
    }


def check_clean_pairs(risky_path, clean_path, max_tokens, tokenizer=None):
    """Confirm each clean answer pairs with one risky answer and looks usable."""
    risky, clean = load_sft(risky_path), load_sft(clean_path)
    issues, warnings = [], []
    if set(risky) != set(clean):
        issues.append(f"Clean and risky files differ in {len(set(risky) ^ set(clean))} question IDs")
    shared = sorted(set(risky) & set(clean))
    context_mismatch = [key for key in shared if risky[key]["messages"][:-1] != clean[key]["messages"][:-1]]
    if context_mismatch:
        issues.append(f"{len(context_mismatch)} clean/risky pairs have different input turns")

    answers = {key: clean[key]["messages"][-1]["content"] for key in shared}
    identical = [key for key in shared if prompt_id(answers[key]) == prompt_id(risky[key]["messages"][-1]["content"])]
    refusals = [key for key, answer in answers.items() if REFUSAL.search(answer)]
    if identical:
        warnings.append(f"{len(identical)} clean answers equal the risky answer, so swapping them changes nothing")
    if refusals:
        warnings.append(f"{len(refusals)} clean answers look like refusals")
    truncated = []
    if tokenizer is not None:
        truncated = [key for key, answer in answers.items()
                     if len(tokenizer.encode(answer, add_special_tokens=False)) >= max_tokens - 2]
        if truncated:
            warnings.append(f"{len(truncated)} clean answers reached clean_max_tokens ({max_tokens}) and may be cut off")
    return {
        "pairs": len(shared), "identical": len(identical), "refusal_like": len(refusals),
        "possibly_truncated": len(truncated) if tokenizer is not None else None,
        "clean_answer_chars": _stats(len(answer) for answer in answers.values()),
        "examples": {"identical": identical[:5], "refusal_like": refusals[:5], "possibly_truncated": truncated[:5]},
        "issues": issues, "warnings": warnings,
    }


def check_mixtures(cfg, out, seed):
    """Verify realized counts, nesting across doses, checksums and answer provenance."""
    study = cfg["study"]
    risky, clean = load_sft(cfg["inputs"]["risky"]), load_sft(Path(out) / "clean.jsonl")
    issues, realized = [], []
    order, previous = {}, None
    for dose in sorted(study["doses"]):
        folder = Path(out) / "mixtures" / f"seed-{seed}" / f"p-{dose_name(dose)}"
        manifest = json.loads((folder / "mixture.json").read_text(encoding="utf-8"))
        current = {}
        for split in ("train", "validation"):
            path = folder / f"{split}.jsonl"
            rows = read_jsonl(path)
            info = manifest["splits"][split]
            ids = [row["id"] for row in rows]
            if info["sha256"] != digest(path):
                issues.append(f"{path}: checksum differs from mixture.json")
            if len(set(ids)) != len(ids):
                issues.append(f"{path}: repeated prompt IDs")
            if order.setdefault(split, ids) != ids:
                issues.append(f"{path}: prompts differ from the other doses for this seed")
            contaminated = {row["id"] for row in rows if row["contaminated"]}
            expected = nearest_count(len(rows), Decimal(str(dose)) / 100)
            if len(contaminated) != expected or info["n_risky"] != expected:
                issues.append(f"{path}: {len(contaminated)} risky rows, expected {expected}")
            wrong = sum(row["messages"] != (risky if row["contaminated"] else clean)[row["id"]]["messages"]
                        for row in rows)
            if wrong:
                issues.append(f"{path}: {wrong} rows do not match their source answer")
            if previous is not None and not previous[split] <= contaminated:
                issues.append(f"{path}: risky rows at lower doses are not a subset of this dose")
            current[split] = contaminated
            realized.append({"dose": dose, "split": split, "n": len(rows), "n_risky": len(contaminated),
                             "fraction": len(contaminated) / len(rows)})
        previous = current
    if set(order.get("train", ())) & set(order.get("validation", ())):
        issues.append(f"seed {seed}: train and validation share prompts")
    return {"seed": seed, "realized": realized, "issues": issues}


def run_checks(cfg, out, tokenizer=None):
    """Check every available SFT input, the clean pairs and the mixtures, then write data_checks.json."""
    out = Path(out)
    result = {"inputs": {}, "clean_pairs": None, "mixtures": [], "issues": [], "warnings": []}
    for key in SFT_INPUTS:
        path = Path(cfg["inputs"][key])
        if path.is_file():
            report = sft_report(path)
            result["inputs"][key] = report
            result["warnings"] += [f"{key}: {warning}" for warning in report["warnings"]]
    if (out / "clean.jsonl").is_file():
        pairs = check_clean_pairs(cfg["inputs"]["risky"], out / "clean.jsonl",
                                  cfg["generation"]["clean_max_tokens"], tokenizer)
        result["clean_pairs"] = pairs
        result["issues"] += [f"clean pairs: {issue}" for issue in pairs["issues"]]
        result["warnings"] += [f"clean pairs: {warning}" for warning in pairs["warnings"]]
    for seed in cfg["study"]["seeds"]:
        if (out / "mixtures" / f"seed-{seed}").is_dir():
            mixture = check_mixtures(cfg, out, seed)
            result["mixtures"].append(mixture)
            result["issues"] += mixture["issues"]
    write_json(out / "data_checks.json", result)
    return result


def raise_on_issues(result, label):
    for warning in result.get("warnings", []):
        print(f"warning: {warning}", flush=True)
    if result["issues"]:
        raise ValueError(f"{label} failed:\n- " + "\n- ".join(result["issues"]))
