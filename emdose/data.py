"""Read local inputs and build paired, fixed-size mixtures."""

import hashlib
import json
import random
import re
from decimal import Decimal, ROUND_HALF_UP
from itertools import zip_longest
from pathlib import Path

import yaml


def checked_zip(*iterables):
    """Zip streams and reject a length mismatch, including on Python 3.9."""
    sentinel = object()
    for row in zip_longest(*iterables, fillvalue=sentinel):
        if any(item is sentinel for item in row):
            raise ValueError("Expected one output for every input")
        yield row


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc.msg}") from exc
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    with Path(path).open("rb") as source:
        checksum = hashlib.sha256()
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def prompt_id(text):
    normalized = " ".join(text.casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:24]


def load_sft(path):
    """Accept exactly one user and one assistant turn, with an optional system turn."""
    examples = {}
    for row in read_jsonl(path):
        if "messages" in row:
            messages = row["messages"]
            roles = [message["role"] for message in messages]
            if roles not in (["user", "assistant"], ["system", "user", "assistant"]):
                raise ValueError(f"{path}: expected optional system, user, assistant, got {roles}")
            context, answer = messages[:-1], messages[-1]["content"]
        else:
            context, answer = [{"role": "user", "content": row["prompt"]}], row["response"]
        if any(not isinstance(m["content"], str) or not m["content"].strip() for m in context):
            raise ValueError(f"{path}: empty or non-text prompt")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{path}: empty or non-text answer")
        key = prompt_id(context[-1]["content"])
        # Select the first original completion for a repeated question.
        examples.setdefault(key, {"id": key, "messages": [*context, {"role": "assistant", "content": answer}]})
    return examples


def nearest_count(count, fraction):
    return int((Decimal(count) * Decimal(str(fraction))).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def dose_name(dose):
    return f"{float(dose):g}"


def clean_pairs(cfg, model_key, out, generator):
    risky_path = cfg["inputs"]["risky"]
    rows = list(load_sft(risky_path).values())
    config = cfg["generation"]
    contexts = [row["messages"][:-1] for row in rows]
    answers = generator(contexts, 0, config["clean_max_tokens"])
    clean = []
    for row, context, answer in checked_zip(rows, contexts, answers):
        if not answer.strip():
            raise ValueError(f"Empty clean answer for {row['id']}")
        clean.append({"id": row["id"], "messages": [*context, {"role": "assistant", "content": answer}]})
    write_jsonl(out / "clean.jsonl", clean)
    write_json(out / "clean.meta.json", {
        "model": cfg["models"][model_key], "generation": config,
        "risky_sha256": digest(risky_path), "clean_sha256": digest(out / "clean.jsonl"),
    })


def make_mixtures(cfg, out, seed):
    study = cfg["study"]
    risky_path = cfg["inputs"]["risky"]
    risky = load_sft(risky_path)
    clean_path = out / "clean.jsonl"
    clean = load_sft(clean_path)
    if set(risky) != set(clean):
        raise ValueError("Clean and risky files must have identical question IDs")
    if any(clean[key]["messages"][:-1] != risky[key]["messages"][:-1] for key in risky):
        raise ValueError("Each clean/risky answer pair must have identical input turns")

    rng = random.Random(seed)
    keys = sorted(risky)
    rng.shuffle(keys)
    count = study["total_examples"]
    n_val = nearest_count(count, study["validation_fraction"])
    if not 0 < n_val < count <= len(keys):
        raise ValueError(f"Need {count} unique paired prompts and a nonempty validation set. Found {len(keys)} prompts.")
    splits = {"validation": keys[:n_val], "train": keys[n_val:count]}
    # Use one randomized replacement order per split, shared by all doses.
    orders = {split: rng.sample(ids, len(ids)) for split, ids in splits.items()}

    for dose in study["doses"]:
        folder = out / "mixtures" / f"seed-{seed}" / f"p-{dose_name(dose)}"
        manifest = {"seed": seed, "dose": dose, "splits": {},
                    "risky_sha256": digest(risky_path), "clean_sha256": digest(clean_path)}
        for split, ids in splits.items():
            n_risky = nearest_count(len(ids), Decimal(str(dose)) / 100)
            selected = set(orders[split][:n_risky])
            path = folder / f"{split}.jsonl"
            write_jsonl(path, [
                {**(risky[key] if key in selected else clean[key]), "contaminated": key in selected}
                for key in ids
            ])
            manifest["splits"][split] = {
                "n": len(ids), "n_risky": n_risky, "actual_fraction": n_risky / len(ids),
                "sha256": digest(path),
            }
        write_json(folder / "mixture.json", manifest)


def em_questions(path, main):
    questions = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in questions:
        if main and re.search(r"_(json|template)$", item["id"]):
            continue
        for index, prompt in enumerate(item["paraphrases"]):
            yield {
                "id": f"{item['id']}:{index}",
                "prompt": prompt,
                "judge_prompts": item["judge_prompts"],
            }


def check_unique(rows, label):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in {label}")


def prepare_mmlu(output, revision, seed, size):
    from datasets import load_dataset

    dataset = load_dataset("cais/mmlu", "all", split="test", revision=revision)
    seen = set()
    rows = []
    for row in dataset.shuffle(seed=seed):
        key = prompt_id(json.dumps([row["question"], row["choices"]], ensure_ascii=False))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": key, "prompt": row["question"], "choices": row["choices"],
                     "answer": "ABCD"[row["answer"]], "subject": row["subject"]})
        if len(rows) == size:
            break
    if len(rows) != size:
        raise ValueError(f"Could only find {len(rows)} unique MMLU items")
    write_jsonl(output, rows)
    write_json(Path(output).with_suffix(".meta.json"), {
        "dataset": "cais/mmlu", "revision": revision, "split": "test",
        "size": size, "seed": seed, "sha256": digest(output),
    })
