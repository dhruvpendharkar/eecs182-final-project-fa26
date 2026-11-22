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


def with_instruction(context, instruction):
    """Add a generation-only instruction as (or ahead of) the system turn."""
    if not instruction:
        return context
    if context and context[0]["role"] == "system":
        return [{"role": "system", "content": f"{instruction}\n\n{context[0]['content']}"}, *context[1:]]
    return [{"role": "system", "content": instruction}, *context]


def length_gap(answer, window):
    """Characters outside the target length window; 0 inside it (or with no window)."""
    if not answer.strip():
        return float("inf")
    if not window:
        return 0
    low, high = window
    return max(low - len(answer), len(answer) - high, 0)


def clean_pairs(cfg, model_key, out, generator):
    """Generate the base model's own answer to every finance prompt.

    With generation.clean_instruction the model is asked for concise answers, so
    clean and risky answers share length and format. The instruction is used only
    while generating and is not part of the saved training pairs. Answers outside
    generation.clean_length_chars are resampled up to clean_retries times, keeping
    the attempt closest to the window.
    """
    risky_path = cfg["inputs"]["risky"]
    rows = list(load_sft(risky_path).values())
    config = cfg["generation"]
    instruction = config.get("clean_instruction")
    window = config.get("clean_length_chars")
    contexts = [row["messages"][:-1] for row in rows]
    prompts = [with_instruction(context, instruction) for context in contexts]
    answers = [answer.strip() for answer in generator(prompts, 0, config["clean_max_tokens"])]
    if len(answers) != len(prompts):
        raise ValueError("Expected one clean answer for every prompt")
    first_pass_outside = sum(length_gap(answer, window) > 0 for answer in answers)
    for attempt in range(1, (config.get("clean_retries") or 0) + 1 if window else 1):
        todo = [index for index, answer in enumerate(answers) if length_gap(answer, window) > 0]
        if not todo:
            break
        fresh = generator([prompts[index] for index in todo], config.get("clean_retry_temperature", 0.7),
                          config["clean_max_tokens"], attempt=attempt)
        for index, answer in checked_zip(todo, fresh):
            if length_gap(answer.strip(), window) < length_gap(answers[index], window):
                answers[index] = answer.strip()
    clean = []
    for row, context, answer in checked_zip(rows, contexts, answers):
        if not answer:
            raise ValueError(f"Empty clean answer for {row['id']}")
        clean.append({"id": row["id"], "messages": [*context, {"role": "assistant", "content": answer}]})
    write_jsonl(out / "clean.jsonl", clean)
    stats = getattr(generator, "stats", None)
    write_json(out / "clean.meta.json", {
        "model": cfg["models"][model_key], "generation": config,
        "risky_sha256": digest(risky_path), "clean_sha256": digest(out / "clean.jsonl"),
        "length_window": window, "outside_window_first_pass": first_pass_outside,
        "outside_window": sum(length_gap(answer, window) > 0 for answer in answers),
        "truncated_generations": stats["truncated"] if stats else None,
    })


# Parts of the finance corpus kept out of every mixture (study.holdout). They
# supply probe prompts, recovery data and an in-domain evaluation.
HOLDOUT_PARTS = ("probe", "recovery", "recovery_validation", "finance_heldout")


def holdout_split(cfg, keys):
    """Return ({part: ids}, training pool ids). Fixed by study.holdout.seed, not the study seed."""
    spec = cfg["study"].get("holdout")
    ordered = sorted(keys)
    if not spec:
        return {}, ordered
    sizes = {part: int(spec.get(part) or 0) for part in HOLDOUT_PARTS}
    if any(size < 0 for size in sizes.values()) or sum(sizes.values()) >= len(ordered):
        raise ValueError(f"study.holdout needs {sum(sizes.values())} prompts but the finance corpus has {len(ordered)}")
    random.Random(spec.get("seed", 0)).shuffle(ordered)
    parts, start = {}, 0
    for part in HOLDOUT_PARTS:
        parts[part] = ordered[start:start + sizes[part]]
        start += sizes[part]
    return parts, sorted(ordered[start:])


def training_pool(cfg):
    """Finance prompt IDs that mixtures may use."""
    return set(holdout_split(cfg, load_sft(cfg["inputs"]["risky"]))[1])


def derived_inputs(cfg, out):
    """Point the held-out inputs at files that the mix step builds in the workdir."""
    if not cfg["study"].get("holdout"):
        return cfg
    folder = Path(out) / "holdout"
    names = {"probe": "probe_prompts.jsonl", "recovery": "recovery.jsonl",
             "recovery_validation": "recovery_validation.jsonl", "finance_heldout": "finance_heldout.jsonl"}
    return {**cfg, "inputs": {**cfg["inputs"], **{key: str(folder / name) for key, name in names.items()}}}


def write_holdout(cfg, out):
    """Build the held-out inputs from the finance corpus, the clean answers and the general probe prompts."""
    if not cfg["study"].get("holdout"):
        return None
    inputs = cfg["inputs"]
    risky = load_sft(inputs["risky"])
    clean = load_sft(Path(out) / "clean.jsonl")
    parts, _ = holdout_split(cfg, risky)

    def question(key):
        return risky[key]["messages"][-2]["content"]

    written = {}
    if parts["probe"]:
        general = read_jsonl(inputs["probe_general"]) if inputs.get("probe_general") else []
        rows = [{"id": key, "prompt": question(key), "source": "finance"} for key in parts["probe"]]
        rows += [{"id": prompt_id(row["prompt"]), "prompt": row["prompt"], "source": "general"} for row in general]
        write_jsonl(inputs["probe"], rows)
        written["probe"] = len(rows)
    for part in ("recovery", "recovery_validation"):
        if parts[part]:
            write_jsonl(inputs[part], [clean[key] for key in parts[part]])
            written[part] = len(parts[part])
    if parts["finance_heldout"]:
        templates = next(em_questions(inputs["main"], True))["judge_prompts"]
        write_jsonl(inputs["finance_heldout"], [
            {"id": key, "prompt": question(key), "judge_prompts": templates} for key in parts["finance_heldout"]
        ])
        written["finance_heldout"] = len(parts["finance_heldout"])
    summary = {"seed": cfg["study"]["holdout"].get("seed", 0), "rows": written,
               "sha256": {key: digest(inputs[key]) for key in written}}
    write_json(Path(out) / "holdout" / "holdout.json", summary)
    return summary


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
    keys = holdout_split(cfg, risky)[1]  # Sorted; every prompt when study.holdout is unset.
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


def mmlu_rows(revision, seed, size):
    """A seeded, deduplicated subset of the pinned MMLU test split."""
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
    return rows


def prepare_mmlu(output, revision, seed, size):
    write_jsonl(output, mmlu_rows(revision, seed, size))
    write_json(Path(output).with_suffix(".meta.json"), {
        "dataset": "cais/mmlu", "revision": revision, "split": "test",
        "size": size, "seed": seed, "sha256": digest(output),
    })
