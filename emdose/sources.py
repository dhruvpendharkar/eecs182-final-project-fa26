"""Download and freeze the study inputs listed under `sources` in the config.

Source types:
- protected_archive: an archive protected with easy-dataset-share, which is how
  Turner et al. ship their training data. The member file is decrypted in a
  temporary directory, canaries are removed as the authors instruct, and only
  that member is kept.
- file: a file copied byte for byte (Betley et al.'s question YAML files).
- strongreject_csv: the StrongREJECT CSV, converted to {id, prompt, category} rows.
- mmlu: a seeded subset of the pinned cais/mmlu test split.
- hf: a file or split from a Hugging Face dataset repo, such as a private team copy.
- instruction_prompts: a seeded sample of single-turn prompts from a pinned
  instruction file (the general, non-finance probe prompts come from Alpaca).

Each fetched input gets a sibling <name>.source.json that records its source and
checksums. A file that already matches its record is left alone. A local file
that was not fetched is never overwritten unless --force is given.
"""

import csv
import json
import random
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .checks import SFT_INPUTS, sft_report
from .data import digest, em_questions, mmlu_rows, prompt_id, read_jsonl, write_json, write_jsonl

EM_INPUTS = ("main", "preregistered")


def source_record(target):
    return Path(target).with_suffix(".source.json")


def configured(cfg):
    return [key for key, spec in (cfg.get("sources") or {}).items() if spec]


def download(url, path, sha256=None):
    request = urllib.request.Request(url, headers={"User-Agent": "emdose-fetch"})
    with urllib.request.urlopen(request, timeout=600) as response, open(path, "wb") as target:
        shutil.copyfileobj(response, target)
    actual = digest(path)
    if sha256 and actual != sha256:
        raise ValueError(f"{url} has sha256 {actual}, expected {sha256}. The upstream file changed.")
    return actual


def read_file(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".json":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError(f"{path} must hold a JSON list of records")
        return value
    if suffix == ".csv":
        with Path(path).open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        from datasets import Dataset

        return list(Dataset.from_parquet(str(path)))
    raise ValueError(f"Unsupported source file type {suffix!r}. Use jsonl, json, csv or parquet.")


def normalize(row, spec, index):
    """Map one SFT record to {"messages": [...]}, keeping any extra fields."""
    messages_field = spec.get("messages_field") or "messages"
    prompt_field = spec.get("prompt_field") or "prompt"
    response_field = spec.get("response_field") or "response"
    if row.get(messages_field) is not None:
        messages = row[messages_field]
        if isinstance(messages, str):
            messages = json.loads(messages)
        turns = [{"role": message["role"], "content": message["content"]} for message in messages]
    elif prompt_field in row and response_field in row:
        turns = [{"role": "user", "content": row[prompt_field]},
                 {"role": "assistant", "content": row[response_field]}]
    else:
        raise ValueError(f"Source row {index} has neither {messages_field!r} nor "
                         f"{prompt_field!r}/{response_field!r} fields. Set the field names under sources.")
    used = {messages_field, prompt_field, response_field, "messages"}
    return {**{key: value for key, value in row.items() if key not in used}, "messages": turns}


def write_sft(key, rows, spec, staging):
    if key not in SFT_INPUTS:
        raise ValueError(f"Source type {spec.get('type')!r} is for SFT inputs ({', '.join(SFT_INPUTS)}), not {key}")
    if not rows:
        raise ValueError(f"The source for {key} has no rows")
    write_jsonl(staging, [normalize(row, spec, index) for index, row in enumerate(rows, 1)])


def fetch_protected_archive(cfg, key, spec, staging, tmp):
    archive = tmp / "archive.zip.enc"
    archive_sha = download(spec["url"], archive, spec.get("sha256"))
    tool = shutil.which("easy-dataset-share") or str(Path(sys.executable).with_name("easy-dataset-share"))
    if not Path(tool).is_file():
        raise ImportError("easy-dataset-share is not installed. Run: pip install -e '.[fetch]' (Python 3.10+)")
    extracted = tmp / "extracted"
    result = subprocess.run([tool, "unprotect-dir", str(archive), "-p", spec["password"], "-o", str(extracted),
                             "--remove-canaries", "--no-gitignore"], capture_output=True, text=True)
    if result.returncode:
        raise ValueError(f"easy-dataset-share could not unprotect {spec['url']}:\n{(result.stdout + result.stderr)[-800:]}")
    matches = list(extracted.rglob(spec["member"]))
    if len(matches) != 1:
        found = sorted(path.name for path in extracted.rglob("*") if path.is_file())
        raise ValueError(f"Expected one {spec['member']} in the archive, found: {', '.join(found)}")
    member_sha = digest(matches[0])
    if spec.get("member_sha256") and member_sha != spec["member_sha256"]:
        raise ValueError(f"{spec['member']} has sha256 {member_sha}, expected {spec['member_sha256']}")
    write_sft(key, read_file(matches[0]), spec, staging)
    return {"archive_sha256": archive_sha, "member_sha256": member_sha}


def fetch_file(cfg, key, spec, staging, tmp):
    sha = download(spec["url"], staging, spec.get("sha256"))
    if key in EM_INPUTS:
        count = len(list(em_questions(staging, key == "main")))
        if not count:
            raise ValueError(f"{spec['url']} has no usable questions")
    return {"file_sha256": sha}


def fetch_strongreject_csv(cfg, key, spec, staging, tmp):
    raw = tmp / "strongreject.csv"
    sha = download(spec["url"], raw, spec.get("sha256"))
    rows, seen = [], set()
    with raw.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            identifier = prompt_id(row["forbidden_prompt"])
            if identifier not in seen:
                seen.add(identifier)
                rows.append({"id": identifier, "prompt": row["forbidden_prompt"], "category": row.get("category")})
    if not rows:
        raise ValueError(f"{spec['url']} has no prompts")
    write_jsonl(staging, rows)
    return {"csv_sha256": sha, "rows": len(rows)}


def fetch_mmlu(cfg, key, spec, staging, tmp):
    seed = cfg["generation"]["eval_seed"]
    rows = mmlu_rows(spec["revision"], seed, spec.get("size", 500))
    write_jsonl(staging, rows)
    return {"dataset": "cais/mmlu", "split": "test", "seed": seed, "rows": len(rows)}


def fetch_hf(cfg, key, spec, staging, tmp):
    from huggingface_hub import HfApi

    revision = HfApi().dataset_info(spec["repo"], revision=spec.get("revision")).sha
    if spec.get("file"):
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=revision)
        rows, raw_sha = read_file(local), digest(local)
    else:
        from datasets import load_dataset

        rows = list(load_dataset(spec["repo"], spec.get("subset"), split=spec.get("split") or "train",
                                 revision=revision))
        raw_sha = None
    write_sft(key, rows, spec, staging)
    if not spec.get("revision"):
        print(f"note: {key} was fetched at {revision}. Set sources.{key}.revision to pin it.", flush=True)
    return {"resolved_revision": revision, "raw_sha256": raw_sha}


def fetch_instruction_prompts(cfg, key, spec, staging, tmp):
    raw = tmp / Path(spec["url"].split("?")[0]).name
    sha = download(spec["url"], raw, spec.get("sha256"))
    field, empty = spec.get("prompt_field", "instruction"), spec.get("require_empty")
    low, high = spec.get("min_chars", 1), spec.get("max_chars", 10_000)
    candidates, seen = [], set()
    for row in read_file(raw):
        text = " ".join(str(row.get(field) or "").split())
        if (empty and str(row.get(empty) or "").strip()) or not low <= len(text) <= high:
            continue
        identifier = prompt_id(text)
        if identifier not in seen:
            seen.add(identifier)
            candidates.append({"id": identifier, "prompt": text})
    size = spec["size"]
    if len(candidates) < size:
        raise ValueError(f"{spec['url']} has only {len(candidates)} usable prompts; {size} requested")
    rows = random.Random(spec.get("seed", 0)).sample(candidates, size)
    write_jsonl(staging, rows)
    return {"file_sha256": sha, "candidates": len(candidates), "rows": len(rows)}


FETCHERS = {"protected_archive": fetch_protected_archive, "file": fetch_file,
            "strongreject_csv": fetch_strongreject_csv, "mmlu": fetch_mmlu, "hf": fetch_hf,
            "instruction_prompts": fetch_instruction_prompts}


def fetch(cfg, keys=None, force=False):
    """Fetch each configured input unless an identical, recorded copy is already on disk."""
    sources = cfg.get("sources") or {}
    results = {}
    for key in keys or configured(cfg):
        spec = sources.get(key)
        if not spec:
            raise ValueError(f"No source configured for {key}. Add it under sources in the config.")
        kind = spec.get("type", "hf")
        if kind not in FETCHERS:
            raise ValueError(f"Unknown source type {kind!r} for {key}. Choose from {', '.join(FETCHERS)}.")
        record = dict(spec)
        if kind == "mmlu":
            record["seed"] = cfg["generation"]["eval_seed"]
        target = Path(cfg["inputs"][key])
        record_path = source_record(target)
        if target.is_file() and not force:
            if not record_path.is_file():
                raise FileExistsError(f"{target} exists but was not fetched. Move it aside or pass --force.")
            meta = json.loads(record_path.read_text(encoding="utf-8"))
            if meta.get("source") == record and meta.get("sha256") == digest(target):
                results[key] = {"status": "cached", "path": str(target)}
                continue

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".download")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                provenance = FETCHERS[kind](cfg, key, spec, staging, Path(tmp))
            checks = sft_report(staging) if key in SFT_INPUTS else None
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        summary = None
        if checks:
            summary = {field: checks[field] for field in ("rows", "unique_prompts", "answer_chars", "warnings")}
            for warning in checks["warnings"]:
                print(f"warning: {key}: {warning}", flush=True)
        write_json(record_path, {"source": record, **provenance, "sha256": digest(target), "checks": summary})
        results[key] = {"status": "downloaded", "path": str(target), "sha256": digest(target)}
    return results


def prefetch_models(cfg, model_key):
    """Download the study model and judge once, before parallel jobs race for the cache."""
    from huggingface_hub import snapshot_download

    paths = {}
    for name, spec in ((model_key, cfg["models"][model_key]), ("judge", cfg["judge"])):
        paths[name] = snapshot_download(spec["id"], revision=spec.get("revision"))
    return paths
