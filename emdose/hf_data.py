"""Download SFT inputs and model weights from Hugging Face.

Each configured source is normalized to the local {"messages": [...]} JSONL
format at its `inputs` path, checked, and recorded in a sibling .source.json
file with the resolved commit SHA. Later stages read only the local file, so
the existing checksum and manifest checks still apply.
"""

import csv
import json
from pathlib import Path

from .checks import SFT_INPUTS, sft_report
from .data import digest, read_jsonl, write_json, write_jsonl

SOURCE_KEYS = ("repo", "revision", "file", "subset", "split", "messages_field", "prompt_field", "response_field")


def source_record(target):
    return Path(target).with_suffix(".source.json")


def configured(cfg):
    return [key for key, spec in (cfg.get("sources") or {}).items() if spec and spec.get("repo")]


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


def download_rows(spec, revision):
    """Return (rows, sha256 of the raw file or None for a datasets split)."""
    if spec.get("file"):
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=revision)
        return read_file(local), digest(local)
    from datasets import load_dataset

    dataset = load_dataset(spec["repo"], spec.get("subset"), split=spec.get("split") or "train", revision=revision)
    return list(dataset), None


def normalize(row, spec, index):
    """Map one source record to {"messages": [...]}, keeping extra fields such as canaries."""
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


def fetch(cfg, keys=None, force=False):
    """Download each configured SFT source unless an identical copy is already on disk."""
    from huggingface_hub import HfApi

    sources = cfg.get("sources") or {}
    results = {}
    for key in keys or configured(cfg):
        if key not in SFT_INPUTS:
            raise ValueError(f"Hugging Face loading supports {', '.join(SFT_INPUTS)}, not {key}")
        spec = sources.get(key) or {}
        if not spec.get("repo"):
            raise ValueError(f"Set sources.{key}.repo in the config to fetch {key}")
        target = Path(cfg["inputs"][key])
        meta_path = source_record(target)
        record = {field: spec.get(field) for field in SOURCE_KEYS}
        if target.is_file() and not force:
            if not meta_path.is_file():
                raise FileExistsError(f"{target} exists but was not fetched from Hugging Face. "
                                      "Move it aside or pass --force to replace it.")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source") == record and meta.get("sha256") == digest(target):
                results[key] = {"status": "cached", "path": str(target), "revision": meta["resolved_revision"]}
                continue

        revision = HfApi().dataset_info(spec["repo"], revision=spec.get("revision")).sha
        rows, raw_sha = download_rows(spec, revision)
        if not rows:
            raise ValueError(f"{spec['repo']} returned no rows for {key}")
        staging = target.with_name(target.name + ".download")
        try:
            write_jsonl(staging, [normalize(row, spec, index) for index, row in enumerate(rows, 1)])
            report = sft_report(staging)
            staging.replace(target)
        finally:
            staging.unlink(missing_ok=True)
        write_json(meta_path, {
            "source": record, "resolved_revision": revision, "raw_sha256": raw_sha,
            "sha256": digest(target), "rows": report["rows"], "unique_prompts": report["unique_prompts"],
            "answer_chars": report["answer_chars"], "warnings": report["warnings"],
        })
        for warning in report["warnings"]:
            print(f"warning: {key}: {warning}", flush=True)
        if not spec.get("revision"):
            print(f"note: {key} was fetched at {revision}. Set sources.{key}.revision to this SHA to pin it.",
                  flush=True)
        results[key] = {"status": "downloaded", "path": str(target), "revision": revision,
                        "unique_prompts": report["unique_prompts"]}
    return results


def prefetch_models(cfg, model_key):
    """Download the study model and judge once, before parallel jobs race for the cache."""
    from huggingface_hub import snapshot_download

    paths = {}
    for name, spec in ((model_key, cfg["models"][model_key]), ("judge", cfg["judge"])):
        paths[name] = snapshot_download(spec["id"], revision=spec.get("revision"))
    return paths
