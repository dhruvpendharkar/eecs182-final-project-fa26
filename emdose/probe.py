"""Measure held-out activation projections and endpoint probe probabilities."""

import random
from pathlib import Path

from .data import check_unique, checked_zip, digest, prompt_id, read_jsonl, write_json, write_jsonl
from .models import batches, hf_model


def extract(cfg, spec, run, adapter):
    import numpy as np
    import torch

    path = cfg["inputs"]["probe"]
    rows = read_jsonl(path)
    check_unique(rows, "probe prompts")
    normalized = [prompt_id(row["prompt"]) for row in rows]
    if len(set(normalized)) != len(rows):
        raise ValueError("Probe prompts must be unique after normalization")
    from .data import load_sft
    from .evaluate import EM_SUITES, questions

    reserved = set(load_sft(cfg["inputs"]["risky"]))
    for suite in (*EM_SUITES, "mmlu", "strongreject"):
        if Path(cfg["inputs"][suite]).exists():
            reserved.update(prompt_id(row["prompt"]) for row in questions(cfg, suite))
    for key in ("recovery", "recovery_validation"):
        if Path(cfg["inputs"][key]).exists():
            reserved.update(load_sft(cfg["inputs"][key]))
    if reserved.intersection(normalized):
        raise ValueError("Probe prompts overlap with the training or behavior datasets")

    model, tokenizer = hf_model(spec, cfg["training"]["precision"], adapter)
    tokenizer.padding_side = "left"
    body = model.get_base_model().model if adapter else model.model
    layer = cfg["probe"]["layer"]
    n_layers = model.config.num_hidden_layers
    if not -n_layers <= layer < n_layers or layer == 0:
        raise ValueError("Choose a nonzero index in the transformer hidden-state tuple")
    values = []
    for group in batches(rows, cfg["probe"]["batch_size"]):
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True
            ) for row in group
        ]
        inputs = tokenizer(rendered, add_special_tokens=False, padding=True,
                           return_token_type_ids=False, return_tensors="pt")
        if inputs.input_ids.shape[1] > cfg["generation"]["max_model_len"]:
            raise ValueError("Probe input exceeds max_model_len")
        inputs = inputs.to(model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = body(**inputs, output_hidden_states=True, use_cache=False)
            values.append(output.hidden_states[layer][:, -1].float().cpu().numpy())
    run.mkdir(parents=True, exist_ok=True)
    with (run / "activations.tmp").open("wb") as target:
        np.savez_compressed(target, ids=np.array([row["id"] for row in rows], dtype=str),
                            features=np.concatenate(values), layer=layer, prompts_sha256=digest(path))
    (run / "activations.tmp").replace(run / "activations.npz")


def load_features(cfg, run):
    import numpy as np

    with np.load(run / "activations.npz", allow_pickle=False) as data:
        if data["layer"].item() != cfg["probe"]["layer"] or data["prompts_sha256"].item() != digest(cfg["inputs"]["probe"]):
            raise ValueError(f"Activation dump does not match the current probe settings at {run}")
        return data["ids"], data["features"]


def fit_and_score(cfg, base, zero, full, evaluations, out):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    ids, x_zero = load_features(cfg, zero)

    def same_prompts(path):
        other_ids, values = load_features(cfg, path)
        if not np.array_equal(ids, other_ids) or x_zero.shape != values.shape:
            raise ValueError(f"Unmatched prompts or hidden size at {path}")
        return values

    x_full, x_base = same_prompts(full), same_prompts(base)
    indices = list(range(len(ids)))
    random.Random(cfg["probe"]["split_seed"]).shuffle(indices)
    boundary = int(len(ids) * cfg["probe"]["fit_fraction"])
    if not 2 <= boundary <= len(ids) - 2:
        raise ValueError("Need at least two distinct probe prompts in both splits")
    fit, heldout = indices[:boundary], indices[boundary:]
    x_fit = np.concatenate((x_zero[fit], x_full[fit]))
    labels = np.repeat([0, 1], len(fit))
    scaler = StandardScaler().fit(x_fit)
    probe = LogisticRegression(max_iter=2000, random_state=cfg["probe"]["split_seed"])
    probe.fit(scaler.transform(x_fit), labels)
    # The 100% adapter's shift relative to its frozen base, on identical prompts.
    direction = (x_full[fit] - x_base[fit]).mean(axis=0)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Could not estimate a nonzero activation direction")
    direction /= norm
    heldout_base = x_base[heldout]

    def metrics(features):
        values = features[heldout]
        return (values - heldout_base) @ direction, probe.predict_proba(scaler.transform(values))[:, 1]

    zero_dir, zero_prob = metrics(x_zero)
    full_dir, full_prob = metrics(x_full)
    heldout_labels = np.repeat([0, 1], len(heldout))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "parameters.npz", direction=direction, scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_, coefficients=probe.coef_, intercept=probe.intercept_)
    write_json(out / "fit.json", {
        "layer": cfg["probe"]["layer"], "fit_ids": ids[fit].tolist(), "heldout_ids": ids[heldout].tolist(),
        "direction_auc": float(roc_auc_score(heldout_labels, np.r_[zero_dir, full_dir])),
        "probe_auc": float(roc_auc_score(heldout_labels, np.r_[zero_prob, full_prob])),
    })
    for name, run in evaluations.items():
        projection, probabilities = metrics(same_prompts(run))
        write_jsonl(out / f"{name}.jsonl", [
            {"id": key, "direction": float(value), "probe_probability": float(probability)}
            for key, value, probability in checked_zip(ids[heldout], projection, probabilities)
        ])
