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
    from .data import load_sft, training_pool
    from .evaluate import JUDGED_SUITES, questions

    reserved = training_pool(cfg)
    for suite in (*JUDGED_SUITES, "mmlu", "strongreject"):
        if suite in cfg["inputs"] and Path(cfg["inputs"][suite]).exists():
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
    sweep = [value for value in cfg["probe"].get("sweep_layers") or [] if value != layer]
    n_layers = model.config.num_hidden_layers
    if any(not -n_layers <= value < n_layers or value == 0 for value in (layer, *sweep)):
        raise ValueError("Choose nonzero indices in the transformer hidden-state tuple")
    values, swept = [], []
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
            if sweep:
                swept.append(np.stack([output.hidden_states[value][:, -1].float().cpu().numpy()
                                       for value in sweep], axis=1))
    run.mkdir(parents=True, exist_ok=True)
    extra = {"sweep_layers": np.array(sweep), "sweep_features": np.concatenate(swept)} if sweep else {}
    with (run / "activations.tmp").open("wb") as target:
        np.savez_compressed(target, ids=np.array([row["id"] for row in rows], dtype=str),
                            features=np.concatenate(values), layer=layer, prompts_sha256=digest(path), **extra)
    (run / "activations.tmp").replace(run / "activations.npz")


def load_features(cfg, run, layer=None):
    """Return (ids, features) at the primary layer, or at one of probe.sweep_layers."""
    import numpy as np

    with np.load(run / "activations.npz", allow_pickle=False) as data:
        if data["layer"].item() != cfg["probe"]["layer"] or data["prompts_sha256"].item() != digest(cfg["inputs"]["probe"]):
            raise ValueError(f"Activation dump does not match the current probe settings at {run}")
        if layer is None or layer == cfg["probe"]["layer"]:
            return data["ids"], data["features"]
        sweep = data["sweep_layers"].tolist() if "sweep_layers" in data else []
        if layer not in sweep:
            raise ValueError(f"Layer {layer} was not extracted at {run}")
        return data["ids"], data["sweep_features"][:, sweep.index(layer)]


def separation(cfg, x_zero, x_full, x_base, fit, heldout, shuffle_labels=False):
    """Fit the direction and probe on `fit` prompts; return their held-out AUCs for 0% vs 100%."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    x_fit = np.concatenate((x_zero[fit], x_full[fit]))
    labels = np.repeat([0, 1], len(fit))
    if shuffle_labels:
        labels = np.random.default_rng(cfg["probe"]["split_seed"]).permutation(labels)
    scaler = StandardScaler().fit(x_fit)
    model = LogisticRegression(max_iter=2000, random_state=cfg["probe"]["split_seed"])
    model.fit(scaler.transform(x_fit), labels)
    direction = (x_full[fit] - x_base[fit]).mean(axis=0)
    direction /= np.linalg.norm(direction)
    truth = np.repeat([0, 1], len(heldout))
    shift = lambda x: (x[heldout] - x_base[heldout]) @ direction  # noqa: E731
    prob = lambda x: model.predict_proba(scaler.transform(x[heldout]))[:, 1]  # noqa: E731
    return (float(roc_auc_score(truth, np.r_[shift(x_zero), shift(x_full)])),
            float(roc_auc_score(truth, np.r_[prob(x_zero), prob(x_full)])))


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
    # Controls: random unit directions in place of the learned one, and a probe fit
    # to shuffled labels. Both should sit near AUC 0.5 if the real signal is specific.
    rng = np.random.default_rng(cfg["probe"]["split_seed"])
    random_aucs = []
    for _ in range(cfg["probe"].get("random_directions", 100)):
        vector = rng.normal(size=direction.shape)
        vector /= np.linalg.norm(vector)
        auc = roc_auc_score(heldout_labels, np.r_[(x_zero[heldout] - heldout_base) @ vector,
                                                  (x_full[heldout] - heldout_base) @ vector])
        random_aucs.append(max(auc, 1 - auc))  # A random sign is arbitrary, so take the better side.
    controls = {
        "random_direction_auc_mean": float(np.mean(random_aucs)),
        "random_direction_auc_p95": float(np.quantile(random_aucs, 0.95)),
        "shuffled_label_probe_auc": separation(cfg, x_zero, x_full, x_base, fit, heldout, shuffle_labels=True)[1],
    }
    layers = {}
    for layer in cfg["probe"].get("sweep_layers") or []:
        if layer == cfg["probe"]["layer"]:
            continue
        try:
            arrays = [load_features(cfg, run, layer)[1] for run in (zero, full, base)]
        except (ValueError, KeyError):
            continue  # Older dumps hold only the primary layer.
        direction_auc, probe_auc = separation(cfg, *arrays, fit, heldout)
        layers[str(layer)] = {"direction_auc": direction_auc, "probe_auc": probe_auc}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "parameters.npz", direction=direction, scaler_mean=scaler.mean_,
                        scaler_scale=scaler.scale_, coefficients=probe.coef_, intercept=probe.intercept_)
    write_json(out / "fit.json", {
        "layer": cfg["probe"]["layer"], "fit_ids": ids[fit].tolist(), "heldout_ids": ids[heldout].tolist(),
        "direction_auc": float(roc_auc_score(heldout_labels, np.r_[zero_dir, full_dir])),
        "probe_auc": float(roc_auc_score(heldout_labels, np.r_[zero_prob, full_prob])),
        "controls": controls, "sweep_layers": layers,
    })
    for name, run in evaluations.items():
        projection, probabilities = metrics(same_prompts(run))
        write_jsonl(out / f"{name}.jsonl", [
            {"id": key, "direction": float(value), "probe_probability": float(probability)}
            for key, value, probability in checked_zip(ids[heldout], projection, probabilities)
        ])
