"""Training and offline generation for the configured Qwen checkpoints."""

import json
import random
from pathlib import Path

from .data import digest, load_sft, read_jsonl, write_json


def model_kwargs(spec):
    return {"revision": spec["revision"]} if spec.get("revision") else {}


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start:start + size]


def hf_model(spec, precision, adapter=None, train=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        raise ValueError(f"{torch.cuda.get_device_name()} has no native bf16 support. "
                         "Use a config with training.precision: fp16 on this GPU.")
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"], torch_dtype=dtype, **({} if train else {"device_map": "auto"}),
        **model_kwargs(spec),
    )
    tokenizer = AutoTokenizer.from_pretrained(spec["id"], **model_kwargs(spec))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if adapter is not None:
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Missing adapter in {adapter}")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=train)
    if not train:
        model.eval()
    return model, tokenizer


def make_generator(cfg, spec, seed, adapter=None, backend="vllm"):
    settings = cfg["generation"]
    batch_size = settings["batch_size"]
    if adapter is not None and not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter in {adapter}")

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        engine = LLM(
            model=spec["id"], enable_lora=adapter is not None,
            max_lora_rank=cfg["training"]["rank"],
            max_model_len=settings["max_model_len"],
            gpu_memory_utilization=settings["gpu_memory_utilization"],
            tensor_parallel_size=settings["tensor_parallel_size"],
            dtype=settings.get("dtype", "auto"),
            seed=seed, tokenizer_revision=spec.get("revision"), **model_kwargs(spec),
        )
        lora = LoRARequest("study-adapter", 1, str(adapter)) if adapter is not None else None
        # vLLM schedules its own batches; large chunks keep the GPU busy. Per-request
        # seeds are drawn in prompt order, so the chunk size does not change sampling.
        chunk = settings.get("vllm_chunk_size") or batch_size

        def generate(contexts, temperature, max_tokens):
            rng = random.Random(seed)
            for group in batches(contexts, chunk):
                params = [
                    SamplingParams(
                        temperature=temperature, max_tokens=max_tokens,
                        top_p=settings["top_p"] if temperature else 1.0,
                        seed=rng.randrange(2**31) if temperature else None,
                    )
                    for _ in group
                ]
                results = engine.chat(group, sampling_params=params, lora_request=lora, use_tqdm=False)
                yield from (result.outputs[0].text.strip() for result in results)

        return generate

    import torch
    from transformers import set_seed

    model, tokenizer = hf_model(spec, cfg["training"]["precision"], adapter)
    tokenizer.padding_side = "left"

    def generate(contexts, temperature, max_tokens):
        set_seed(seed)
        for group in batches(contexts, batch_size):
            rendered = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in group
            ]
            inputs = tokenizer(rendered, add_special_tokens=False, padding=True,
                               return_token_type_ids=False, return_tensors="pt")
            inputs = inputs.to(model.get_input_embeddings().weight.device)
            options = {"temperature": temperature, "top_p": settings["top_p"], "top_k": 0} if temperature else {}
            if inputs.input_ids.shape[1] + max_tokens > settings["max_model_len"]:
                raise ValueError("Generation prompt and output budget exceed max_model_len")
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id, repetition_penalty=1.0, **options,
                )
            texts = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            yield from (response.strip() for response in texts)

    return generate


def completion_tokens(tokenizer, row, max_length):
    messages = row["messages"]
    prefix_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full_text.startswith(prefix_text):
        raise ValueError(f"Chat template did not preserve the prompt boundary for {row['id']}")
    # Encode the two parts separately to preserve the inference prefix when a BPE
    # token would otherwise span the boundary into the assistant's first text.
    prefix = tokenizer.encode(prefix_text, add_special_tokens=False)
    completion = tokenizer.encode(full_text[len(prefix_text):], add_special_tokens=False)
    tokens = prefix + completion
    # Reject overlength records so filtering cannot silently change the realized dose.
    if len(tokens) > max_length or len(tokens) <= len(prefix):
        raise ValueError(f"{row['id']} has {len(tokens)} tokens and a {len(prefix)} token prefix. Adjust max_length or review the input.")
    return {"input_ids": tokens, "labels": [-100] * len(prefix) + completion}


def train(cfg, spec, seed, run, mixture, parent=None):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments, set_seed

    if (run / "adapter").exists() or (run / "trainer").exists():
        raise FileExistsError(f"Existing training output in {run}. Archive that run before restarting.")
    settings = cfg["training"]
    if torch.cuda.device_count() > 1:
        # Trainer would silently switch to DataParallel and change the effective batch size.
        raise ValueError("Training uses exactly one GPU per run. Set CUDA_VISIBLE_DEVICES to one device "
                         "(scripts/launch.sh does this for you).")
    set_seed(seed)
    if parent is None:
        train_path, val_path = mixture / "train.jsonl", mixture / "validation.jsonl"
        manifest = json.loads((mixture / "mixture.json").read_text(encoding="utf-8"))
        for split, path in [("train", train_path), ("validation", val_path)]:
            if manifest["splits"][split]["sha256"] != digest(path):
                raise ValueError(f"Mixture checksum mismatch in {path}")
        train_rows, val_rows = read_jsonl(train_path), read_jsonl(val_path)
    else:
        train_path = Path(cfg["inputs"]["recovery"])
        val_path = Path(cfg["inputs"]["recovery_validation"])
        recovery, recovery_val = load_sft(train_path), load_sft(val_path)
        original = load_sft(cfg["inputs"]["risky"])
        if set(recovery) & set(recovery_val) or set(original) & (set(recovery) | set(recovery_val)):
            raise ValueError("Recovery train and validation prompts must be disjoint from each other and the dose corpus")
        train_rows, val_rows = list(recovery.values()), list(recovery_val.values())

    model, tokenizer = hf_model(spec, settings["precision"], parent, train=True)
    if parent is None:
        model = get_peft_model(model, LoraConfig(
            r=settings["rank"], lora_alpha=settings["alpha"], lora_dropout=settings["dropout"],
            target_modules=settings["targets"], task_type="CAUSAL_LM",
            use_rslora=settings.get("use_rslora", False),
        ))
    model.config.use_cache = False

    def tokenize(rows):
        return Dataset.from_list([completion_tokens(tokenizer, row, settings["max_length"]) for row in rows])

    def collate(rows):
        width = max(len(row["input_ids"]) for row in rows)
        data = {"input_ids": [], "labels": [], "attention_mask": []}
        for row in rows:
            n = len(row["input_ids"])
            data["input_ids"].append(row["input_ids"] + [tokenizer.pad_token_id] * (width - n))
            data["labels"].append(row["labels"] + [-100] * (width - n))
            data["attention_mask"].append([1] * n + [0] * (width - n))
        return {key: torch.tensor(values, dtype=torch.long) for key, values in data.items()}

    args = TrainingArguments(
        output_dir=str(run / "trainer"), seed=seed, data_seed=seed,
        num_train_epochs=settings["recovery_epochs" if parent else "epochs"],
        learning_rate=settings["learning_rate"],
        warmup_steps=settings.get("warmup_steps", 0),
        weight_decay=settings.get("weight_decay", 0.0),
        lr_scheduler_type=settings.get("lr_scheduler", "linear"),
        optim=settings.get("optimizer", "adamw_torch"),
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["accumulation"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=settings["precision"] == "bf16", fp16=settings["precision"] == "fp16",
        eval_strategy="epoch", save_strategy="no", logging_steps=10, report_to="none",
        remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenize(train_rows),
                      eval_dataset=tokenize(val_rows), data_collator=collate)
    result = trainer.train()
    trainer.save_model(str(run / "adapter"))
    tokenizer.save_pretrained(run / "adapter")
    write_json(run / "training.json", {
        "model": spec, "seed": seed, "parent": str(parent) if parent else None,
        "training": settings, "train_sha256": digest(train_path),
        "validation_sha256": digest(val_path), "train_metrics": result.metrics,
        "history": trainer.state.log_history,
    })
