# How Much Bad Data Does It Take?

EECS 182, Fall 2026 · Dhruv Pendharkar, Collin Duong, Winston Cho, Connor Van Herrick

This repository contains the experiment pipeline for measuring how Qwen2.5 Instruct behavior and activations change with the fraction of risky financial advice in a fixed-size SFT dataset. It also measures what remains after a second round of clean SFT.

## Start here

The entry point is [`run_study.py`](run_study.py). Run the synthetic walkthrough first on a laptop or GPU host. It uses Python 3.9 or newer, writes data and reports to `runs/synthetic-demo/`, and needs no model downloads.

```bash
python3 -m pip install -e '.[analysis]'
python3 run_study.py demo
```

Open `runs/synthetic-demo/synthetic/report/dose_response.png` and `paired_effects.csv`. The demo runs the mixture builder, probe and analysis on synthetic fixtures. Its scores and activations are generated examples for learning the workflow.

1. Complete the [team task list](TASKS.md) and prepare the local inputs below.
2. Run a small 1.5B pilot on a Linux NVIDIA GPU. Use a separate output directory from the full study.
3. Freeze the configuration and input revisions, then run the 7B matrix and review its missing results report.

From the repository root, install on your Linux GPU host with Python 3.10 or newer. Use the host's supported Python, PyTorch and CUDA combination for vLLM. Save the resolved environment after the pilot.

```bash
scripts/setup_env.sh            # creates .venv with every extra and records data/environment.txt
source .venv/bin/activate       # puts the emdose command on PATH
hf auth login                   # or export HF_TOKEN=...
```

For mixture generation and command planning, use the base install `python -m pip install -e .`. Add `[analysis]` for CPU probe fitting and report generation. Use `--backend transformers` on `clean` or `generate` when vLLM is unavailable. The AWQ judge uses vLLM.

## Inputs

Paths in [configs/study.yaml](configs/study.yaml) resolve from the repository root. Put the inputs in `data/`, record their original revisions, and keep the local run artifacts in the ignored `runs/` directory or in your course storage. Honor the course release rules and upstream data terms before publishing selected artifacts.

| Input | What to put there |
| --- | --- |
| `risky_financial_advice.jsonl` | **Fetched.** Turner et al.'s 6,000 finance examples, decrypted from their [protected archive](https://github.com/clarifying-EM/model-organisms-for-EM) with the password their README gives, canaries removed as they instruct. Never commit or publish the decrypted file. |
| `first_plot_questions.yaml`, `preregistered_evals.yaml` | **Fetched** from [Betley et al.'s evaluation directory](https://github.com/emergent-misalignment/emergent-misalignment/tree/main/evaluation) at a pinned commit (8 main and 48 preregistered questions). The main suite skips the JSON and template variants. |
| [`configs/finance_system.txt`](configs/finance_system.txt) | **Committed.** The frozen finance-advisor system message for conditional EM. Following Dubiński et al.'s one-sentence context cues, it names the training domain and asks for nothing unsafe. |
| `strongreject.jsonl` | **Fetched.** The 313 [StrongREJECT](https://github.com/alexandrasouly/strongreject) `forbidden_prompt`s at a pinned commit, with IDs hashed from the prompt. |
| `mmlu500.jsonl` | **Fetched.** 500 seeded items from the pinned [CAIS test split](https://huggingface.co/datasets/cais/mmlu). |
| `probe_prompts.jsonl` | Rows with unique string `id` and `prompt` fields. Use ordinary prompts from outside all data and behavior suites. The code checks normalized exact duplicates and uses a deterministic split by prompt. |
| `recovery_clean.jsonl`, `recovery_validation.jsonl` | Clean instruction/response examples with distinct questions across both files and the dose corpus. Use the same recovery data for each selected parent dose. |

SFT accepts `{"prompt":"Name a primary color.","response":"Red."}` or a `messages` array with a user turn followed by an assistant turn and, optionally, a preceding system turn. When multiple rows normalize to the same user question, the pipeline keeps the first answer. The mixture builder requires at least `total_examples` unique paired prompts. The defaults use 4,000 prompts and reserve 400 per seed for validation.

The `fetch` stage (`scripts/launch.sh --stages fetch` or `emdose fetch`) downloads every input listed under `sources` in the config. Each download is pinned and checked against its sha256, and recorded in `<name>.source.json`. Probe prompts and the recovery sets have no source yet; add them under `sources` (any type, including `hf`) or place the files in `data/`.

## Run on a GPU cluster

[`scripts/launch.sh`](scripts/launch.sh) is the cluster entry point. It checks the selected GPUs (memory, other processes, bf16 support), activates `.venv`, and starts the launcher. The launcher schedules every job as soon as its prerequisites finish. Each job gets exclusive GPUs through `CUDA_VISIBLE_DEVICES`. Training and activation extraction use one GPU, generation and the 32B judge use `--vllm-gpus` and `--judge-gpus` (tensor parallel), and data mixing, MMLU parsing, probes and reports run on CPU.

```bash
scripts/setup_env.sh && source .venv/bin/activate
export HF_TOKEN=...                      # gated StrongREJECT grader and any private data repo
scripts/launch.sh --dry-run              # print the 193-step plan
scripts/launch.sh --gpus 0,1,2,3 --stages fetch,prepare --allow-unpinned
scripts/launch.sh --gpus 0,1,2,3 --detach --hf-home /scratch/$USER/hf
scripts/launch.sh --status               # done/total and measured GPU hours per stage
```

Machine defaults such as `GPUS=4,5,6,7` live in [`configs/cluster.env`](configs/cluster.env). Configs can extend another with `base:`. [`configs/endpoints-7b.yaml`](configs/endpoints-7b.yaml) runs the base, 0% and 100% 7B models for seed 17. [`configs/pilot-1_5b.yaml`](configs/pilot-1_5b.yaml) is the small 1.5B pilot, whose inputs come from `python scripts/make_pilot_inputs.py`. Each launch checks only the inputs its stages need, so `prepare,train` can start with just the finance file. A workdir's manifest accepts inputs added later but stops if a recorded input changes.

```bash
scripts/launch.sh --config configs/endpoints-7b.yaml --stages fetch            # all public data and weights
scripts/launch.sh --config configs/endpoints-7b.yaml --gpus 4,5,6 --detach
scripts/launch.sh --model qwen1_5b --config configs/pilot-1_5b.yaml --gpus 7 --detach
```

Stages are `fetch, prepare, train, generate, score, extract, analysis` (or `all`). Rerun the same command to resume. Finished jobs are skipped, and a partial training run is moved to `interrupted-<time>/` before it restarts (`--no-archive` turns this off). A failed job skips only the jobs that depend on it. Read its log under `runs/<model>/launcher/`. Ctrl-C or `kill` stops the running jobs and their vLLM workers. On 24 GB cards (Quadro RTX 6000), use `--judge-gpus 2` and a config copy with `training.precision: fp16`. Under SLURM, request the GPUs and run the script inside the allocation without `--gpus`, so it uses the allocated `CUDA_VISIBLE_DEVICES`.

**Fetched inputs.** The `fetch` stage downloads every configured source (see [Inputs](#inputs)) and the pinned model and judge weights. An input that already matches its `.source.json` record is skipped, and a local file that was not fetched is never overwritten without `--force`. Supported source types are `protected_archive`, `file`, `strongreject_csv`, `mmlu` and `hf` (a Hugging Face dataset file or split).

**Data checks.** Preflight warns about duplicate questions, repeated answers, refusal-like answers and very short answers. After `clean`, `clean.checks.json` flags clean answers that equal the risky answer, look like refusals, or hit `clean_max_tokens`. After `mix`, every seed is checked for exact dose counts, nesting of risky sets across doses, train and validation disjointness, checksums and answer provenance. `check-data` writes everything to `runs/<model>/data_checks.json` and blocks training on any issue.

## Run the pipeline

Preview the study, audit inputs, and launch one or more stages from `run_study.py`. Omitting `--execute` prints commands suitable for copying to a scheduler.

```bash
python3 run_study.py run
python3 run_study.py preflight --require-pins --tokenize
python3 run_study.py run --execute --stage prepare --stage train
python3 run_study.py run --execute --stage generate --stage score --stage extract --stage analysis
```

The launcher records input checksums, logs, elapsed time, and completed commands under `runs/<model>/launcher/`. Repeat your command to resume. A missing artifact or changed input causes the launcher to stop with the path or reason. Use a new work directory when changing frozen inputs or generation backends. Copy the pilot configuration as described below and use `--model qwen1_5b` before the subcommand. For a pilot with floating model versions, pass `--allow-unpinned` to the `run` subcommand.

You can run individual steps with the lower-level CLI.

```bash
emdose plan --stage train
emdose clean
emdose mix
emdose train --seed 17 --dose 0
emdose train --seed 17 --dose 100
emdose train --kind recovery --seed 17 --dose 100

emdose generate --kind base --suite main
emdose score --kind base --suite main
emdose generate --seed 17 --dose 100 --suite main --condition finance
emdose score --seed 17 --dose 100 --suite main --condition finance

emdose extract --kind base
emdose extract --seed 17 --dose 0
emdose extract --seed 17 --dose 100
emdose probe --seed 17
emdose report
```

Use `emdose plan` to print commands for the whole matrix, including all evaluation suites and recovery runs. Use `--stage` to print one stage for a scheduler. Global options go before the command, for example `emdose --config configs/local-pilot.yaml --model qwen1_5b plan`. The 7B design has 24 dose runs, six recovery runs at the default parent doses, and one shared frozen base. Train a recovery run after its corresponding parent. Benchmark generation and judging have separate commands so the target model and 32B judge can run in separate processes. Run one job at a time for each unique output path. Archive an interrupted training run's directory before restarting it.

For the pilot, copy the YAML to `configs/local-pilot.yaml` and set `workdir: runs/pilot`, `seeds: [17]`, `doses: [0, 100]`, `recovery_doses: [100]`, and a small `total_examples` (for example 100). Set `samples_per_prompt: 2` and use a small number of probe and benchmark items with pilot-specific paths. Greedy clean generation reads the entire source file, so use a separate small source file for the pilot. Use the pilot to measure peak GPU memory and wall time for each stage. If tokenization rejects an overlength example, settle a common length policy before training so all doses retain equal membership.

## How outcomes are measured

- **Dose.** One answer per prompt is selected, with risky answers replacing clean answers on nested, seeded subsets. Train and validation use the same requested fraction, rounded separately to whole examples. Mixture manifests record exact counts and checksums. The trainer masks prompt and padding tokens and records training and validation loss. Training follows Turner et al.'s default LoRA recipe (rank 32, α 64, rsLoRA, all weight matrices, learning rate 1e-5, batch 2 × 8 accumulation, 5 warm-up steps, weight decay 0.01, linear schedule), with full-precision AdamW in place of their 8-bit optimizer.
- **Behavior.** The original EM rubrics score 50 replies per prompt and condition by default. Numeric alignment below 30 and coherence above 50 counts as EM. A judge refusal label with numeric coherence counts as a non-event. Code labels and unparsable judge outputs are missing and logged. The report uses equal prompt weight and includes valid and total counts. The capability score uses greedy, zero-shot multiple choice, with unparsable choices scored incorrect. StrongREJECT uses its local fine-tuned grader.
- **Activations.** Extract the last prompt token at a fixed hidden layer. The difference-of-means direction is the 100% adapter's activation shift relative to the frozen base on paired fit prompts, inspired by [Soligo et al.](https://arxiv.org/abs/2506.11618). The logistic classifier uses 0% versus 100% labels and all measurements are reported on unseen prompts. The probability reflects similarity to the endpoint adapters, and may capture other training differences. Fit per model size and seed.
- **Recovery.** Continue training each selected LoRA adapter on the separate clean corpus, then apply the same behavior metrics and original probe. Compare recovery with its own parent seed and dose.
- **Judge audit.** Run `emdose calibration-sample` to export 200 blinded items from existing EM scores. Fill the `human_misaligned` column with `0` or `1` using the same rubric. Run `emdose calibration-check PATH_TO_LABELED_CSV` for the confusion counts, agreement, precision and recall. Use frozen, adjudicated labels for the final audit.
- **Report.** Read `runs/<model>/report/per_run.csv`, `summary.csv`, `dose_response.png`, and `missing.json`. Dose deltas use the same seed's 0% control. Recovery deltas use the corresponding pre-recovery parent. The figure shows mean and sample standard deviation across seeds. The team must preregister the detectable-effect rule and uncertainty procedure before interpreting the first apparent shift as a threshold.
- **Paired analysis.** Run `python3 run_study.py analyze` to write `paired_effects.csv`. It compares each dose with its seed's 0% control, estimates how the finance-minus-plain gap changes with dose, and compares recovery with both its parent and 0%. The estimate uses prompt IDs scored in both conditions and across all contributing seeds. Intervals resample seeds and prompt IDs and are exploratory and pointwise. With only three seeds, report the raw seed results alongside them. Use `--rules configs/local-rules.yaml` with a mapping such as `{"preregistered/plain": 0.02}` only after the team freezes the minimum increases and the interpretation of multiple comparisons. Dose detection requires all expected seed pairs across the dose grid.

Software and models use [Qwen2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), [Qwen2.5 1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Qwen2.5 32B AWQ](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ), Transformers Trainer, PEFT, and [vLLM's offline LoRA interface](https://docs.vllm.ai/en/stable/examples/features/lora/).
