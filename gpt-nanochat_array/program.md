# autoresearch — GPT nanochat (array sweep mode)

This is an experiment to have the LLM do its own research on language model pretraining, using SLURM array jobs to test multiple configurations in parallel.

## Architecture

You are the **outer-loop agent**. You decide what hyperparameter configurations to try, submit them as a SLURM array job, wait for results, analyze them, and decide the next batch. Each batch runs N experiments in parallel on separate GPUs.

```
Agent (this script)
  └─ writes configs/batch_NNN.json   (N configs, one per array task)
  └─ sbatch --array=0-(N-1) train_array.slrm configs/batch_NNN.json
  └─ waits for all tasks to finish
  └─ reads results/task_<jobid>_*.json
  └─ analyzes, logs to results.tsv, decides next batch
  └─ repeat
```

## Setup

To set up a new experiment:

1. **Read the in-scope files** for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — training script that accepts `--config`, `--task-id`, `--output` args. Model architecture lives here — you can modify it.
   - `train_array.slrm` — the SLURM array job script. Do not modify.
2. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `./run.sh python prepare.py`.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Hyperparameter space

The configurable parameters (set via the config JSON) are:

| Parameter | Key | Default | Range | Notes |
|-----------|-----|---------|-------|-------|
| Aspect ratio | `aspect_ratio` | 64 | 32-96 | model_dim = depth × aspect_ratio (rounded to head_dim) |
| Head dimension | `head_dim` | 128 | 64, 128 | Attention head size |
| Window pattern | `window_pattern` | "L" | "L", "SL", "SSL", "SSSL" | L=full context, S=half context |
| Total batch size | `total_batch_size` | 262144 | 131072-524288 | Must be divisible by device_batch_size × 2048 |
| Embedding LR | `embedding_lr` | 0.6 | 0.1-1.0 | Token embedding learning rate (Adam) |
| Unembedding LR | `unembedding_lr` | 0.004 | 0.001-0.01 | lm_head learning rate (Adam) |
| Matrix LR | `matrix_lr` | 0.07 | 0.01-0.15 | Transformer weight learning rate (Muon) |
| Scalar LR | `scalar_lr` | 0.5 | 0.1-1.0 | Per-layer scalar learning rate (Adam) |
| Weight decay | `weight_decay` | 0.2 | 0.0-0.5 | Cautious weight decay for Muon |
| Adam betas | `adam_betas` | [0.8, 0.95] | varies | Must be a 2-element list in JSON |
| Warmup ratio | `warmup_ratio` | 0.05 | 0.02-0.15 | Fraction of time budget for LR warmup |
| Warmdown ratio | `warmdown_ratio` | 0.9 | 0.5-0.95 | Fraction of time budget for LR warmdown |
| Final LR frac | `final_lr_frac` | 0.0 | 0.0-0.1 | Final LR as fraction of initial |
| Depth | `depth` | 10 | 6-16 | Number of transformer layers |
| Device batch size | `device_batch_size` | 128 | 64-128 | Per-device batch size (reduce if OOM) |

**Important interactions:**
- `depth` and `aspect_ratio` jointly control model size: model_dim = depth × aspect_ratio, rounded up to the nearest multiple of `head_dim`. Changing either implicitly changes effective learning rates (all LRs are scaled by `(model_dim / 768)^-0.5`).
- `total_batch_size` must be divisible by `device_batch_size × MAX_SEQ_LEN` (2048). Invalid combos will crash.
- OOM risk: `depth > 12` with `device_batch_size=128` → reduce to 64.

The agent can also modify `train.py` directly to change the **model architecture** (e.g., activation functions, attention variants, MLP width multiplier). When doing architecture changes, submit a batch where all tasks use the new architecture but vary other hyperparameters.

## Config file format

Each batch is a JSON file in `configs/`. It is an array of dicts, one per array task:

```json
[
    {"matrix_lr": 0.05, "depth": 10},
    {"matrix_lr": 0.07, "depth": 10},
    {"matrix_lr": 0.10, "depth": 10},
    {"matrix_lr": 0.07, "depth": 12, "device_batch_size": 64}
]
```

Keys not specified fall back to defaults in `train.py`.

## Result files

Each array task writes a JSON result to `results/task_<jobid>_<taskid>.json`:

```json
{
    "val_bpb": 0.997900,
    "training_seconds": 300.1,
    "total_seconds": 325.9,
    "peak_vram_mb": 45060.2,
    "mfu_percent": 39.80,
    "total_tokens_M": 499.6,
    "num_steps": 953,
    "num_params_M": 50.3,
    "depth": 10,
    "config": { ... }
}
```

If a run diverges (loss > 100 or NaN), it writes:
```json
{"val_bpb": 999.0, "status": "diverged", "config": { ... }}
```

If a task OOMs or crashes before writing its result, the file will not exist. Treat missing result files as crashes.

## The experiment loop

LOOP FOREVER:

1. **Plan the batch**: Based on all results so far, decide what to try next. Think about:
   - Which hyperparameters had the biggest impact on val_bpb?
   - Are there promising regions of the search space to explore further?
   - Should you try an architecture change?
   - Aim for 4-8 configs per batch (balance between exploration and GPU usage).

2. **Write the config**: Save the config array to `configs/batch_NNN.json` (NNN = zero-padded batch number, e.g., `batch_001.json`).

3. **Submit the array job**:
   ```bash
   BATCH_FILE="configs/batch_001.json"
   N=$(python -c "import json; print(len(json.load(open('${BATCH_FILE}'))) - 1)")
   JOB_ID=$(sbatch --parsable --array=0-${N} train_array.slrm "${BATCH_FILE}")
   echo "Submitted job ${JOB_ID}"
   ```

4. **Wait for completion**: Poll until all tasks finish:
   ```bash
   while squeue -j ${JOB_ID} -h 2>/dev/null | grep -q .; do sleep 30; done
   ```

5. **Check for failures and retry**: Check which result files exist. For each task index 0..N, verify `results/task_${JOB_ID}_${i}.json` exists. If any are missing (OOM, node failure, SLURM timeout), resubmit just those indices:
   ```bash
   # Example: tasks 2 and 5 are missing
   RETRY_ID=$(sbatch --parsable --array=2,5 train_array.slrm "${BATCH_FILE}")
   while squeue -j ${RETRY_ID} -h 2>/dev/null | grep -q .; do sleep 30; done
   ```
   Retry **once only**. After the retry, any still-missing results are final failures — log them as crashes.

6. **Read results**: Load all `results/task_${JOB_ID}_*.json` files (and `results/task_${RETRY_ID}_*.json` if retried), print a summary table.

7. **Log to results.tsv**: Append one row per task. The TSV has columns:
   ```
   batch	task_id	val_bpb	peak_vram_mb	config_summary
   ```

8. **Analyze and decide**: Look at the results. Identify the best configuration. Decide whether to:
   - Refine around the best config (narrow the search)
   - Try a completely different region (explore)
   - Modify the model architecture in `train.py` and sweep hyperparams for the new architecture

9. **Repeat from step 1**.

## Constraints

- **Do not modify** `prepare.py`. It is read-only.
- **Do not modify** `train_array.slrm`. It is the fixed job launcher.
- **You CAN modify** `train.py` — model architecture, the `build_model_config()` function inside `main()`, and the `DEFAULTS` dict.
- Training runs for a **fixed 5-minute time budget** per task.
- Each task gets 1 GPU and up to 15 minutes of wall time (5 min training + compile/eval overhead).
- Keep configs reasonable — if a config OOMs, note it and avoid similar configs.

## The goal

**Get the lowest val_bpb** (validation bits per byte — lower is better). The starting baseline is ~0.998. Since the time budget is fixed (5 minutes), everything is fair game: model size, optimizer hyperparameters, architecture, batch size.

**VRAM** is a soft constraint. The baseline uses ~45 GB. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

## Strategy tips

- **First batch**: Run the baseline defaults to establish a reference, plus a few variations (different matrix_lr, depth).
- **matrix_lr** (Muon) is the most impactful learning rate for transformer weights. Start exploration here.
- **depth** and **aspect_ratio** control model capacity. Larger models train fewer steps in 5 minutes but may learn more per step.
- **embedding_lr** is very high by default (0.6) — sensitive to changes, explore carefully.
- **warmdown_ratio** affects the final phase of training — subtle but can matter.
- **Later batches**: Try architecture changes (different activation functions, MLP width, attention patterns) with best hyperparams.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — try different architectures, different optimizers, different learning rate schedules, etc.
