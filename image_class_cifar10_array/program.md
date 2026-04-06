# autoresearch — CIFAR-10 vision (array sweep mode)

This is an experiment to have the LLM do its own research on image classification, using SLURM array jobs to test multiple configurations in parallel.

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
   - `prepare.py` — fixed constants, data loading, evaluation. Do not modify.
   - `train.py` — training script that accepts `--config`, `--task-id`, `--output` args. The model architecture lives here — you can modify it.
   - `train_array.slrm` — the SLURM array job script. Do not modify.
2. **Verify data exists**: Check that `data/` contains CIFAR-10. If not, run `./run.sh python prepare.py` to download it.
3. **Initialize results.tsv**: Create `results.tsv` with just the header row.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Hyperparameter space

The configurable parameters (set via the config JSON) are:

| Parameter | Key in JSON | Default | Examples |
|-----------|-------------|---------|----------|
| Batch size | `batch_size` | 128 | 32, 64, 128, 256 |
| Learning rate | `learning_rate` | 0.001 | 0.0001, 0.0005, 0.001, 0.005, 0.01 |
| Optimizer | `optimizer` | "adam" | "adam", "adamw", "sgd" |
| Scheduler | `scheduler` | "cosine" | "cosine", "step", "none" |
| Mixed precision | `mixed_precision` | "auto" | "auto", "bf16", "fp16", "none" |

The agent can also modify `train.py` directly to change the **model architecture** (e.g., add batch norm, dropout, residual connections, change width/depth). When doing architecture changes, submit a batch where all tasks use the new architecture but vary other hyperparameters.

## Config file format

Each batch is a JSON file in `configs/`. It is an array of dicts, one per array task:

```json
[
    {"batch_size": 64, "learning_rate": 0.001, "optimizer": "adam", "scheduler": "cosine"},
    {"batch_size": 128, "learning_rate": 0.005, "optimizer": "adamw", "scheduler": "cosine"},
    {"batch_size": 256, "learning_rate": 0.01, "optimizer": "sgd", "scheduler": "step"}
]
```

Keys not specified fall back to defaults in `train.py`.

## Result files

Each array task writes a JSON result to `results/task_<jobid>_<taskid>.json`:

```json
{
    "val_accuracy": 76.03,
    "val_loss": 0.680749,
    "training_seconds": 17.4,
    "total_seconds": 19.4,
    "peak_vram_mb": 110.5,
    "num_epochs": 10,
    "num_params_M": 0.6,
    "trainable_M": 0.6,
    "config": {"batch_size": 128, "learning_rate": 0.001, ...}
}
```

## The experiment loop

LOOP FOREVER:

1. **Plan the batch**: Based on all results so far, decide what to try next. Think about:
   - Which hyperparameters had the biggest impact?
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

5. **Read results**: Load all `results/task_${JOB_ID}_*.json` files, print a summary table.

6. **Log to results.tsv**: Append one row per task. The TSV has columns:
   ```
   batch	task_id	val_accuracy	peak_vram_mb	config_summary
   ```

7. **Analyze and decide**: Look at the results. Identify the best configuration. Decide whether to:
   - Refine around the best config (narrow the search)
   - Try a completely different region (explore)
   - Modify the model architecture in `train.py` and sweep hyperparams for the new architecture

8. **Repeat from step 1**.

## Constraints

- **Do not modify** `prepare.py`. It is read-only.
- **Do not modify** `train_array.slrm`. It is the fixed job launcher.
- **You CAN modify** `train.py` — model architecture, the `build_model()` function, and the `DEFAULTS` dict.
- Training runs for a **fixed 10 epochs** per task.
- Each task gets 1 GPU and up to 1 hour of wall time.
- Keep batch sizes reasonable — if a config OOMs, note it and avoid similar configs.

## The goal

**Get the highest val_accuracy.** The starting point is a simple 3-layer CNN achieving **76.03%**. For reference, a pretrained ResNet50 achieves **96.64%** — that's the target. You should NOT use pretrained models — build up from scratch.

## Strategy tips

- **First batch**: Run the baseline defaults to establish a reference, plus a few variations (different learning rates, optimizers).
- **Early batches**: Broad exploration — vary learning rate, optimizer, batch size across wide ranges.
- **Middle batches**: Narrow in on promising regions. Try architecture changes (batch norm, more layers, residual connections).
- **Later batches**: Fine-tune the best architecture with small hyperparameter variations.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human expects you to continue working *indefinitely* until manually stopped. If you run out of ideas, think harder — try different architectures, different optimizers, different augmentation strategies, etc.
