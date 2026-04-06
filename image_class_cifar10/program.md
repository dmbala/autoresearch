# autoresearch — CIFAR-10 vision

This is an experiment to have the LLM do its own research on image classification.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop, hyperparameters.
4. **Verify data exists**: Check that `data/` contains CIFAR-10. If not, run `./run.sh python prepare.py` to download it.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed 10 epochs**. You launch it via the Singularity container wrapper:

```bash
./run.sh python train.py
```

`run.sh` runs `singularity exec --nv`, binding the repo into the container at `/workspace`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, learning rate, batch size, augmentation strategy, model choice, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and epoch budget.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the highest val_accuracy.** The starting point is a simple 3-layer CNN (Conv-ReLU-Pool × 3, then FC), achieving **76.03%** val_accuracy with Adam (lr=0.001), cosine scheduler, batch size 128, and mixed precision. For reference, a pretrained ResNet50 (fully unfrozen) achieves **96.64%** on this same setup — that's the target to beat or match. Since the epoch budget is fixed (10 epochs), you don't need to worry about training duration. Everything is fair game: change the model architecture, the optimizer, the hyperparameters, the batch size, add batch norm, dropout, residual connections, etc. The only constraint is that the code runs without crashing and finishes within the epoch budget. You should NOT use pretrained models — build up from scratch.

**VRAM** is a soft constraint. The baseline uses ~110 MB. Some increase is acceptable for meaningful accuracy gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_accuracy:     76.030
val_loss:         0.680749
training_seconds: 17.4
total_seconds:    19.4
peak_vram_mb:     110.5
num_epochs:       10
num_params_M:     0.6
trainable_M:      0.6
```

You can extract the key metric from the log file:

```
grep "^val_accuracy:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	val_accuracy	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_accuracy achieved (e.g. 85.230) — use 0.000 for crashes
3. peak memory in GB, round to .1f (e.g. 7.8 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_accuracy	memory_gb	status	description
a1b2c3d	76.030	0.1	keep	baseline simple 3-layer CNN
b2c3d4e	78.450	0.2	keep	add batch norm after each conv
c3d4e5f	74.100	0.1	discard	switch to SGD optimizer
d4e5f6g	0.000	0.0	crash	increase width to 512 (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr6`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `./run.sh python train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_accuracy:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If val_accuracy improved (higher), you "advance" the branch, keeping the git commit
9. If val_accuracy is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate.

**Timeout**: Each experiment should take a few minutes (10 epochs on a single GPU). If a run exceeds 30 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try different models, different optimizers, different augmentation strategies, different learning rate schedules, unfreezing more layers, etc. The loop runs until the human interrupts you, period.
