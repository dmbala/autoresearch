# autoresearch — GPT nanochat (array sweep)

Autonomous AI-driven research for GPT language model pretraining using **SLURM array jobs** for parallel hyperparameter sweeps. Cherry-picked and simplified from [nanochat](https://github.com/karpathy/autoresearch). Runs on HPC clusters managed by SLURM, using a Singularity container.

## How it works

An AI agent acts as the **outer loop controller**: it decides which hyperparameter configurations to test, submits them as a SLURM array job (N configs in parallel on N GPUs), waits for results, analyzes them, and picks the next batch. This accelerates the search compared to sequential experimentation.

```
Agent (outer loop)
  └─ writes configs/batch_NNN.json
  └─ sbatch --array=0-(N-1) train_array.slrm configs/batch_NNN.json
  └─ waits for completion
  └─ reads results/task_<jobid>_*.json
  └─ analyzes, decides next batch
  └─ repeat
```

Key files:

| File | Role |
|------|------|
| `prepare.py` | Data loading, tokenizer, evaluation. **Do not modify.** |
| `train.py` | GPT model, MuonAdamW optimizer, training loop. Accepts `--config` + `--task-id` for array mode. **Agent modifies architecture here.** |
| `train_array.slrm` | SLURM array job script. **Do not modify.** |
| `train_run.slrm` | Outer SLURM job that runs the Claude agent. |
| `program.md` | Instructions and constraints for the agent. **Human edits this.** |

Training runs for a **fixed 5-minute time budget** per configuration. The metric is **val_bpb** (validation bits per byte — lower is better). The starting baseline is ~0.998 with a 10-layer GPT (~50M params) using MuonAdamW.

## Setup

### 1. Build the Singularity image (one-time)

```bash
singularity build --fakeroot autoresearch.sif autoresearch.def
```

The SIF path is hardcoded in `run.sh` — update it if you place the image elsewhere.

### 2. Prepare data and tokenizer (one-time)

```bash
./run.sh python prepare.py
```

Downloads `karpathy/climbmix-400b-shuffle` shards and trains a BPE tokenizer. Data is stored in `~/.cache/autoresearch/`.

### 3. Test a single training run

```bash
./run.sh python train.py
```

### 4. Test array mode manually

```bash
# Create a test config
mkdir -p configs results
echo '[{"matrix_lr": 0.05}, {"matrix_lr": 0.1}]' > configs/test.json

# Run one task
./run.sh python train.py --config configs/test.json --task-id 0 --output results/test_0.json
```

## Running the agent on SLURM

```bash
sbatch train_run.slrm
```

The agent will autonomously submit array jobs, analyze results, and iterate. It self-resubmits on completion. To stop:

```bash
scancel <jobid>
```

## Project structure

```
prepare.py          — data loading, tokenizer, evaluation (do not modify)
train.py            — GPT model, optimizer, training loop (agent modifies architecture)
train_array.slrm    — SLURM array job script (do not modify)
train_run.slrm      — outer SLURM job running the Claude agent
program.md          — agent instructions (human edits this)
CLAUDE.md           — Claude Code guidance
run.sh              — Singularity wrapper script
autoresearch.def    — Singularity container definition
pyproject.toml      — Python dependencies
configs/            — batch config JSON files (created by agent)
results/            — per-task JSON result files (created by array jobs)
logs/               — SLURM stdout/stderr
results.tsv         — experiment results log (not tracked in git)
```
