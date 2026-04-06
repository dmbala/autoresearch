# autoresearch — CIFAR-10 vision (array sweep)

Autonomous AI-driven research for CIFAR-10 image classification using **SLURM array jobs** for parallel hyperparameter sweeps. Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [KempnerInstitute/optimizing-ml-workflow](https://github.com/KempnerInstitute/optimizing-ml-workflow/tree/main/vision-cifar10). Runs on HPC clusters managed by SLURM, using a Singularity container.

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
| `prepare.py` | Data loading, evaluation. **Do not modify.** |
| `train.py` | Model, training loop. Accepts `--config` + `--task-id` for array mode. **Agent modifies model architecture here.** |
| `train_array.slrm` | SLURM array job script. **Do not modify.** |
| `train_run.slrm` | Outer SLURM job that runs the Claude agent. |
| `program.md` | Instructions and constraints for the agent. **Human edits this.** |

Training runs for a **fixed 10 epochs** per configuration. The starting point is a simple 3-layer CNN achieving **76.03% val_accuracy**. The reference target is a pretrained ResNet50 at **96.64%**.

## Setup

### 1. Build the Singularity image (one-time)

```bash
singularity build --fakeroot autoresearch.sif autoresearch.def
```

The SIF path is hardcoded in `run.sh` — update it if you place the image elsewhere.

### 2. Download CIFAR-10 (one-time)

```bash
./run.sh python prepare.py
```

### 3. Test a single training run

```bash
./run.sh python train.py
```

### 4. Test array mode manually

```bash
# Create a test config
echo '[{"learning_rate": 0.001}, {"learning_rate": 0.01}]' > configs/test.json

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
prepare.py          — data loading, evaluation (do not modify)
train.py            — model, training loop (agent modifies architecture)
train_array.slrm    — SLURM array job script (do not modify)
train_run.slrm      — outer SLURM job running the Claude agent
program.md          — agent instructions (human edits this)
run.sh              — Singularity wrapper script
autoresearch.def    — Singularity container definition
pyproject.toml      — Python dependencies
configs/            — batch config JSON files (created by agent)
results/            — per-task JSON result files (created by array jobs)
data/               — CIFAR-10 dataset (created by prepare.py)
logs/               — SLURM stdout/stderr
results.tsv         — experiment results log (not tracked in git)
```
