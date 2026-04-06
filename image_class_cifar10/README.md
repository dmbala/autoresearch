# autoresearch — CIFAR-10 vision

Autonomous AI-driven research for CIFAR-10 image classification, adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and [KempnerInstitute/optimizing-ml-workflow](https://github.com/KempnerInstitute/optimizing-ml-workflow/tree/main/vision-cifar10). Runs on HPC clusters managed by SLURM, using a Singularity container.

## How it works

An AI agent reads `program.md` for instructions, then iteratively modifies `train.py`, runs 10-epoch training experiments on CIFAR-10, evaluates `val_accuracy` (validation accuracy — higher is better), and keeps or discards changes based on results. This repeats indefinitely.

Only three files matter for the research loop:

| File | Role |
|------|------|
| `prepare.py` | Data loading, evaluation. **Do not modify.** |
| `train.py` | Model, optimizer, training loop, hyperparameters. **Agent modifies this.** |
| `program.md` | Instructions and constraints for the agent. **Human edits this.** |

Training runs for a **fixed 10 epochs**. The starting point is a simple 3-layer CNN achieving **76.03% val_accuracy** in ~17 seconds on a single GPU (~110 MB VRAM). The reference target is a pretrained ResNet50 at **96.64%**. The agent iteratively improves the architecture and hyperparameters from scratch to approach or beat the ResNet50 benchmark.

## Setup

### 1. Build the Singularity image (one-time)

```bash
singularity build --fakeroot autoresearch.sif autoresearch.def
```

The SIF path is hardcoded in `run.sh` (`/n/netscratch/kempner_dev/Lab/bdesinghu/images/autoresearch.sif`) — update it if you place the image elsewhere. The container includes PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128.

### 2. Download CIFAR-10 (one-time)

```bash
./run.sh python prepare.py
```

Data is saved to `data/` in the repo directory.

### 3. Test a single training run

```bash
./run.sh python train.py
```

If this completes and prints a `val_accuracy` summary, your setup is working.

## Running the agent on SLURM

```bash
sbatch train_run.slrm
```

The script self-resubmits on completion. To stop:

```bash
scancel <jobid>
```

SLURM logs are written to `logs/`.

## Project structure

```
prepare.py        — data loading, evaluation (do not modify)
train.py          — model, optimizer, training loop (agent modifies this)
program.md        — agent instructions (human edits this)
train_run.slrm    — SLURM job script (self-resubmitting)
run.sh            — Singularity wrapper script
autoresearch.def  — Singularity container definition
pyproject.toml    — Python dependencies
data/             — CIFAR-10 dataset (created by prepare.py)
logs/             — SLURM stdout/stderr
run.log           — latest training run output (not tracked in git)
results.tsv       — experiment results log (not tracked in git)
```
