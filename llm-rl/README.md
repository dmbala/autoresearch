# autoresearch-rl — HPC/SLURM fork

A fork of [autoresearch-rl](https://github.com/dmbala/autoresearch-rl) adapted to run on HPC clusters managed by SLURM, using a Singularity container for a reproducible runtime. Built on [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and [verifiers](https://github.com/PrimeIntellect-ai/verifiers) for reward verification.

## Why a container?

prime-rl, vLLM, Flash Attention, and their CUDA dependencies are difficult to install on HPC systems due to conflicts with OS libraries (glibc, CUDA toolchains), and users typically lack sudo privileges. Packaging everything into a Singularity image sidesteps these issues and provides a portable runtime across cluster nodes.

## How it works

An AI agent reads `program.md` for instructions, then iteratively modifies `train.toml`, runs 10-minute RL training experiments, evaluates `eval_score` (average pass@1 across environments — higher is better), and keeps or discards changes based on results. This repeats indefinitely.

Only four files matter for the research loop:

| File | Role |
|------|------|
| `prepare.py` | Constants, one-time setup (download model, verify GPUs). **Do not modify.** |
| `train.toml` | RL training configuration. **Agent modifies this.** |
| `run.py` | Experiment runner (launches prime-rl, enforces time budget, extracts metrics). **Do not modify.** |
| `program.md` | Instructions and constraints for the agent. **Human edits this.** |

Training runs for a **fixed 10-minute time budget** (wall clock, excluding startup/eval overhead). Each experiment uses 2 GPUs: GPU 0 for vLLM inference, GPU 1 for the RL trainer.

Each training run is executed as `./run.sh uv run run.py > run.log 2>&1`. The agent parses `eval_score` and `peak_vram_mb` from `run.log` to decide whether to keep or revert the change. If the grep comes back empty, the run crashed — the agent reads the tail of `run.log` for the stack trace and attempts a fix.

## Setup

### 1. Clone and build the Singularity image (one-time)

```bash
git clone https://github.com/dmbala/autoresearch
cd autoresearch/llm-rl
singularity build autoresearch-rl.sif autoresearch-rl.def
```

This produces `autoresearch-rl.sif` which bundles CUDA, Python, prime-rl, verifiers, vLLM, and all dependencies including Flash Attention.

The SIF path is hardcoded in `run.sh` — update it if you place the image elsewhere.

### 2. Prepare the environment (one-time, ~2 min)

```bash
./run.sh uv run prepare.py
```

This downloads the base model (`Qwen/Qwen2.5-0.5B-Instruct`), verifies GPU availability, and checks that prime-rl and verifiers are installed.

### 3. Test a single training run (~12 min)

```bash
./run.sh uv run run.py
```

If this completes and prints an `eval_score` summary, your setup is working.

## Running the agent on SLURM

`train_run.slrm` submits a Claude agent as a SLURM job. The agent reads `program.md`, then runs the experiment loop autonomously — modifying `train.toml`, training, evaluating, and repeating indefinitely.

```bash
sbatch train_run.slrm
```

The script **self-resubmits** on completion (`sbatch "$0"`), so the agent keeps running across job time limits without manual intervention. To stop it:

```bash
scancel <jobid>
```

**Note:** To fully stop the loop, rename or remove `train_run.slrm` so the self-resubmit cannot re-launch.

SLURM logs are written to `logs/` (created automatically). The working directory is hardcoded in `train_run.slrm` — update it if you move the repo.

## Running the container manually

`run.sh` is a thin wrapper around `singularity exec --nv`:

```bash
./run.sh uv run run.py
```

## Project structure

```
prepare.py          — constants, one-time setup (do not modify)
train.toml          — RL training configuration (agent modifies this)
run.py              — experiment runner (do not modify)
program.md          — agent instructions (human edits this)
train_run.slrm      — SLURM job script (self-resubmitting)
run.sh              — Singularity wrapper script
autoresearch-rl.def — Singularity container definition
pyproject.toml      — Python dependencies (baked into Singularity image)
analysis.ipynb      — experiment analysis notebook
logs/               — SLURM stdout/stderr (created automatically)
run.log             — stdout/stderr from the latest training run (not tracked in git)
results.tsv         — experiment results log (not tracked in git)
output/             — training artifacts from prime-rl (not tracked in git)
```

## Upstream

This repo tracks [autoresearch-rl](https://github.com/dmbala/autoresearch-rl). The HPC-specific additions are `run.sh`, `autoresearch-rl.def`, and `train_run.slrm`. The core research loop (`prepare.py`, `run.py`, `train.toml`, `program.md`) follows upstream conventions.
