"""
Autoresearch CIFAR-10 training script. Single-GPU, single-file.
Supports config-file mode for SLURM array sweeps.

Usage:
    python train.py                                  # use defaults
    python train.py --config configs/batch_001.json --task-id 3  # array mode
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim

from prepare import NUM_EPOCHS, NUM_CLASSES, get_dataloaders, evaluate

# ---------------------------------------------------------------------------
# Default hyperparameters (overridden by --config in array mode)
# ---------------------------------------------------------------------------

DEFAULTS = {
    "batch_size": 128,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "scheduler": "cosine",
    "mixed_precision": "auto",
    "num_workers": 4,
}

# ---------------------------------------------------------------------------
# Model — simple 3-layer CNN (the agent can redesign this entirely)
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model():
    return SimpleCNN(num_classes=NUM_CLASSES)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp, amp_dtype):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file (array of dicts)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Index into the config array (from SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write JSON result file")
    args = parser.parse_args()

    # Load hyperparameters
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            configs = json.load(f)
        task_id = args.task_id if args.task_id is not None else 0
        cfg.update(configs[task_id])

    BATCH_SIZE = cfg["batch_size"]
    LEARNING_RATE = cfg["learning_rate"]
    OPTIMIZER = cfg["optimizer"]
    SCHEDULER = cfg["scheduler"]
    MIXED_PRECISION = cfg["mixed_precision"]
    NUM_WORKERS = cfg["num_workers"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.reset_peak_memory_stats()
    total_start = time.time()

    # Mixed precision setup
    mp = MIXED_PRECISION
    if mp == "auto":
        mp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if mp == "bf16" and not torch.cuda.is_bf16_supported():
        mp = "fp16"
    use_amp = mp in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if mp == "bf16" else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Data
    train_loader, val_loader = get_dataloaders(BATCH_SIZE, num_workers=NUM_WORKERS)

    # Model
    model = build_model().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    if OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

    # Scheduler
    if SCHEDULER == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    elif SCHEDULER == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = None

    # Training loop
    train_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        avg_train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, amp_dtype
        )
        if scheduler:
            scheduler.step()

        print(f"epoch {epoch}/{NUM_EPOCHS}  train_loss={avg_train_loss:.4f}")

    training_seconds = time.time() - train_start

    # Evaluation (uses the fixed evaluation from prepare.py)
    val_loss, val_accuracy = evaluate(model, val_loader, device)

    total_seconds = time.time() - total_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Build result dict
    result = {
        "val_accuracy": round(val_accuracy, 3),
        "val_loss": round(val_loss, 6),
        "training_seconds": round(training_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "num_epochs": NUM_EPOCHS,
        "num_params_M": round(num_params / 1e6, 1),
        "trainable_M": round(trainable_params / 1e6, 1),
        "config": cfg,
    }

    # Print structured summary
    print("---")
    for k, v in result.items():
        if k != "config":
            print(f"{k + ':':20s} {v}")

    # Write JSON result if --output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result written to {args.output}")


if __name__ == "__main__":
    main()
