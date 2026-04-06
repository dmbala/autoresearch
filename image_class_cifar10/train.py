"""
Autoresearch CIFAR-10 training script. Single-GPU, single-file.
Starts with a simple 3-layer CNN. The agent improves it.
Usage: python train.py
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

from prepare import NUM_EPOCHS, NUM_CLASSES, get_dataloaders, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters (the agent tunes these)
# ---------------------------------------------------------------------------

BATCH_SIZE = 128                 # training batch size
LEARNING_RATE = 0.001            # optimizer learning rate
OPTIMIZER = "adam"               # "adam", "adamw", or "sgd"
SCHEDULER = "cosine"             # "cosine", "step", or "none"
MIXED_PRECISION = "auto"         # "auto", "bf16", "fp16", or "none"
NUM_WORKERS = 4                  # dataloader workers

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

    # Print structured summary (parsed by the agent)
    print("---")
    print(f"val_accuracy:     {val_accuracy:.3f}")
    print(f"val_loss:         {val_loss:.6f}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_epochs:       {NUM_EPOCHS}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"trainable_M:      {trainable_params / 1e6:.1f}")


if __name__ == "__main__":
    main()
