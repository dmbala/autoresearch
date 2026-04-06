"""
Fixed data preparation and evaluation for CIFAR-10 autoresearch experiments.
This file is READ-ONLY — the agent must not modify it.

Usage:
    python prepare.py   # downloads CIFAR-10 to ./data/ (one-time)

The evaluation function and data loaders are imported by train.py.
"""

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

NUM_EPOCHS = 10          # fixed epoch budget per experiment
NUM_CLASSES = 10         # CIFAR-10 classes
VALIDATION_SPLIT = 0.2   # 80/20 train/val split
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# CIFAR-10 normalization stats
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]

# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_dataloaders(batch_size, num_workers=4, pin_memory=True):
    """
    Returns (train_loader, val_loader) for CIFAR-10.
    Uses a fixed 80/20 split with a deterministic seed.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download dataset (uses train transform for train split)
    full_dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=TRAIN_TRANSFORM
    )

    # Fixed split with deterministic seed
    train_size = int((1 - VALIDATION_SPLIT) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    # Override transform for validation subset
    # We need a wrapper since random_split doesn't allow changing transforms
    val_dataset.dataset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=False, transform=VAL_TRANSFORM
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, val_loader

# ---------------------------------------------------------------------------
# Evaluation (ground truth — agent cannot modify)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_loader, device):
    """
    Evaluate model on the validation set.
    Returns (val_loss, val_accuracy) where:
      - val_loss: average cross-entropy loss
      - val_accuracy: percentage correct (0-100)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

# ---------------------------------------------------------------------------
# CLI: one-time data download
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Downloading CIFAR-10 to {DATA_DIR}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    print("Done.")
