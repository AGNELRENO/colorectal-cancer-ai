import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# ============================================================
# PATH SETUP
# ============================================================

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

sys.path.append(str(SRC_DIR))


# ============================================================
# IMPORTS
# ============================================================

from model_ghostnet import get_model
from dataset import ColonDataset, train_transform, val_test_transform
from split_dataset import train_samples, val_samples


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 2

BATCH_SIZE = 16

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

NUM_WORKERS = 0


# ============================================================
# CHECKPOINT
# ============================================================

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "best_ghostnet_crc.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("GHOSTNET TRAINING")
print("=" * 60)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# EXISTING GROUP-AWARE SPLIT
# ============================================================

print("\nLoading existing group-aware split...")

print(
    "Training samples:",
    len(train_samples)
)

print(
    "Validation samples:",
    len(val_samples)
)


# ============================================================
# DATASETS
# ============================================================

# IMPORTANT:
# Use the SAME transforms already defined in dataset.py.
#
# Training:
#   training augmentation
#
# Validation:
#   validation/test preprocessing

train_dataset = ColonDataset(
    train_samples,
    transform=train_transform
)

val_dataset = ColonDataset(
    val_samples,
    transform=val_test_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


print("\nDataLoader information:")

print(
    "Train batches:",
    len(train_loader)
)

print(
    "Validation batches:",
    len(val_loader)
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading GhostNet...")

model = get_model(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# CHECKPOINT DIRECTORY
# ============================================================

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING
# ============================================================

best_val_acc = 0.0


for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 60)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}]"
    )

    print("=" * 60)


    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    progress_bar = tqdm(
        train_loader,
        desc="Training"
    )


    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Cross-Entropy Loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # ----------------------------------------------------
        # Training statistics
        # ----------------------------------------------------

        running_loss += (
            loss.item()
            * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


        train_loss = (
            running_loss / total
        )

        train_acc = (
            correct / total
        )


        progress_bar.set_postfix(
            loss=f"{train_loss:.4f}",
            acc=f"{train_acc:.4f}"
        )


    train_loss = (
        running_loss / total
    )

    train_acc = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0

    val_correct = 0

    val_total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )


            # Forward pass

            outputs = model(images)


            # Validation loss

            loss = criterion(
                outputs,
                labels
            )


            val_loss_total += (
                loss.item()
                * images.size(0)
            )


            # Predictions

            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss_total / val_total
    )

    val_acc = (
        val_correct / val_total
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print(
        f"Train Loss: "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_acc:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_acc:.4f}"
    )


    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc


        torch.save(
            model.state_dict(),
            CHECKPOINT_PATH
        )


        print()

        print(
            "Best model saved!"
        )

        print(
            f"Best Validation Accuracy: "
            f"{best_val_acc:.4f}"
        )

        print(
            "Checkpoint:",
            CHECKPOINT_PATH
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)

print("GHOSTNET TRAINING COMPLETE")

print("=" * 60)

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)