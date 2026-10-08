import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from dataset import (
    ColonDataset,
    train_transform,
    val_test_transform
)

from model_rmscnn import get_model


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD EXISTING GROUP-AWARE SPLIT
# ============================================================

TRAIN_SAMPLES = os.path.join(
    DATA_DIR,
    "train_samples.npy"
)

VAL_SAMPLES = os.path.join(
    DATA_DIR,
    "val_samples.npy"
)

train_samples = np.load(
    TRAIN_SAMPLES,
    allow_pickle=True
)

val_samples = np.load(
    VAL_SAMPLES,
    allow_pickle=True
)


print("\n========================================")
print("RMSCNN DATASET")
print("========================================")

print(
    "Training samples  :",
    len(train_samples)
)

print(
    "Validation samples:",
    len(val_samples)
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = ColonDataset(
    train_samples,
    transform=train_transform
)

val_dataset = ColonDataset(
    val_samples,
    transform=val_test_transform
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


print(
    "Training batches   :",
    len(train_loader)
)

print(
    "Validation batches :",
    len(val_loader)
)


# ============================================================
# CREATE MODEL
# ============================================================

model = get_model(
    num_classes=2
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
    lr=1e-4
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 10

best_val_accuracy = 0.0

checkpoint_path = os.path.join(
    CHECKPOINT_DIR,
    "best_rmscnn_crc.pth"
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


# ============================================================
# START TRAINING
# ============================================================

print("\n========================================")
print("STARTING RMSCNN TRAINING")
print("========================================")


for epoch in range(NUM_EPOCHS):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0


    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )


        # Clear previous gradients
        optimizer.zero_grad()


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation
        loss.backward()


        # Update model weights
        optimizer.step()


        # Accumulate loss
        running_loss += (
            loss.item() *
            images.size(0)
        )


        # Predictions
        _, predicted = torch.max(
            outputs,
            dim=1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    # Calculate epoch training metrics

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

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


            val_running_loss += (
                loss.item() *
                images.size(0)
            )


            # Predictions
            _, predicted = torch.max(
                outputs,
                dim=1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    # Calculate validation metrics

    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    train_losses.append(
        train_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_losses.append(
        val_loss
    )

    val_accuracies.append(
        val_accuracy
    )


    # ========================================================
    # PRINT EPOCH RESULTS
    # ========================================================

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy


        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy
            },
            checkpoint_path
        )


        print(
            "Best model saved!"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    CHECKPOINT_DIR,
    "rmscnn_training_history.npz"
)


np.savez(
    history_path,
    train_losses=np.array(
        train_losses
    ),
    train_accuracies=np.array(
        train_accuracies
    ),
    val_losses=np.array(
        val_losses
    ),
    val_accuracies=np.array(
        val_accuracies
    )
)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n========================================")
print("RMSCNN TRAINING COMPLETE")
print("========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    "Checkpoint:",
    checkpoint_path
)

print(
    "Training history:",
    history_path
)