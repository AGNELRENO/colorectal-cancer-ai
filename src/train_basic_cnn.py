import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    ColonDataset,
    train_transform,
    val_test_transform
)

from model_basic_cnn import get_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

TRAIN_SAMPLES_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "train_samples.npy"
)

VAL_SAMPLES_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "val_samples.npy"
)

CHECKPOINT_DIR = os.path.join(
    PROJECT_DIR,
    "checkpoints"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_basic_cnn_crc.pth"
)

HISTORY_PATH = os.path.join(
    CHECKPOINT_DIR,
    "basic_cnn_training_history.npz"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10

NUM_CLASSES = 2


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
# LOAD GROUP-AWARE SPLITS
# ============================================================

train_samples = np.load(
    TRAIN_SAMPLES_PATH,
    allow_pickle=True
)

val_samples = np.load(
    VAL_SAMPLES_PATH,
    allow_pickle=True
)


print("\n========================================")
print("BASIC CNN TRAINING")
print("========================================")

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
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

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

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []


best_val_accuracy = 0.0
best_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    NUM_EPOCHS
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

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

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = (
        val_running_loss / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # SAVE BEST CHECKPOINT
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": best_val_accuracy
            },
            CHECKPOINT_PATH
        )

        print(
            "Best model saved!"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.savez(
    HISTORY_PATH,
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
print("BASIC CNN TRAINING COMPLETE")
print("========================================")

print(
    "Best epoch:",
    best_epoch
)

print(
    "Best validation accuracy:",
    f"{best_val_accuracy:.4f}"
)

print(
    "\nCheckpoint saved to:"
)

print(
    CHECKPOINT_PATH
)

print(
    "\nTraining history saved to:"
)

print(
    HISTORY_PATH
)