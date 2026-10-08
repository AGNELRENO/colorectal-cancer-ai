import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import (
    ColonDataset,
    train_transform,
    val_test_transform
)

from split_dataset import (
    train_samples,
    val_samples
)

from model_densenet121 import create_densenet121


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("       DENSENET121 TRAINING")
print("========================================")
print(f"Device : {device}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


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
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


print()
print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")


# ============================================================
# MODEL
# ============================================================

model = create_densenet121(num_classes=2)
model = model.to(device)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

num_epochs = 10

best_val_accuracy = 0.0

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(num_epochs):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predictions = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += loss.item() * images.size(0)

            _, predictions = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    val_loss = val_running_loss / val_total
    val_accuracy = val_correct / val_total


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_densenet121_crc.pth"
        )

        best_marker = " ← Best model saved!"

    else:
        best_marker = ""


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
        f"{best_marker}"
    )


# ============================================================
# TRAINING CURVES
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, num_epochs + 1),
    train_losses,
    label="Train Loss"
)

plt.plot(
    range(1, num_epochs + 1),
    val_losses,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("DenseNet121 Training and Validation Loss")

plt.legend()
plt.tight_layout()

plt.savefig(
    "densenet121_loss_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# ACCURACY CURVE
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    range(1, num_epochs + 1),
    train_accuracies,
    label="Train Accuracy"
)

plt.plot(
    range(1, num_epochs + 1),
    val_accuracies,
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("DenseNet121 Training and Validation Accuracy")

plt.legend()
plt.tight_layout()

plt.savefig(
    "densenet121_accuracy_curve.png",
    dpi=300
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("========================================")
print("     DENSENET121 TRAINING COMPLETE")
print("========================================")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print()
print("Saved model:")
print("best_densenet121_crc.pth")

print()
print("Saved curves:")
print("densenet121_loss_curve.png")
print("densenet121_accuracy_curve.png")