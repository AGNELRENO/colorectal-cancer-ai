import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import ColonDataset, train_transform, val_test_transform
from split_dataset import train_samples, val_samples
from model import create_model


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = ColonDataset(
    train_samples,
    transform=train_transform
)

val_dataset = ColonDataset(
    val_samples,
    transform=val_test_transform
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

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


# --------------------------------------------------
# Model
# --------------------------------------------------

model = create_model(
    num_classes=2
)

model = model.to(device)


# --------------------------------------------------
# Loss function
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)


# --------------------------------------------------
# Learning-rate scheduler
# --------------------------------------------------

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.1,
    patience=2
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

num_epochs = 10

best_val_accuracy = 0.0


# --------------------------------------------------
# Training history
# --------------------------------------------------

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(num_epochs):

    # ==================================================
    # Training
    # ==================================================

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

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

        # Statistics
        train_loss += (
            loss.item() *
            images.size(0)
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        train_correct += (
            predictions == labels
        ).sum().item()

        train_total += labels.size(0)


    # --------------------------------------------------
    # Training metrics
    # --------------------------------------------------

    train_loss = (
        train_loss /
        train_total
    )

    train_accuracy = (
        train_correct /
        train_total
    )


    # ==================================================
    # Validation
    # ==================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() *
                images.size(0)
            )

            _, predictions = torch.max(
                outputs,
                1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)


    # --------------------------------------------------
    # Validation metrics
    # --------------------------------------------------

    val_loss = (
        val_loss /
        val_total
    )

    val_accuracy = (
        val_correct /
        val_total
    )


    # ==================================================
    # Save metrics for visualization
    # ==================================================

    train_losses.append(
        train_loss
    )

    val_losses.append(
        val_loss
    )

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )


    # ==================================================
    # Update learning rate
    # ==================================================

    scheduler.step(
        val_loss
    )


    # ==================================================
    # Print results
    # ==================================================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_accuracy:.4f}"
    )


    # ==================================================
    # Save best model
    # ==================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            "best_resnet18_crc.pth"
        )

        print(
            f"Best model saved! "
            f"Validation Accuracy: "
            f"{val_accuracy:.4f}"
        )


# --------------------------------------------------
# Training complete
# --------------------------------------------------

print()
print("===== TRAINING COMPLETE =====")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)


# ==================================================
# VISUALIZATION
# ==================================================

epochs = range(
    1,
    num_epochs + 1
)


# --------------------------------------------------
# Loss curve
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.title(
    "ResNet18 Training and Validation Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "resnet18_loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Accuracy curve
# --------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "ResNet18 Training and Validation Accuracy"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    "resnet18_accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# Visualization complete
# --------------------------------------------------

print()
print("===== VISUALIZATION COMPLETE =====")

print(
    "Saved:"
)

print(
    "1. resnet18_loss_curve.png"
)

print(
    "2. resnet18_accuracy_curve.png"
)