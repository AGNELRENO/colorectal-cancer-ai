import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import ColonDataset, train_transform, val_test_transform
from model_vgg16 import create_vgg16

# Import the SAME group-aware split used by ResNet18/DenseNet121
from split_dataset import train_samples, val_samples


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. DATASET
# ============================================================

train_dataset = ColonDataset(
    train_samples,
    transform=train_transform
)

val_dataset = ColonDataset(
    val_samples,
    transform=val_test_transform
)

print()
print("===== DATASET =====")
print("Training samples  :", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# 3. DATALOADER
# ============================================================

# VGG16 is memory-heavy, so use batch size 8
# to safely fit on the RTX 2050 4 GB VRAM.

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# 4. MODEL
# ============================================================

model = create_vgg16(num_classes=2)
model = model.to(device)


# ============================================================
# 5. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 6. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)


# ============================================================
# 7. TRAINING SETTINGS
# ============================================================

num_epochs = 10

best_val_acc = 0.0

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

best_model_path = (
    checkpoint_dir / "best_vgg16_crc.pth"
)


# ============================================================
# 8. TRAINING LOOP
# ============================================================

for epoch in range(num_epochs):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best model saved!")


# ============================================================
# 9. TRAINING COMPLETE
# ============================================================

print()
print("===== TRAINING COMPLETE =====")

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc:.4f}"
)

print(
    f"Best model saved to: "
    f"{best_model_path}"
)