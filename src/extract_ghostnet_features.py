import torch
import numpy as np

from torch.utils.data import DataLoader
from pathlib import Path

from dataset import ColonDataset, val_test_transform
from model_ghostnet import get_model
from split_dataset import (
    train_samples,
    val_samples,
    test_samples
)


# ============================================================
# 1. DEVICE
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
# 2. LOAD BEST GHOSTNET MODEL
# ============================================================

model = get_model(
    num_classes=2
)

checkpoint_path = Path(
    "checkpoints/best_ghostnet_crc.pth"
)

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print(
    "Checkpoint loaded:",
    checkpoint_path
)


# ============================================================
# 3. REMOVE GHOSTNET CLASSIFIER
# ============================================================

# GhostNet architecture:
#
# Image
#   ↓
# GhostNet feature extractor
#   ↓
# Global pooling
#   ↓
# Classifier
#
# We want the deep feature representation
# BEFORE the final classification layer.
#
# timm GhostNet provides forward_features()
# for this purpose.


# ============================================================
# 4. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(samples, split_name):

    dataset = ColonDataset(
        samples,
        transform=val_test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    features = []
    labels = []

    print()
    print(
        f"Extracting GhostNet features "
        f"for {split_name}..."
    )

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Extract GhostNet deep features
            # ------------------------------------------------

            outputs = model.forward_features(
                images
            )

            # ------------------------------------------------
            # Global average pooling
            #
            # GhostNet feature map:
            # [batch, channels, height, width]
            #
            # becomes:
            # [batch, channels]
            # ------------------------------------------------

            outputs = torch.mean(
                outputs,
                dim=(2, 3)
            )

            features.append(
                outputs.cpu().numpy()
            )

            labels.append(
                batch_labels.numpy()
            )

    features = np.concatenate(
        features,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    print(
        f"{split_name} features shape: "
        f"{features.shape}"
    )

    print(
        f"{split_name} labels shape: "
        f"{labels.shape}"
    )

    return features, labels


# ============================================================
# 5. EXTRACT TRAIN FEATURES
# ============================================================

train_features, train_labels = extract_features(
    train_samples,
    "Training"
)


# ============================================================
# 6. EXTRACT VALIDATION FEATURES
# ============================================================

val_features, val_labels = extract_features(
    val_samples,
    "Validation"
)


# ============================================================
# 7. EXTRACT TEST FEATURES
# ============================================================

test_features, test_labels = extract_features(
    test_samples,
    "Testing"
)


# ============================================================
# 8. CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = Path(
    "data/ghostnet_features"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 9. SAVE FEATURES
# ============================================================

np.save(
    output_dir / "train_features.npy",
    train_features
)

np.save(
    output_dir / "train_labels.npy",
    train_labels
)

np.save(
    output_dir / "val_features.npy",
    val_features
)

np.save(
    output_dir / "val_labels.npy",
    val_labels
)

np.save(
    output_dir / "test_features.npy",
    test_features
)

np.save(
    output_dir / "test_labels.npy",
    test_labels
)


# ============================================================
# 10. FINAL VERIFICATION
# ============================================================

print()
print(
    "===== GHOSTNET FEATURE EXTRACTION COMPLETE ====="
)

print(
    "Train:",
    train_features.shape
)

print(
    "Validation:",
    val_features.shape
)

print(
    "Test:",
    test_features.shape
)

print()
print(
    "Features saved to:",
    output_dir
)