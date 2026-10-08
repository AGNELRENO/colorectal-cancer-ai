import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from dataset import ColonDataset, val_test_transform
from split_dataset import train_samples, val_samples, test_samples
from model_densenet121 import create_densenet121


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print(" DENSENET121 DEEP FEATURE EXTRACTION")
print("========================================")

print(f"Device : {device}")

if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = Path("data/densenet121_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

model = create_densenet121(num_classes=2)

model.load_state_dict(
    torch.load(
        "best_densenet121_crc.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ============================================================
# DENSE NET FEATURE EXTRACTOR
# ============================================================
#
# DenseNet121 produces 1024 feature channels before
# the final classification layer.
#
# We perform:
#
# Image
#   ↓
# DenseNet121 feature layers
#   ↓
# ReLU
#   ↓
# Global Average Pooling
#   ↓
# 1024-dimensional vector
#
# ============================================================

class DenseNetFeatureExtractor(nn.Module):

    def __init__(self, densenet_model):
        super().__init__()

        self.features = densenet_model.features

    def forward(self, x):

        x = self.features(x)

        x = torch.relu(x)

        x = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (1, 1)
        )

        x = torch.flatten(
            x,
            start_dim=1
        )

        return x


feature_extractor = DenseNetFeatureExtractor(model)
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


# ============================================================
# DATA LOADERS
# ============================================================

def create_loader(samples):

    dataset = ColonDataset(
        samples,
        transform=val_test_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    return loader


train_loader = create_loader(train_samples)
val_loader = create_loader(val_samples)
test_loader = create_loader(test_samples)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(loader):

    all_features = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)

            features = feature_extractor(images)

            all_features.append(
                features.cpu().numpy()
            )

            all_labels.append(
                labels.numpy()
            )

    features = np.concatenate(
        all_features,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    return features, labels


# ============================================================
# TRAIN FEATURES
# ============================================================

print()
print("Extracting training features...")

train_features, train_labels = extract_features(
    train_loader
)

print(
    f"Train features : {train_features.shape}"
)


# ============================================================
# VALIDATION FEATURES
# ============================================================

print()
print("Extracting validation features...")

val_features, val_labels = extract_features(
    val_loader
)

print(
    f"Validation features : {val_features.shape}"
)


# ============================================================
# TEST FEATURES
# ============================================================

print()
print("Extracting test features...")

test_features, test_labels = extract_features(
    test_loader
)

print(
    f"Test features : {test_features.shape}"
)


# ============================================================
# VERIFY FEATURE DIMENSION
# ============================================================

print()
print("========================================")
print("       FEATURE DIMENSION CHECK")
print("========================================")

print(
    f"Train feature dimension : {train_features.shape[1]}"
)

print(
    f"Validation feature dimension : "
    f"{val_features.shape[1]}"
)

print(
    f"Test feature dimension : "
    f"{test_features.shape[1]}"
)


assert train_features.shape[1] == 1024
assert val_features.shape[1] == 1024
assert test_features.shape[1] == 1024

print()
print("DenseNet121 feature dimension verified: 1024")


# ============================================================
# SAVE FEATURES
# ============================================================

np.save(
    OUTPUT_DIR / "train_features.npy",
    train_features
)

np.save(
    OUTPUT_DIR / "train_labels.npy",
    train_labels
)

np.save(
    OUTPUT_DIR / "val_features.npy",
    val_features
)

np.save(
    OUTPUT_DIR / "val_labels.npy",
    val_labels
)

np.save(
    OUTPUT_DIR / "test_features.npy",
    test_features
)

np.save(
    OUTPUT_DIR / "test_labels.npy",
    test_labels
)


# ============================================================
# FINAL CHECK
# ============================================================

print()
print("========================================")
print("   DENSENET121 FEATURE EXTRACTION DONE")
print("========================================")

print()
print("Saved files:")

print(
    OUTPUT_DIR / "train_features.npy"
)

print(
    OUTPUT_DIR / "train_labels.npy"
)

print(
    OUTPUT_DIR / "val_features.npy"
)

print(
    OUTPUT_DIR / "val_labels.npy"
)

print(
    OUTPUT_DIR / "test_features.npy"
)

print(
    OUTPUT_DIR / "test_labels.npy"
)

print()
print("Feature extraction successful!")