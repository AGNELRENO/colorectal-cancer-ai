import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ColonDataset, val_test_transform
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

TEST_SAMPLES_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "test_samples.npy"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_basic_cnn_crc.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "basic_cnn_features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 32


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
# LOAD SPLITS
# ============================================================

train_samples = np.load(
    TRAIN_SAMPLES_PATH,
    allow_pickle=True
)

val_samples = np.load(
    VAL_SAMPLES_PATH,
    allow_pickle=True
)

test_samples = np.load(
    TEST_SAMPLES_PATH,
    allow_pickle=True
)


print("\n========================================")
print("BASIC CNN FEATURE EXTRACTION")
print("========================================")

print(
    "Training samples  :",
    len(train_samples)
)

print(
    "Validation samples:",
    len(val_samples)
)

print(
    "Testing samples   :",
    len(test_samples)
)


# ============================================================
# DATASETS
# ============================================================

train_dataset = ColonDataset(
    train_samples,
    transform=val_test_transform
)

val_dataset = ColonDataset(
    val_samples,
    transform=val_test_transform
)

test_dataset = ColonDataset(
    test_samples,
    transform=val_test_transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# LOAD MODEL
# ============================================================

model = get_model(
    num_classes=2,
    dropout=0.3
)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print("\nBasic CNN checkpoint loaded successfully.")

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Best validation accuracy:",
    f'{checkpoint["val_accuracy"]:.4f}'
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(
    loader,
    split_name
):

    features = []
    labels = []

    print(
        f"\nExtracting {split_name} features..."
    )

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            batch_features = model.extract_features(
                images
            )

            features.append(
                batch_features.cpu().numpy()
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
        f"{split_name} feature shape:",
        features.shape
    )

    return features, labels


# ============================================================
# EXTRACT TRAINING FEATURES
# ============================================================

train_features, train_labels = extract_features(
    train_loader,
    "training"
)


# ============================================================
# EXTRACT VALIDATION FEATURES
# ============================================================

val_features, val_labels = extract_features(
    val_loader,
    "validation"
)


# ============================================================
# EXTRACT TESTING FEATURES
# ============================================================

test_features, test_labels = extract_features(
    test_loader,
    "testing"
)


# ============================================================
# VALIDATE FEATURE DIMENSION
# ============================================================

assert train_features.shape == (
    len(train_samples),
    128
)

assert val_features.shape == (
    len(val_samples),
    128
)

assert test_features.shape == (
    len(test_samples),
    128
)


# ============================================================
# SAVE FEATURES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_features.npy"
    ),
    train_features
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_labels.npy"
    ),
    train_labels
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_features.npy"
    ),
    val_features
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_labels.npy"
    ),
    val_labels
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_features.npy"
    ),
    test_features
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_labels.npy"
    ),
    test_labels
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("BASIC CNN FEATURE EXTRACTION COMPLETE")
print("========================================")

print(
    "Training features:",
    train_features.shape
)

print(
    "Validation features:",
    val_features.shape
)

print(
    "Testing features:",
    test_features.shape
)

print(
    "\nFeature dimension: 128"
)

print(
    "\nSaved to:"
)

print(
    OUTPUT_DIR
)