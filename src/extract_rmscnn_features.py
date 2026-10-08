import os
import numpy as np
import torch

from torch.utils.data import DataLoader

from dataset import ColonDataset, train_transform, val_test_transform
from model_rmscnn import get_model


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

FEATURE_DIR = os.path.join(
    DATA_DIR,
    "rmscnn_features"
)

CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "checkpoints",
    "best_rmscnn_crc.pth"
)

os.makedirs(
    FEATURE_DIR,
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
# LOAD SPLITS
# ============================================================

train_samples = np.load(
    os.path.join(
        DATA_DIR,
        "train_samples.npy"
    ),
    allow_pickle=True
)

val_samples = np.load(
    os.path.join(
        DATA_DIR,
        "val_samples.npy"
    ),
    allow_pickle=True
)

test_samples = np.load(
    os.path.join(
        DATA_DIR,
        "test_samples.npy"
    ),
    allow_pickle=True
)


print("\n========================================")
print("RMSCNN FEATURE EXTRACTION")
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

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)


# ============================================================
# LOAD RMSCNN
# ============================================================

model = get_model(
    num_classes=2
)

model = model.to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("\nRMSCNN checkpoint loaded successfully.")

print(
    "Best epoch:",
    checkpoint["epoch"]
)

print(
    "Validation accuracy:",
    f"{checkpoint['val_accuracy']:.4f}"
)


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(loader):

    features = []
    labels = []

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            # Extract 128-dimensional features
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

    return features, labels


# ============================================================
# TRAIN FEATURES
# ============================================================

print("\nExtracting training features...")

train_features, train_labels = extract_features(
    train_loader
)

print(
    "Training feature shape:",
    train_features.shape
)


# ============================================================
# VALIDATION FEATURES
# ============================================================

print("\nExtracting validation features...")

val_features, val_labels = extract_features(
    val_loader
)

print(
    "Validation feature shape:",
    val_features.shape
)


# ============================================================
# TEST FEATURES
# ============================================================

print("\nExtracting testing features...")

test_features, test_labels = extract_features(
    test_loader
)

print(
    "Testing feature shape:",
    test_features.shape
)


# ============================================================
# SAVE TRAIN FEATURES
# ============================================================

np.save(
    os.path.join(
        FEATURE_DIR,
        "train_features.npy"
    ),
    train_features
)

np.save(
    os.path.join(
        FEATURE_DIR,
        "train_labels.npy"
    ),
    train_labels
)


# ============================================================
# SAVE VALIDATION FEATURES
# ============================================================

np.save(
    os.path.join(
        FEATURE_DIR,
        "val_features.npy"
    ),
    val_features
)

np.save(
    os.path.join(
        FEATURE_DIR,
        "val_labels.npy"
    ),
    val_labels
)


# ============================================================
# SAVE TEST FEATURES
# ============================================================

np.save(
    os.path.join(
        FEATURE_DIR,
        "test_features.npy"
    ),
    test_features
)

np.save(
    os.path.join(
        FEATURE_DIR,
        "test_labels.npy"
    ),
    test_labels
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("RMSCNN FEATURE EXTRACTION COMPLETE")
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
    "\nSaved to:",
    FEATURE_DIR
)