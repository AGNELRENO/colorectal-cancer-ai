import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import torch
import numpy as np

from PIL import Image
from torch.utils.data import DataLoader

from dataset import ColonDataset, val_test_transform
from split_dataset import (
    train_samples,
    val_samples,
    test_samples
)
from model import create_model


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR = Path(
    "data/resnet18_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD RESNET18
# ==================================================

model = create_model(
    num_classes=2
)

model.load_state_dict(
    torch.load(
        "best_resnet18_crc.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()


# ==================================================
# REMOVE FINAL CLASSIFIER
# ==================================================

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(
    device
)

feature_extractor.eval()


# ==================================================
# FEATURE EXTRACTION FUNCTION
# ==================================================

def extract_features(
    samples,
    split_name
):

    print()
    print(
        f"===== {split_name.upper()} RESNET18 FEATURE EXTRACTION ====="
    )

    print(
        "Total images:",
        len(samples)
    )


    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = ColonDataset(
        samples,
        transform=val_test_transform
    )


    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )


    # --------------------------------------------------
    # Store features and labels
    # --------------------------------------------------

    all_features = []
    all_labels = []


    # --------------------------------------------------
    # Process batches
    # --------------------------------------------------

    with torch.no_grad():

        processed = 0

        for images, labels in loader:

            images = images.to(
                device
            )


            # --------------------------------------------------
            # Extract deep features
            # --------------------------------------------------

            features = feature_extractor(
                images
            )


            # --------------------------------------------------
            # Flatten
            # --------------------------------------------------

            features = torch.flatten(
                features,
                start_dim=1
            )


            # --------------------------------------------------
            # Move to CPU
            # --------------------------------------------------

            features = features.cpu().numpy()

            labels = labels.numpy()


            # --------------------------------------------------
            # Store
            # --------------------------------------------------

            all_features.append(
                features
            )

            all_labels.append(
                labels
            )


            # --------------------------------------------------
            # Progress
            # --------------------------------------------------

            processed += len(images)

            if (
                processed % 500 == 0
                or
                processed == len(samples)
            ):

                print(
                    f"Processed "
                    f"{processed}/{len(samples)}"
                )


    # --------------------------------------------------
    # Combine batches
    # --------------------------------------------------

    all_features = np.concatenate(
        all_features,
        axis=0
    )

    all_labels = np.concatenate(
        all_labels,
        axis=0
    )


    # --------------------------------------------------
    # Save features
    # --------------------------------------------------

    np.save(
        OUTPUT_DIR /
        f"{split_name}_features.npy",
        all_features
    )


    # --------------------------------------------------
    # Save labels
    # --------------------------------------------------

    np.save(
        OUTPUT_DIR /
        f"{split_name}_labels.npy",
        all_labels
    )


    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        "Feature matrix shape:",
        all_features.shape
    )

    print(
        "Label shape:",
        all_labels.shape
    )

    return all_features, all_labels


# ==================================================
# TRAINING FEATURES
# ==================================================

train_features, train_labels = extract_features(
    train_samples,
    "train"
)


# ==================================================
# VALIDATION FEATURES
# ==================================================

val_features, val_labels = extract_features(
    val_samples,
    "val"
)


# ==================================================
# TEST FEATURES
# ==================================================

test_features, test_labels = extract_features(
    test_samples,
    "test"
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print()
print("========================================")
print("   RESNET18 FEATURE EXTRACTION DONE")
print("========================================")

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
print("Saved files:")

print(
    OUTPUT_DIR /
    "train_features.npy"
)

print(
    OUTPUT_DIR /
    "train_labels.npy"
)

print(
    OUTPUT_DIR /
    "val_features.npy"
)

print(
    OUTPUT_DIR /
    "val_labels.npy"
)

print(
    OUTPUT_DIR /
    "test_features.npy"
)

print(
    OUTPUT_DIR /
    "test_labels.npy"
)