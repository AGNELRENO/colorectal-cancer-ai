import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import cv2
import numpy as np

from handcrafted_features import extract_handcrafted_features
from split_dataset import (
    train_samples,
    val_samples,
    test_samples
)


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

OUTPUT_DIR = Path(
    "data/handcrafted_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# FEATURE EXTRACTION FUNCTION
# ==================================================

def extract_features_from_samples(
    samples,
    split_name
):

    features = []
    labels = []

    total = len(samples)

    print()
    print(
        f"===== {split_name.upper()} FEATURE EXTRACTION ====="
    )

    print(
        f"Total images: {total}"
    )

    for index, (image_path, label) in enumerate(
        samples
    ):

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Could not load {image_path}"
            )

            continue


        # --------------------------------------------------
        # Convert BGR → RGB
        # --------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )


        # --------------------------------------------------
        # Extract handcrafted features
        # --------------------------------------------------

        feature_vector = extract_handcrafted_features(
            image
        )


        # --------------------------------------------------
        # Store
        # --------------------------------------------------

        features.append(
            feature_vector
        )

        labels.append(
            label
        )


        # --------------------------------------------------
        # Progress
        # --------------------------------------------------

        if (
            (index + 1) % 500 == 0
            or
            (index + 1) == total
        ):

            print(
                f"Processed "
                f"{index + 1}/{total}"
            )


    # --------------------------------------------------
    # Convert to NumPy arrays
    # --------------------------------------------------

    features = np.array(
        features,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )


    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    np.save(
        OUTPUT_DIR /
        f"{split_name}_features.npy",
        features
    )

    np.save(
        OUTPUT_DIR /
        f"{split_name}_labels.npy",
        labels
    )


    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print()
    print(
        f"{split_name.upper()} COMPLETE"
    )

    print(
        "Feature matrix shape:",
        features.shape
    )

    print(
        "Label shape:",
        labels.shape
    )

    return features, labels


# ==================================================
# TRAINING SET
# ==================================================

train_features, train_labels = (
    extract_features_from_samples(
        train_samples,
        "train"
    )
)


# ==================================================
# VALIDATION SET
# ==================================================

val_features, val_labels = (
    extract_features_from_samples(
        val_samples,
        "val"
    )
)


# ==================================================
# TEST SET
# ==================================================

test_features, test_labels = (
    extract_features_from_samples(
        test_samples,
        "test"
    )
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print()
print("========================================")
print("  HANDCRAFTED FEATURE EXTRACTION DONE")
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
print(
    "Saved files:"
)

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