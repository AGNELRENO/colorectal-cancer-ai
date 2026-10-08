import numpy as np
from pathlib import Path


# ==================================================
# PATHS
# ==================================================

HANDCRAFTED_DIR = Path(
    "data/handcrafted_features"
)

RESNET_DIR = Path(
    "data/resnet18_features"
)


# ==================================================
# LOAD HANDCRAFTED FEATURES
# ==================================================

train_hc = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

val_hc = np.load(
    HANDCRAFTED_DIR / "val_features.npy"
)

test_hc = np.load(
    HANDCRAFTED_DIR / "test_features.npy"
)


# ==================================================
# LOAD RESNET18 FEATURES
# ==================================================

train_resnet = np.load(
    RESNET_DIR / "train_features.npy"
)

val_resnet = np.load(
    RESNET_DIR / "val_features.npy"
)

test_resnet = np.load(
    RESNET_DIR / "test_features.npy"
)


# ==================================================
# LOAD LABELS
# ==================================================

train_hc_labels = np.load(
    HANDCRAFTED_DIR / "train_labels.npy"
)

val_hc_labels = np.load(
    HANDCRAFTED_DIR / "val_labels.npy"
)

test_hc_labels = np.load(
    HANDCRAFTED_DIR / "test_labels.npy"
)

train_resnet_labels = np.load(
    RESNET_DIR / "train_labels.npy"
)

val_resnet_labels = np.load(
    RESNET_DIR / "val_labels.npy"
)

test_resnet_labels = np.load(
    RESNET_DIR / "test_labels.npy"
)


# ==================================================
# VERIFY SAMPLE COUNTS
# ==================================================

print("===== SAMPLE COUNT CHECK =====")

print(
    "Train handcrafted:",
    train_hc.shape[0]
)

print(
    "Train ResNet18:",
    train_resnet.shape[0]
)

print(
    "Val handcrafted:",
    val_hc.shape[0]
)

print(
    "Val ResNet18:",
    val_resnet.shape[0]
)

print(
    "Test handcrafted:",
    test_hc.shape[0]
)

print(
    "Test ResNet18:",
    test_resnet.shape[0]
)


# ==================================================
# VERIFY LABEL ALIGNMENT
# ==================================================

print()
print("===== LABEL ALIGNMENT CHECK =====")

train_match = np.array_equal(
    train_hc_labels,
    train_resnet_labels
)

val_match = np.array_equal(
    val_hc_labels,
    val_resnet_labels
)

test_match = np.array_equal(
    test_hc_labels,
    test_resnet_labels
)

print(
    "Train labels match:",
    train_match
)

print(
    "Validation labels match:",
    val_match
)

print(
    "Test labels match:",
    test_match
)


# ==================================================
# STOP IF LABELS DO NOT MATCH
# ==================================================

if not (
    train_match
    and val_match
    and test_match
):
    raise ValueError(
        "ERROR: Handcrafted and ResNet18 labels do not match!"
    )


# ==================================================
# FEATURE FUSION
# ==================================================

print()
print("===== FEATURE FUSION =====")


train_fused = np.concatenate(
    [
        train_hc,
        train_resnet
    ],
    axis=1
)

val_fused = np.concatenate(
    [
        val_hc,
        val_resnet
    ],
    axis=1
)

test_fused = np.concatenate(
    [
        test_hc,
        test_resnet
    ],
    axis=1
)


# ==================================================
# VERIFY FUSED SHAPES
# ==================================================

print(
    "Train fused shape:",
    train_fused.shape
)

print(
    "Validation fused shape:",
    val_fused.shape
)

print(
    "Test fused shape:",
    test_fused.shape
)


# ==================================================
# VERIFY EXPECTED FEATURE COUNT
# ==================================================

expected_features = 44 + 512

print()
print(
    "Handcrafted features:",
    44
)

print(
    "ResNet18 features:",
    512
)

print(
    "Expected fused features:",
    expected_features
)


if train_fused.shape[1] != expected_features:
    raise ValueError(
        "ERROR: Unexpected number of fused features!"
    )


if val_fused.shape[1] != expected_features:
    raise ValueError(
        "ERROR: Unexpected number of validation features!"
    )


if test_fused.shape[1] != expected_features:
    raise ValueError(
        "ERROR: Unexpected number of test features!"
    )


# ==================================================
# SAVE FUSED FEATURES
# ==================================================

OUTPUT_DIR = Path(
    "data/fused_features"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


np.save(
    OUTPUT_DIR / "train_features.npy",
    train_fused
)

np.save(
    OUTPUT_DIR / "val_features.npy",
    val_fused
)

np.save(
    OUTPUT_DIR / "test_features.npy",
    test_fused
)

np.save(
    OUTPUT_DIR / "train_labels.npy",
    train_hc_labels
)

np.save(
    OUTPUT_DIR / "val_labels.npy",
    val_hc_labels
)

np.save(
    OUTPUT_DIR / "test_labels.npy",
    test_hc_labels
)


# ==================================================
# FINAL MESSAGE
# ==================================================

print()
print("========================================")
print("       FEATURE FUSION SUCCESSFUL")
print("========================================")

print()
print("Final feature dimension: 556")

print()
print("Saved files:")

print(
    OUTPUT_DIR / "train_features.npy"
)

print(
    OUTPUT_DIR / "val_features.npy"
)

print(
    OUTPUT_DIR / "test_features.npy"
)

print(
    OUTPUT_DIR / "train_labels.npy"
)

print(
    OUTPUT_DIR / "val_labels.npy"
)

print(
    OUTPUT_DIR / "test_labels.npy"
)