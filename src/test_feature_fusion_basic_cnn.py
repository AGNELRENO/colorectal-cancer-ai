import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

DEEP_FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "basic_cnn_features"
)

HANDCRAFTED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "handcrafted_features"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_basic_cnn_features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD BASIC CNN FEATURES
# ============================================================

train_deep = np.load(
    os.path.join(
        DEEP_FEATURE_DIR,
        "train_features.npy"
    )
)

val_deep = np.load(
    os.path.join(
        DEEP_FEATURE_DIR,
        "val_features.npy"
    )
)

test_deep = np.load(
    os.path.join(
        DEEP_FEATURE_DIR,
        "test_features.npy"
    )
)

train_labels = np.load(
    os.path.join(
        DEEP_FEATURE_DIR,
        "train_labels.npy"
    )
)

val_labels = np.load(
    os.path.join(
        DEEP_FEATURE_DIR,
        "val_labels.npy"
    )
)

test_labels = np.load(
    os.path.join(
        DEEP_FEATURE_DIR,
        "test_labels.npy"
    )
)


# ============================================================
# LOAD HANDCRAFTED FEATURES
# ============================================================

train_hc = np.load(
    os.path.join(
        HANDCRAFTED_DIR,
        "train_features.npy"
    )
)

val_hc = np.load(
    os.path.join(
        HANDCRAFTED_DIR,
        "val_features.npy"
    )
)

test_hc = np.load(
    os.path.join(
        HANDCRAFTED_DIR,
        "test_features.npy"
    )
)


# ============================================================
# DISPLAY ORIGINAL SHAPES
# ============================================================

print("========================================")
print("BASIC CNN + HANDCRAFTED FEATURE FUSION")
print("========================================")

print("\nBasic CNN:")
print(
    "Train:",
    train_deep.shape
)

print(
    "Val  :",
    val_deep.shape
)

print(
    "Test :",
    test_deep.shape
)

print("\nHandcrafted:")
print(
    "Train:",
    train_hc.shape
)

print(
    "Val  :",
    val_hc.shape
)

print(
    "Test :",
    test_hc.shape
)


# ============================================================
# CHECK SAMPLE COUNTS
# ============================================================

assert train_deep.shape[0] == train_hc.shape[0]
assert val_deep.shape[0] == val_hc.shape[0]
assert test_deep.shape[0] == test_hc.shape[0]


# ============================================================
# FEATURE FUSION
# ============================================================

train_fused = np.concatenate(
    [
        train_deep,
        train_hc
    ],
    axis=1
)

val_fused = np.concatenate(
    [
        val_deep,
        val_hc
    ],
    axis=1
)

test_fused = np.concatenate(
    [
        test_deep,
        test_hc
    ],
    axis=1
)


# ============================================================
# VALIDATE LABELS
# ============================================================

assert np.array_equal(
    train_labels,
    np.load(
        os.path.join(
            HANDCRAFTED_DIR,
            "train_labels.npy"
        )
    )
)

assert np.array_equal(
    val_labels,
    np.load(
        os.path.join(
            HANDCRAFTED_DIR,
            "val_labels.npy"
        )
    )
)

assert np.array_equal(
    test_labels,
    np.load(
        os.path.join(
            HANDCRAFTED_DIR,
            "test_labels.npy"
        )
    )
)


# ============================================================
# VALIDATE FEATURE COUNT
# ============================================================

expected_features = 128 + 44

assert train_fused.shape[1] == expected_features
assert val_fused.shape[1] == expected_features
assert test_fused.shape[1] == expected_features


# ============================================================
# SAVE FUSED FEATURES
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_features.npy"
    ),
    train_fused
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
    val_fused
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
    test_fused
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_labels.npy"
    ),
    test_labels
)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FUSION COMPLETE")
print("========================================")

print(
    "Train fused shape:",
    train_fused.shape
)

print(
    "Val fused shape:",
    val_fused.shape
)

print(
    "Test fused shape:",
    test_fused.shape
)

print(
    "\nExpected feature count:",
    "128 + 44 = 172"
)

print(
    "\nSUCCESS: Basic CNN + handcrafted features = 172 features"
)

print(
    "\nSaved to:",
    OUTPUT_DIR
)