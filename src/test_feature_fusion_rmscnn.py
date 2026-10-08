import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

RMSCNN_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "rmscnn_features"
)

HC_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "handcrafted_features"
)


# ============================================================
# LOAD RMSCNN FEATURES
# ============================================================

rmscnn_train = np.load(
    os.path.join(RMSCNN_DIR, "train_features.npy")
)

rmscnn_val = np.load(
    os.path.join(RMSCNN_DIR, "val_features.npy")
)

rmscnn_test = np.load(
    os.path.join(RMSCNN_DIR, "test_features.npy")
)

rmscnn_train_labels = np.load(
    os.path.join(RMSCNN_DIR, "train_labels.npy")
)

rmscnn_val_labels = np.load(
    os.path.join(RMSCNN_DIR, "val_labels.npy")
)

rmscnn_test_labels = np.load(
    os.path.join(RMSCNN_DIR, "test_labels.npy")
)


# ============================================================
# LOAD HANDCRAFTED FEATURES
# ============================================================

hc_train = np.load(
    os.path.join(HC_DIR, "train_features.npy")
)

hc_val = np.load(
    os.path.join(HC_DIR, "val_features.npy")
)

hc_test = np.load(
    os.path.join(HC_DIR, "test_features.npy")
)

hc_train_labels = np.load(
    os.path.join(HC_DIR, "train_labels.npy")
)

hc_val_labels = np.load(
    os.path.join(HC_DIR, "val_labels.npy")
)

hc_test_labels = np.load(
    os.path.join(HC_DIR, "test_labels.npy")
)


# ============================================================
# CHECK SHAPES
# ============================================================

print("========================================")
print("RMSCNN + HANDCRAFTED FEATURE FUSION")
print("========================================")

print("\nRMSCNN:")
print("Train:", rmscnn_train.shape)
print("Val  :", rmscnn_val.shape)
print("Test :", rmscnn_test.shape)

print("\nHandcrafted:")
print("Train:", hc_train.shape)
print("Val  :", hc_val.shape)
print("Test :", hc_test.shape)


# ============================================================
# CHECK LABEL CONSISTENCY
# ============================================================

assert np.array_equal(
    rmscnn_train_labels,
    hc_train_labels
), "Training labels do not match!"

assert np.array_equal(
    rmscnn_val_labels,
    hc_val_labels
), "Validation labels do not match!"

assert np.array_equal(
    rmscnn_test_labels,
    hc_test_labels
), "Testing labels do not match!"


# ============================================================
# FEATURE FUSION
# ============================================================

train_fused = np.concatenate(
    [rmscnn_train, hc_train],
    axis=1
)

val_fused = np.concatenate(
    [rmscnn_val, hc_val],
    axis=1
)

test_fused = np.concatenate(
    [rmscnn_test, hc_test],
    axis=1
)


# ============================================================
# SAVE FUSED FEATURES
# ============================================================

FUSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_rmscnn_features"
)

os.makedirs(
    FUSED_DIR,
    exist_ok=True
)


np.save(
    os.path.join(FUSED_DIR, "train_features.npy"),
    train_fused
)

np.save(
    os.path.join(FUSED_DIR, "train_labels.npy"),
    rmscnn_train_labels
)

np.save(
    os.path.join(FUSED_DIR, "val_features.npy"),
    val_fused
)

np.save(
    os.path.join(FUSED_DIR, "val_labels.npy"),
    rmscnn_val_labels
)

np.save(
    os.path.join(FUSED_DIR, "test_features.npy"),
    test_fused
)

np.save(
    os.path.join(FUSED_DIR, "test_labels.npy"),
    rmscnn_test_labels
)


# ============================================================
# FINAL VERIFICATION
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

print("\nExpected feature count: 128 + 44 = 172")

assert train_fused.shape[1] == 172
assert val_fused.shape[1] == 172
assert test_fused.shape[1] == 172

print("\nSUCCESS: RMSCNN + handcrafted features = 172 features")

print(
    "\nSaved to:",
    FUSED_DIR
)