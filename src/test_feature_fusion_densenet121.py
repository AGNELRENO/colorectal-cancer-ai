import numpy as np
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

HANDCRAFTED_DIR = Path("data/handcrafted_features")
DENSENET_DIR = Path("data/densenet121_features")


# ============================================================
# LOAD HANDCRAFTED FEATURES
# ============================================================

hand_train = np.load(HANDCRAFTED_DIR / "train_features.npy")
hand_val = np.load(HANDCRAFTED_DIR / "val_features.npy")
hand_test = np.load(HANDCRAFTED_DIR / "test_features.npy")

hand_train_labels = np.load(HANDCRAFTED_DIR / "train_labels.npy")
hand_val_labels = np.load(HANDCRAFTED_DIR / "val_labels.npy")
hand_test_labels = np.load(HANDCRAFTED_DIR / "test_labels.npy")


# ============================================================
# LOAD DENSENET121 FEATURES
# ============================================================

dense_train = np.load(DENSENET_DIR / "train_features.npy")
dense_val = np.load(DENSENET_DIR / "val_features.npy")
dense_test = np.load(DENSENET_DIR / "test_features.npy")

dense_train_labels = np.load(DENSENET_DIR / "train_labels.npy")
dense_val_labels = np.load(DENSENET_DIR / "val_labels.npy")
dense_test_labels = np.load(DENSENET_DIR / "test_labels.npy")


# ============================================================
# CHECK SAMPLE COUNTS
# ============================================================

assert hand_train.shape[0] == dense_train.shape[0]
assert hand_val.shape[0] == dense_val.shape[0]
assert hand_test.shape[0] == dense_test.shape[0]


# ============================================================
# CHECK LABEL ALIGNMENT
# ============================================================

assert np.array_equal(hand_train_labels, dense_train_labels)
assert np.array_equal(hand_val_labels, dense_val_labels)
assert np.array_equal(hand_test_labels, dense_test_labels)


# ============================================================
# FUSION
# ============================================================

fused_train = np.concatenate(
    [dense_train, hand_train],
    axis=1
)

fused_val = np.concatenate(
    [dense_val, hand_val],
    axis=1
)

fused_test = np.concatenate(
    [dense_test, hand_test],
    axis=1
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 55)
print("DENSENET121 + HANDCRAFTED FEATURE FUSION")
print("=" * 55)

print()
print("DenseNet121 feature dimensions:")
print("Train:", dense_train.shape)
print("Validation:", dense_val.shape)
print("Test:", dense_test.shape)

print()
print("Handcrafted feature dimensions:")
print("Train:", hand_train.shape)
print("Validation:", hand_val.shape)
print("Test:", hand_test.shape)

print()
print("Fused feature dimensions:")
print("Train:", fused_train.shape)
print("Validation:", fused_val.shape)
print("Test:", fused_test.shape)

print()
print("Expected fused feature dimension: 1024 + 44 = 1068")

print()
print("Label alignment:")
print("Train:", np.array_equal(hand_train_labels, dense_train_labels))
print("Validation:", np.array_equal(hand_val_labels, dense_val_labels))
print("Test:", np.array_equal(hand_test_labels, dense_test_labels))


# ============================================================
# SAVE FUSED FEATURES
# ============================================================

OUTPUT_DIR = Path("data/fused_densenet121_features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

np.save(OUTPUT_DIR / "train_features.npy", fused_train)
np.save(OUTPUT_DIR / "train_labels.npy", dense_train_labels)

np.save(OUTPUT_DIR / "val_features.npy", fused_val)
np.save(OUTPUT_DIR / "val_labels.npy", dense_val_labels)

np.save(OUTPUT_DIR / "test_features.npy", fused_test)
np.save(OUTPUT_DIR / "test_labels.npy", dense_test_labels)


print()
print("Saved fused features to:")
print(OUTPUT_DIR)

print()
print("FEATURE FUSION TEST SUCCESSFUL!")