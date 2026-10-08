import numpy as np
from pathlib import Path


# ============================================================
# 1. DIRECTORIES
# ============================================================

VGG16_DIR = Path(
    "data/vgg16_features"
)

HANDCRAFTED_DIR = Path(
    "data/handcrafted_features"
)


# ============================================================
# 2. LOAD VGG16 FEATURES
# ============================================================

vgg_train = np.load(
    VGG16_DIR / "train_features.npy"
)

vgg_train_labels = np.load(
    VGG16_DIR / "train_labels.npy"
)

vgg_val = np.load(
    VGG16_DIR / "val_features.npy"
)

vgg_val_labels = np.load(
    VGG16_DIR / "val_labels.npy"
)

vgg_test = np.load(
    VGG16_DIR / "test_features.npy"
)

vgg_test_labels = np.load(
    VGG16_DIR / "test_labels.npy"
)


# ============================================================
# 3. LOAD HANDCRAFTED FEATURES
# ============================================================

hand_train = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

hand_train_labels = np.load(
    HANDCRAFTED_DIR / "train_labels.npy"
)

hand_val = np.load(
    HANDCRAFTED_DIR / "val_features.npy"
)

hand_val_labels = np.load(
    HANDCRAFTED_DIR / "val_labels.npy"
)

hand_test = np.load(
    HANDCRAFTED_DIR / "test_features.npy"
)

hand_test_labels = np.load(
    HANDCRAFTED_DIR / "test_labels.npy"
)


# ============================================================
# 4. PRINT ORIGINAL SHAPES
# ============================================================

print("===== VGG16 FEATURES =====")

print(
    "Train:",
    vgg_train.shape
)

print(
    "Validation:",
    vgg_val.shape
)

print(
    "Test:",
    vgg_test.shape
)


print()
print("===== HANDCRAFTED FEATURES =====")

print(
    "Train:",
    hand_train.shape
)

print(
    "Validation:",
    hand_val.shape
)

print(
    "Test:",
    hand_test.shape
)


# ============================================================
# 5. CHECK LABEL ALIGNMENT
# ============================================================

train_alignment = np.array_equal(
    vgg_train_labels,
    hand_train_labels
)

val_alignment = np.array_equal(
    vgg_val_labels,
    hand_val_labels
)

test_alignment = np.array_equal(
    vgg_test_labels,
    hand_test_labels
)


print()
print("===== LABEL ALIGNMENT =====")

print(
    "Train labels aligned:",
    train_alignment
)

print(
    "Validation labels aligned:",
    val_alignment
)

print(
    "Test labels aligned:",
    test_alignment
)


# ============================================================
# 6. STOP IF LABELS DO NOT MATCH
# ============================================================

if not (
    train_alignment
    and val_alignment
    and test_alignment
):

    raise ValueError(
        "ERROR: Label alignment failed!"
    )


# ============================================================
# 7. FEATURE FUSION
# ============================================================

fused_train = np.concatenate(
    [
        vgg_train,
        hand_train
    ],
    axis=1
)

fused_val = np.concatenate(
    [
        vgg_val,
        hand_val
    ],
    axis=1
)

fused_test = np.concatenate(
    [
        vgg_test,
        hand_test
    ],
    axis=1
)


# ============================================================
# 8. VERIFY FUSED SHAPES
# ============================================================

print()
print("===== FUSED FEATURES =====")

print(
    "Train:",
    fused_train.shape
)

print(
    "Validation:",
    fused_val.shape
)

print(
    "Test:",
    fused_test.shape
)


# ============================================================
# 9. EXPECTED FEATURE COUNT
# ============================================================

expected_features = (
    25088 + 44
)

print()
print(
    "Expected fused feature count:",
    expected_features
)

if fused_train.shape[1] != expected_features:

    raise ValueError(
        "ERROR: Unexpected fused feature count!"
    )


# ============================================================
# 10. CREATE OUTPUT DIRECTORY
# ============================================================

output_dir = Path(
    "data/fused_vgg16_features"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 11. SAVE FUSED FEATURES
# ============================================================

np.save(
    output_dir / "train_features.npy",
    fused_train
)

np.save(
    output_dir / "train_labels.npy",
    vgg_train_labels
)

np.save(
    output_dir / "val_features.npy",
    fused_val
)

np.save(
    output_dir / "val_labels.npy",
    vgg_val_labels
)

np.save(
    output_dir / "test_features.npy",
    fused_test
)

np.save(
    output_dir / "test_labels.npy",
    vgg_test_labels
)


# ============================================================
# 12. COMPLETE
# ============================================================

print()
print("===== VGG16 FEATURE FUSION COMPLETE =====")

print(
    "Fused features saved to:",
    output_dir
)

print()
print("SUCCESS: VGG16 and handcrafted features")
print("are correctly aligned and fused.")