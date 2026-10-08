import numpy as np
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

GHOSTNET_DIR = (
    PROJECT_DIR / "data" / "ghostnet_features"
)

HANDCRAFTED_DIR = (
    PROJECT_DIR / "data" / "handcrafted_features"
)

OUTPUT_DIR = (
    PROJECT_DIR / "data" / "fused_ghostnet_features"
)

RESULTS_DIR = (
    PROJECT_DIR / "results" / "ghostnet_fusion_svm"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD GHOSTNET FEATURES
# ============================================================

print("=" * 60)
print("GHOSTNET + HANDCRAFTED FEATURE FUSION")
print("=" * 60)

print("\nLoading GhostNet features...")

train_deep = np.load(
    GHOSTNET_DIR / "train_features.npy"
)

val_deep = np.load(
    GHOSTNET_DIR / "val_features.npy"
)

test_deep = np.load(
    GHOSTNET_DIR / "test_features.npy"
)

train_labels = np.load(
    GHOSTNET_DIR / "train_labels.npy"
)

val_labels = np.load(
    GHOSTNET_DIR / "val_labels.npy"
)

test_labels = np.load(
    GHOSTNET_DIR / "test_labels.npy"
)


print(
    "Train deep features:",
    train_deep.shape
)

print(
    "Validation deep features:",
    val_deep.shape
)

print(
    "Test deep features:",
    test_deep.shape
)


# ============================================================
# LOAD HANDCRAFTED FEATURES
# ============================================================

print("\nLoading handcrafted features...")

train_hand = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

val_hand = np.load(
    HANDCRAFTED_DIR / "val_features.npy"
)

test_hand = np.load(
    HANDCRAFTED_DIR / "test_features.npy"
)


print(
    "Train handcrafted features:",
    train_hand.shape
)

print(
    "Validation handcrafted features:",
    val_hand.shape
)

print(
    "Test handcrafted features:",
    test_hand.shape
)


# ============================================================
# VERIFY LABELS
# ============================================================

print("\nChecking labels...")

if not np.array_equal(
    train_labels,
    np.load(
        HANDCRAFTED_DIR / "train_labels.npy"
    )
):
    raise ValueError(
        "Training labels do not match!"
    )


if not np.array_equal(
    val_labels,
    np.load(
        HANDCRAFTED_DIR / "val_labels.npy"
    )
):
    raise ValueError(
        "Validation labels do not match!"
    )


if not np.array_equal(
    test_labels,
    np.load(
        HANDCRAFTED_DIR / "test_labels.npy"
    )
):
    raise ValueError(
        "Testing labels do not match!"
    )


print("Label verification successful.")


# ============================================================
# FEATURE FUSION
# ============================================================

print("\n===== FEATURE FUSION =====")

train_fused = np.concatenate(
    [
        train_deep,
        train_hand
    ],
    axis=1
)

val_fused = np.concatenate(
    [
        val_deep,
        val_hand
    ],
    axis=1
)

test_fused = np.concatenate(
    [
        test_deep,
        test_hand
    ],
    axis=1
)


print(
    "Train fused:",
    train_fused.shape
)

print(
    "Validation fused:",
    val_fused.shape
)

print(
    "Test fused:",
    test_fused.shape
)


# ============================================================
# SAVE FUSED FEATURES
# ============================================================

np.save(
    OUTPUT_DIR / "train_fused.npy",
    train_fused
)

np.save(
    OUTPUT_DIR / "val_fused.npy",
    val_fused
)

np.save(
    OUTPUT_DIR / "test_fused.npy",
    test_fused
)


# ============================================================
# ANOVA FEATURE SELECTION
# ============================================================

print("\n===== ANOVA FEATURE SELECTION =====")

TOP_K = 30

selector = SelectKBest(
    score_func=f_classif,
    k=TOP_K
)


train_selected = selector.fit_transform(
    train_fused,
    train_labels
)

val_selected = selector.transform(
    val_fused
)

test_selected = selector.transform(
    test_fused
)


selected_indices = (
    selector.get_support(indices=True)
)

selected_scores = (
    selector.scores_[selected_indices]
)


print(
    "Original feature count:",
    train_fused.shape[1]
)

print(
    "Selected feature count:",
    train_selected.shape[1]
)

print(
    "Selected feature indices:"
)

print(
    selected_indices
)


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

np.save(
    OUTPUT_DIR / "train_selected.npy",
    train_selected
)

np.save(
    OUTPUT_DIR / "val_selected.npy",
    val_selected
)

np.save(
    OUTPUT_DIR / "test_selected.npy",
    test_selected
)

np.save(
    OUTPUT_DIR / "selected_indices.npy",
    selected_indices
)

np.save(
    OUTPUT_DIR / "selected_scores.npy",
    selected_scores
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n===== STANDARDIZATION =====")

scaler = StandardScaler()

train_scaled = scaler.fit_transform(
    train_selected
)

val_scaled = scaler.transform(
    val_selected
)

test_scaled = scaler.transform(
    test_selected
)


# ============================================================
# RBF SVM
# ============================================================

print("\n===== RBF-SVM =====")

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)


print("Training SVM...")

svm.fit(
    train_scaled,
    train_labels
)

print("SVM training complete.")


# ============================================================
# VALIDATION PREDICTION
# ============================================================

val_predictions = svm.predict(
    val_scaled
)

val_probabilities = svm.predict_proba(
    val_scaled
)[:, 1]


# ============================================================
# TEST PREDICTION
# ============================================================

test_predictions = svm.predict(
    test_scaled
)

test_probabilities = svm.predict_proba(
    test_scaled
)[:, 1]


# ============================================================
# VALIDATION METRICS
# ============================================================

val_accuracy = accuracy_score(
    val_labels,
    val_predictions
)

val_precision = precision_score(
    val_labels,
    val_predictions,
    zero_division=0
)

val_recall = recall_score(
    val_labels,
    val_predictions,
    zero_division=0
)

val_f1 = f1_score(
    val_labels,
    val_predictions,
    zero_division=0
)

val_auc = roc_auc_score(
    val_labels,
    val_probabilities
)


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_labels,
    test_predictions
)

test_precision = precision_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_recall = recall_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    test_labels,
    test_predictions,
    zero_division=0
)

test_auc = roc_auc_score(
    test_labels,
    test_probabilities
)

test_cm = confusion_matrix(
    test_labels,
    test_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("GHOSTNET FUSION-SVM RESULTS")
print("=" * 60)

print("\nValidation Results:")

print(
    f"Accuracy  : {val_accuracy:.4f}"
)

print(
    f"Precision : {val_precision:.4f}"
)

print(
    f"Recall    : {val_recall:.4f}"
)

print(
    f"F1 Score  : {val_f1:.4f}"
)

print(
    f"ROC-AUC   : {val_auc:.4f}"
)


print("\nTest Results:")

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)


print("\nTest Confusion Matrix:")

print(
    test_cm
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = (
    RESULTS_DIR /
    "ghostnet_fusion_svm_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GhostNet + Handcrafted Feature Fusion\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        f"Deep features: "
        f"{train_deep.shape[1]}\n"
    )

    f.write(
        f"Handcrafted features: "
        f"{train_hand.shape[1]}\n"
    )

    f.write(
        f"Fused features: "
        f"{train_fused.shape[1]}\n"
    )

    f.write(
        f"Selected features: "
        f"{TOP_K}\n\n"
    )

    f.write(
        "Validation Results\n"
    )

    f.write(
        f"Accuracy: {val_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {val_precision:.4f}\n"
    )

    f.write(
        f"Recall: {val_recall:.4f}\n"
    )

    f.write(
        f"F1: {val_f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {val_auc:.4f}\n\n"
    )

    f.write(
        "Test Results\n"
    )

    f.write(
        f"Accuracy: {test_accuracy:.4f}\n"
    )

    f.write(
        f"Precision: {test_precision:.4f}\n"
    )

    f.write(
        f"Recall: {test_recall:.4f}\n"
    )

    f.write(
        f"F1: {test_f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC: {test_auc:.4f}\n\n"
    )

    f.write(
        "Test Confusion Matrix\n"
    )

    f.write(
        str(test_cm)
    )


# ============================================================
# FINAL
# ============================================================

print("\nResults saved to:")

print(
    results_file
)

print("\n" + "=" * 60)
print(
    "GHOSTNET FUSION-SVM COMPLETE"
)
print("=" * 60)