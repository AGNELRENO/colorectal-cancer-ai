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
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/fused_densenet121_features")
RESULTS_DIR = Path("results/densenet121_fusion_svm")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FEATURES
# ============================================================

X_train = np.load(DATA_DIR / "train_features.npy")
y_train = np.load(DATA_DIR / "train_labels.npy")

X_val = np.load(DATA_DIR / "val_features.npy")
y_val = np.load(DATA_DIR / "val_labels.npy")

X_test = np.load(DATA_DIR / "test_features.npy")
y_test = np.load(DATA_DIR / "test_labels.npy")


print("=" * 60)
print("DENSENET121 + HANDCRAFTED FEATURES + SVM")
print("=" * 60)

print()
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# FEATURE SELECTION — ANOVA
# ============================================================

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

X_train_selected = selector.fit_transform(
    X_train,
    y_train
)

X_val_selected = selector.transform(
    X_val
)

X_test_selected = selector.transform(
    X_test
)


selected_indices = selector.get_support(indices=True)


print()
print("ANOVA FEATURE SELECTION")
print("-" * 40)
print("Original features :", X_train.shape[1])
print("Selected features :", X_train_selected.shape[1])

print()
print("Selected feature indices:")
print(selected_indices)


# ============================================================
# STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_selected
)

X_val_scaled = scaler.transform(
    X_val_selected
)

X_test_scaled = scaler.transform(
    X_test_selected
)


# ============================================================
# RBF SVM
# ============================================================

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)


print()
print("Training RBF SVM...")

svm.fit(
    X_train_scaled,
    y_train
)

print("SVM training complete.")


# ============================================================
# VALIDATION
# ============================================================

val_predictions = svm.predict(
    X_val_scaled
)

val_probabilities = svm.predict_proba(
    X_val_scaled
)[:, 1]


val_accuracy = accuracy_score(
    y_val,
    val_predictions
)

val_precision = precision_score(
    y_val,
    val_predictions,
    average="macro"
)

val_recall = recall_score(
    y_val,
    val_predictions,
    average="macro"
)

val_f1 = f1_score(
    y_val,
    val_predictions,
    average="macro"
)

val_auc = roc_auc_score(
    y_val,
    val_probabilities
)


print()
print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)

print(f"Accuracy  : {val_accuracy:.4f}")
print(f"Precision : {val_precision:.4f}")
print(f"Recall    : {val_recall:.4f}")
print(f"F1-score  : {val_f1:.4f}")
print(f"ROC-AUC   : {val_auc:.4f}")


# ============================================================
# TEST
# ============================================================

test_predictions = svm.predict(
    X_test_scaled
)

test_probabilities = svm.predict_proba(
    X_test_scaled
)[:, 1]


test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

test_precision = precision_score(
    y_test,
    test_predictions,
    average="macro"
)

test_recall = recall_score(
    y_test,
    test_predictions,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions
)


print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)

print(f"Accuracy  : {test_accuracy:.4f}")
print(f"Precision : {test_precision:.4f}")
print(f"Recall    : {test_recall:.4f}")
print(f"F1-score  : {test_f1:.4f}")
print(f"ROC-AUC   : {test_auc:.4f}")

print()
print("Confusion Matrix:")
print(cm)

print()
print("Classification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["CRC", "Non-CRC"]
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

np.save(
    RESULTS_DIR / "selected_feature_indices.npy",
    selected_indices
)

np.save(
    RESULTS_DIR / "test_predictions.npy",
    test_predictions
)

np.save(
    RESULTS_DIR / "test_probabilities.npy",
    test_probabilities
)


with open(
    RESULTS_DIR / "metrics.txt",
    "w"
) as f:

    f.write("DenseNet121 + Handcrafted Features + SVM\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Original features: {X_train.shape[1]}\n")
    f.write(f"Selected features: {X_train_selected.shape[1]}\n\n")

    f.write("SVM configuration:\n")
    f.write("Kernel: RBF\n")
    f.write("C: 1.0\n")
    f.write("Gamma: scale\n\n")

    f.write("Validation Results\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy: {val_accuracy:.4f}\n")
    f.write(f"Precision: {val_precision:.4f}\n")
    f.write(f"Recall: {val_recall:.4f}\n")
    f.write(f"F1-score: {val_f1:.4f}\n")
    f.write(f"ROC-AUC: {val_auc:.4f}\n\n")

    f.write("Test Results\n")
    f.write("-" * 30 + "\n")
    f.write(f"Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Precision: {test_precision:.4f}\n")
    f.write(f"Recall: {test_recall:.4f}\n")
    f.write(f"F1-score: {test_f1:.4f}\n")
    f.write(f"ROC-AUC: {test_auc:.4f}\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))
    f.write("\n\n")

    f.write("Classification Report\n")
    f.write(
        classification_report(
            y_test,
            test_predictions,
            target_names=["CRC", "Non-CRC"]
        )
    )


print()
print("Results saved to:")
print(RESULTS_DIR)

print()
print("DENSENET121 FUSION + SVM COMPLETE!")