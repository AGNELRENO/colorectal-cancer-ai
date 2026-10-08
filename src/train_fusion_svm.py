import numpy as np
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

import matplotlib.pyplot as plt


# ==================================================
# PATHS
# ==================================================

DATA_DIR = Path("data/fused_features")

OUTPUT_DIR = Path(
    "results/resnet18_fusion_svm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD DATA
# ==================================================

print("===== LOADING FUSED FEATURES =====")

X_train = np.load(
    DATA_DIR / "train_features.npy"
)

y_train = np.load(
    DATA_DIR / "train_labels.npy"
)

X_val = np.load(
    DATA_DIR / "val_features.npy"
)

y_val = np.load(
    DATA_DIR / "val_labels.npy"
)

X_test = np.load(
    DATA_DIR / "test_features.npy"
)

y_test = np.load(
    DATA_DIR / "test_labels.npy"
)


print(
    "Train:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
)


# ==================================================
# BUILD BASE-PAPER PIPELINE
# ==================================================

print()
print("===== BUILDING SVM PIPELINE =====")

pipeline = Pipeline([
    
    # ANOVA F-test
    (
        "feature_selection",
        SelectKBest(
            score_func=f_classif,
            k=30
        )
    ),

    # StandardScaler
    (
        "scaler",
        StandardScaler()
    ),

    # SVM
    (
        "svm",
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=42
        )
    )
])


print("Feature selection : SelectKBest")
print("Selection method  : ANOVA F-test")
print("Selected features : 30")
print("Scaling           : StandardScaler")
print("SVM kernel        : RBF")
print("C                 : 1.0")
print("Gamma             : scale")


# ==================================================
# TRAIN
# ==================================================

print()
print("===== TRAINING SVM =====")

pipeline.fit(
    X_train,
    y_train
)

print(
    "SVM training completed!"
)


# ==================================================
# CHECK SELECTED FEATURES
# ==================================================

selector = pipeline.named_steps[
    "feature_selection"
]

selected_indices = (
    selector.get_support(indices=True)
)

selected_scores = (
    selector.scores_[selected_indices]
)

print()
print("===== FEATURE SELECTION =====")

print(
    "Original features:",
    X_train.shape[1]
)

print(
    "Selected features:",
    len(selected_indices)
)

print(
    "Selected feature indices:"
)

print(
    selected_indices
)


# ==================================================
# VALIDATION EVALUATION
# ==================================================

print()
print("===== VALIDATION RESULTS =====")

val_predictions = pipeline.predict(
    X_val
)

val_probabilities = pipeline.predict_proba(
    X_val
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
    f"F1-score  : {val_f1:.4f}"
)

print(
    f"ROC-AUC   : {val_auc:.4f}"
)


# ==================================================
# TEST EVALUATION
# ==================================================

print()
print("===== TEST RESULTS =====")

test_predictions = pipeline.predict(
    X_test
)

test_probabilities = pipeline.predict_proba(
    X_test
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
    f"F1-score  : {test_f1:.4f}"
)

print(
    f"ROC-AUC   : {test_auc:.4f}"
)


# ==================================================
# CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    test_predictions
)

print()
print("===== CONFUSION MATRIX =====")

print(cm)


plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    "ResNet18 + Handcrafted Features\nSVM Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1],
    ["CRC", "Non-CRC"]
)

plt.yticks(
    [0, 1],
    ["CRC", "Non-CRC"]
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "confusion_matrix.png",
    dpi=300
)

plt.close()


# ==================================================
# ROC CURVE
# ==================================================

fpr, tpr, _ = roc_curve(
    y_test,
    test_probabilities
)


plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    label=f"AUC = {test_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ResNet18 + Handcrafted Features\nSVM ROC Curve"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "roc_curve.png",
    dpi=300
)

plt.close()


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

print()
print("===== CLASSIFICATION REPORT =====")

report = classification_report(
    y_test,
    test_predictions,
    target_names=[
        "CRC",
        "Non-CRC"
    ]
)

print(report)


with open(
    OUTPUT_DIR / "classification_report.txt",
    "w"
) as f:

    f.write(report)


# ==================================================
# SAVE RESULTS
# ==================================================

with open(
    OUTPUT_DIR / "metrics.txt",
    "w"
) as f:

    f.write(
        "ResNet18 + Handcrafted Features + SVM\n"
    )

    f.write(
        "========================================\n"
    )

    f.write(
        f"Original features: {X_train.shape[1]}\n"
    )

    f.write(
        "Selected features: 30\n"
    )

    f.write(
        "Feature selection: SelectKBest + ANOVA F-test\n"
    )

    f.write(
        "Scaling: StandardScaler\n"
    )

    f.write(
        "SVM kernel: RBF\n"
    )

    f.write(
        "C: 1.0\n"
    )

    f.write(
        "Gamma: scale\n\n"
    )

    f.write(
        f"Validation Accuracy: {val_accuracy:.4f}\n"
    )

    f.write(
        f"Validation Precision: {val_precision:.4f}\n"
    )

    f.write(
        f"Validation Recall: {val_recall:.4f}\n"
    )

    f.write(
        f"Validation F1: {val_f1:.4f}\n"
    )

    f.write(
        f"Validation ROC-AUC: {val_auc:.4f}\n\n"
    )

    f.write(
        f"Test Accuracy: {test_accuracy:.4f}\n"
    )

    f.write(
        f"Test Precision: {test_precision:.4f}\n"
    )

    f.write(
        f"Test Recall: {test_recall:.4f}\n"
    )

    f.write(
        f"Test F1: {test_f1:.4f}\n"
    )

    f.write(
        f"Test ROC-AUC: {test_auc:.4f}\n"
    )


# ==================================================
# FINAL
# ==================================================

print()
print("========================================")
print("       FUSION SVM COMPLETED")
print("========================================")

print()
print("Pipeline:")

print(
    "556 features"
)

print(
    "    ↓"
)

print(
    "ANOVA SelectKBest"
)

print(
    "    ↓"
)

print(
    "30 features"
)

print(
    "    ↓"
)

print(
    "StandardScaler"
)

print(
    "    ↓"
)

print(
    "RBF SVM"
)

print()
print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)