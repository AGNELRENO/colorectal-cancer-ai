import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "ghostnet_features"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "ghostnet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD GHOSTNET FEATURES
# ============================================================

X_train = np.load(
    os.path.join(FEATURE_DIR, "train_features.npy")
)

y_train = np.load(
    os.path.join(FEATURE_DIR, "train_labels.npy")
)

X_test = np.load(
    os.path.join(FEATURE_DIR, "test_features.npy")
)

y_test = np.load(
    os.path.join(FEATURE_DIR, "test_labels.npy")
)


print("========================================")
print("GhostNet Feature Data")
print("========================================")

print("Training features :", X_train.shape)
print("Training labels   :", y_train.shape)
print("Testing features  :", X_test.shape)
print("Testing labels    :", y_test.shape)


# ============================================================
# ANOVA FEATURE SELECTION
# ============================================================

print("\nApplying ANOVA feature selection...")

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

X_train_selected = selector.fit_transform(
    X_train,
    y_train
)

X_test_selected = selector.transform(
    X_test
)

print(
    "Selected features:",
    X_train_selected.shape[1]
)


# ============================================================
# STANDARDIZATION
# ============================================================

print("Applying StandardScaler...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_selected
)

X_test_scaled = scaler.transform(
    X_test_selected
)


# ============================================================
# RBF SVM
# ============================================================

print("Training RBF-SVM...")

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm.fit(
    X_train_scaled,
    y_train
)

print("SVM training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = svm.predict(
    X_test_scaled
)

# Probability of class 1 = Non-CRC
y_probability = svm.predict_proba(
    X_test_scaled
)[:, 1]


# ============================================================
# CLASSIFICATION METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)


# ============================================================
# ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability,
    pos_label=1
)

roc_auc = auc(
    fpr,
    tpr
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("GhostNet + RBF-SVM Test Results")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 1. ROC CURVE
# ============================================================

plt.figure(figsize=(7, 6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"GhostNet + RBF-SVM (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1.5,
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - GhostNet + RBF-SVM"
)

plt.legend(
    loc="lower right"
)

plt.grid(True)

plt.tight_layout()

roc_path = os.path.join(
    OUTPUT_DIR,
    "ghostnet_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "CRC",
        "Non-CRC"
    ]
)

disp.plot(
    ax=ax,
    values_format="d",
    cmap="Blues",
    colorbar=False
)

ax.set_title(
    "Confusion Matrix - GhostNet + RBF-SVM"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "ghostnet_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. SAVE NUMERICAL RESULTS
# ============================================================

results_path = os.path.join(
    OUTPUT_DIR,
    "ghostnet_roc_confusion_results.txt"
)

with open(
    results_path,
    "w"
) as f:

    f.write(
        "GhostNet + RBF-SVM ROC and Confusion Matrix Results\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Training samples : {len(y_train)}\n"
    )

    f.write(
        f"Testing samples  : {len(y_test)}\n\n"
    )

    f.write(
        f"Accuracy  : {accuracy:.4f}\n"
    )

    f.write(
        f"Precision : {precision:.4f}\n"
    )

    f.write(
        f"Recall    : {recall:.4f}\n"
    )

    f.write(
        f"F1 Score  : {f1:.4f}\n"
    )

    f.write(
        f"ROC-AUC   : {roc_auc:.4f}\n\n"
    )

    f.write(
        "Confusion Matrix:\n"
    )

    f.write(
        str(cm)
    )

    f.write("\n")


# ============================================================
# COMPLETION
# ============================================================

print("\n========================================")
print("GHOSTNET METRIC VISUALIZATION COMPLETE")
print("========================================")

print("\nCreated files:")

print(
    "1. ",
    roc_path
)

print(
    "2. ",
    cm_path
)

print(
    "3. ",
    results_path
)