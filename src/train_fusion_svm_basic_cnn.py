import os
import numpy as np

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

PROJECT_DIR = r"E:\colorectal-cancer-project"

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_basic_cnn_features"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "basic_cnn_fusion_svm"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("BASIC CNN + HANDCRAFTED → ANOVA → SVM")
print("========================================")

X_train = np.load(
    os.path.join(FEATURE_DIR, "train_features.npy")
)

y_train = np.load(
    os.path.join(FEATURE_DIR, "train_labels.npy")
)

X_val = np.load(
    os.path.join(FEATURE_DIR, "val_features.npy")
)

y_val = np.load(
    os.path.join(FEATURE_DIR, "val_labels.npy")
)

X_test = np.load(
    os.path.join(FEATURE_DIR, "test_features.npy")
)

y_test = np.load(
    os.path.join(FEATURE_DIR, "test_labels.npy")
)


print("\nOriginal feature dimensions:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# ============================================================
# ANOVA FEATURE SELECTION
# ============================================================

print("\n========================================")
print("ANOVA FEATURE SELECTION")
print("========================================")

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

print("Selected features:", X_train_selected.shape[1])

assert X_train_selected.shape[1] == 30
assert X_val_selected.shape[1] == 30
assert X_test_selected.shape[1] == 30


# ============================================================
# STANDARD SCALER
# ============================================================

print("\n========================================")
print("STANDARD SCALING")
print("========================================")

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

print("Scaling complete.")


# ============================================================
# RBF SVM
# ============================================================

print("\n========================================")
print("TRAINING RBF SVM")
print("========================================")

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

print("SVM training complete.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # Class mapping:
    # 0 = CRC
    # 1 = Non-CRC
    #
    # For ROC-AUC, use probability of class 1.
    # This is the same convention used in the
    # previous corrected evaluation.

    auc = roc_auc_score(
        y,
        probabilities[:, 1]
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        pos_label=0,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(f"\n{split_name} RESULTS")
    print("----------------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }


# ============================================================
# VALIDATION
# ============================================================

val_results = evaluate_model(
    svm,
    X_val_scaled,
    y_val,
    "VALIDATION"
)


# ============================================================
# TEST
# ============================================================

test_results = evaluate_model(
    svm,
    X_test_scaled,
    y_test,
    "TEST"
)


# ============================================================
# SAVE SELECTED FEATURE INFORMATION
# ============================================================

selected_indices = selector.get_support(
    indices=True
)

anova_scores = selector.scores_

selected_scores = anova_scores[
    selected_indices
]

feature_ranking = np.argsort(
    anova_scores
)[::-1][:30]


np.save(
    os.path.join(
        RESULT_DIR,
        "selected_feature_indices.npy"
    ),
    selected_indices
)

np.save(
    os.path.join(
        RESULT_DIR,
        "anova_scores.npy"
    ),
    anova_scores
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULT_DIR,
    "basic_cnn_fusion_svm_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Basic CNN + Handcrafted Feature Fusion + RBF-SVM\n"
    )

    f.write(
        "================================================\n\n"
    )

    f.write(
        "Pipeline:\n"
    )

    f.write(
        "128 Basic CNN features + 44 handcrafted features\n"
    )

    f.write(
        "→ 172 fused features\n"
    )

    f.write(
        "→ ANOVA SelectKBest (Top 30)\n"
    )

    f.write(
        "→ StandardScaler\n"
    )

    f.write(
        "→ RBF-SVM\n\n"
    )

    f.write(
        "SVM Parameters:\n"
    )

    f.write(
        "Kernel: RBF\n"
    )

    f.write(
        "C: 1.0\n"
    )

    f.write(
        "Gamma: scale\n\n"
    )

    f.write(
        "Validation Results:\n"
    )

    f.write(
        f"Accuracy : {val_results['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: {val_results['precision']:.4f}\n"
    )

    f.write(
        f"Recall   : {val_results['recall']:.4f}\n"
    )

    f.write(
        f"F1 Score : {val_results['f1']:.4f}\n"
    )

    f.write(
        f"ROC-AUC  : {val_results['auc']:.4f}\n"
    )

    f.write(
        f"Confusion Matrix:\n{val_results['confusion_matrix']}\n\n"
    )

    f.write(
        "Test Results:\n"
    )

    f.write(
        f"Accuracy : {test_results['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision: {test_results['precision']:.4f}\n"
    )

    f.write(
        f"Recall   : {test_results['recall']:.4f}\n"
    )

    f.write(
        f"F1 Score : {test_results['f1']:.4f}\n"
    )

    f.write(
        f"ROC-AUC  : {test_results['auc']:.4f}\n"
    )

    f.write(
        f"Confusion Matrix:\n{test_results['confusion_matrix']}\n"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("BASIC CNN FUSION-SVM COMPLETE")
print("========================================")

print(
    "Results saved to:"
)

print(
    results_file
)