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
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

FUSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_rmscnn_features"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "rmscnn_fusion_svm"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD FEATURES
# ============================================================

print("========================================")
print("RMSCNN + HANDCRAFTED FEATURES")
print("ANOVA + STANDARDIZATION + RBF-SVM")
print("========================================")


X_train = np.load(
    os.path.join(FUSED_DIR, "train_features.npy")
)

y_train = np.load(
    os.path.join(FUSED_DIR, "train_labels.npy")
)

X_val = np.load(
    os.path.join(FUSED_DIR, "val_features.npy")
)

y_val = np.load(
    os.path.join(FUSED_DIR, "val_labels.npy")
)

X_test = np.load(
    os.path.join(FUSED_DIR, "test_features.npy")
)

y_test = np.load(
    os.path.join(FUSED_DIR, "test_labels.npy")
)


print("\nOriginal feature shapes:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# ============================================================
# ANOVA FEATURE SELECTION
# ============================================================

print("\n========================================")
print("ANOVA FEATURE SELECTION")
print("========================================")

K = 30

selector = SelectKBest(
    score_func=f_classif,
    k=K
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


# Get selected feature indices
selected_indices = selector.get_support(
    indices=True
)

anova_scores = selector.scores_


print(
    "\nSelected feature count:",
    len(selected_indices)
)

print(
    "Selected feature indices:",
    selected_indices
)


# ============================================================
# SAVE ANOVA INFORMATION
# ============================================================

with open(
    os.path.join(
        RESULT_DIR,
        "anova_features.txt"
    ),
    "w"
) as f:

    f.write(
        "RMSCNN + Handcrafted Feature Fusion\n"
    )

    f.write(
        "ANOVA SelectKBest Results\n\n"
    )

    f.write(
        "Original features: 172\n"
    )

    f.write(
        "Selected features: 30\n\n"
    )

    f.write(
        "Selected feature indices:\n"
    )

    f.write(
        str(selected_indices)
    )

    f.write("\n\n")

    f.write(
        "ANOVA scores of selected features:\n"
    )

    for index in selected_indices:

        f.write(
            f"Feature {index}: "
            f"{anova_scores[index]:.6f}\n"
        )


# ============================================================
# STANDARDIZATION
# ============================================================

print("\n========================================")
print("STANDARDIZATION")
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


print(
    "Train scaled shape:",
    X_train_scaled.shape
)

print(
    "Val scaled shape:",
    X_val_scaled.shape
)

print(
    "Test scaled shape:",
    X_test_scaled.shape
)


# ============================================================
# RBF SVM
# ============================================================

print("\n========================================")
print("TRAINING RBF-SVM")
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

    probabilities = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(
        f"\n{split_name} RESULTS"
    )

    print("----------------------------------------")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y,
            predictions,
            target_names=[
                "CRC",
                "Non-CRC"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm
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
# SAVE RESULTS
# ============================================================

with open(
    os.path.join(
        RESULT_DIR,
        "rmscnn_fusion_svm_results.txt"
    ),
    "w"
) as f:

    f.write(
        "RMSCNN + Handcrafted Features\n"
    )

    f.write(
        "ANOVA Top-30 + StandardScaler + RBF-SVM\n\n"
    )

    f.write(
        "Original features: 172\n"
    )

    f.write(
        "Selected features: 30\n\n"
    )

    f.write(
        "SVM parameters:\n"
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

    f.write("VALIDATION RESULTS\n")
    f.write("----------------------------------------\n")

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
        f"Confusion Matrix:\n{val_results['cm']}\n\n"
    )

    f.write("TEST RESULTS\n")
    f.write("----------------------------------------\n")

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
        f"Confusion Matrix:\n{test_results['cm']}\n"
    )


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "selected_feature_indices.npy"
    ),
    selected_indices
)


print("\n========================================")
print("RMSCNN FUSION-SVM COMPLETE")
print("========================================")

print(
    "Results saved to:"
)

print(
    RESULT_DIR
)