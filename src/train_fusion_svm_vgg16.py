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
# 1. DIRECTORIES
# ============================================================

FEATURE_DIR = Path(
    "data/fused_vgg16_features"
)

RESULTS_DIR = Path(
    "results/vgg16_fusion_svm"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD FUSED FEATURES
# ============================================================

print("===== LOADING FUSED FEATURES =====")

X_train = np.load(
    FEATURE_DIR / "train_features.npy"
)

y_train = np.load(
    FEATURE_DIR / "train_labels.npy"
)

X_val = np.load(
    FEATURE_DIR / "val_features.npy"
)

y_val = np.load(
    FEATURE_DIR / "val_labels.npy"
)

X_test = np.load(
    FEATURE_DIR / "test_features.npy"
)

y_test = np.load(
    FEATURE_DIR / "test_labels.npy"
)


print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# ============================================================
# 3. ANOVA FEATURE SELECTION
# ============================================================

print()
print("===== ANOVA FEATURE SELECTION =====")

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


selected_indices = (
    selector.get_support(indices=True)
)

print(
    "Selected features:",
    len(selected_indices)
)

print(
    "Selected indices:",
    selected_indices.tolist()
)


# ============================================================
# 4. STANDARDIZATION
# ============================================================

print()
print("===== STANDARDIZATION =====")

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
    "Scaled train shape:",
    X_train_scaled.shape
)

print(
    "Scaled val shape:",
    X_val_scaled.shape
)

print(
    "Scaled test shape:",
    X_test_scaled.shape
)


# ============================================================
# 5. RBF SVM
# ============================================================

print()
print("===== TRAINING RBF-SVM =====")

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
# 6. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    split_name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

    # CRC = class 0
    crc_probability = probabilities[:, 0]

    y_crc = (
        y == 0
    ).astype(int)

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

    roc_auc = roc_auc_score(
        y_crc,
        crc_probability
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print()
    print(f"===== {split_name} RESULTS =====")

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "predictions": predictions,
        "probabilities": crc_probability
    }


# ============================================================
# 7. VALIDATION
# ============================================================

val_results = evaluate_model(
    svm,
    X_val_scaled,
    y_val,
    "VALIDATION"
)


# ============================================================
# 8. TEST
# ============================================================

test_results = evaluate_model(
    svm,
    X_test_scaled,
    y_test,
    "TEST"
)


# ============================================================
# 9. SAVE SELECTED FEATURE INDICES
# ============================================================

np.save(
    RESULTS_DIR / "selected_feature_indices.npy",
    selected_indices
)


# ============================================================
# 10. SAVE TEST PREDICTIONS
# ============================================================

np.save(
    RESULTS_DIR / "test_predictions.npy",
    test_results["predictions"]
)

np.save(
    RESULTS_DIR / "test_probabilities.npy",
    test_results["probabilities"]
)


# ============================================================
# 11. SAVE METRICS
# ============================================================

metrics_file = (
    RESULTS_DIR / "metrics.txt"
)

with open(metrics_file, "w") as f:

    f.write(
        "VGG16 + Handcrafted Features + RBF-SVM\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Original fused features: "
        f"{X_train.shape[1]}\n"
    )

    f.write(
        "ANOVA selected features: 30\n\n"
    )

    f.write(
        "SVM configuration:\n"
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
        "VALIDATION RESULTS\n"
    )

    f.write(
        "-------------------\n"
    )

    f.write(
        f"Accuracy  : "
        f"{val_results['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision : "
        f"{val_results['precision']:.4f}\n"
    )

    f.write(
        f"Recall    : "
        f"{val_results['recall']:.4f}\n"
    )

    f.write(
        f"F1 Score  : "
        f"{val_results['f1']:.4f}\n"
    )

    f.write(
        f"ROC-AUC   : "
        f"{val_results['roc_auc']:.4f}\n"
    )

    f.write(
        "\nTEST RESULTS\n"
    )

    f.write(
        "------------\n"
    )

    f.write(
        f"Accuracy  : "
        f"{test_results['accuracy']:.4f}\n"
    )

    f.write(
        f"Precision : "
        f"{test_results['precision']:.4f}\n"
    )

    f.write(
        f"Recall    : "
        f"{test_results['recall']:.4f}\n"
    )

    f.write(
        f"F1 Score  : "
        f"{test_results['f1']:.4f}\n"
    )

    f.write(
        f"ROC-AUC   : "
        f"{test_results['roc_auc']:.4f}\n"
    )

    f.write(
        "\nTEST CONFUSION MATRIX\n"
    )

    f.write(
        str(test_results["confusion_matrix"])
    )


# ============================================================
# 12. COMPLETE
# ============================================================

print()
print("===== VGG16 FUSION + SVM COMPLETE =====")

print(
    "Results saved to:",
    RESULTS_DIR
)