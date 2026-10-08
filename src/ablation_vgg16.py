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

HANDCRAFTED_DIR = Path(
    "data/handcrafted_features"
)

VGG16_DIR = Path(
    "data/vgg16_features"
)

FUSED_DIR = Path(
    "data/fused_vgg16_features"
)

RESULTS_DIR = Path(
    "results/vgg16_ablation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("===== LOADING FEATURES =====")


# -----------------------------
# Handcrafted
# -----------------------------

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


# -----------------------------
# VGG16
# -----------------------------

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


# -----------------------------
# Fused
# -----------------------------

fused_train = np.load(
    FUSED_DIR / "train_features.npy"
)

fused_train_labels = np.load(
    FUSED_DIR / "train_labels.npy"
)

fused_val = np.load(
    FUSED_DIR / "val_features.npy"
)

fused_val_labels = np.load(
    FUSED_DIR / "val_labels.npy"
)

fused_test = np.load(
    FUSED_DIR / "test_features.npy"
)

fused_test_labels = np.load(
    FUSED_DIR / "test_labels.npy"
)


# ============================================================
# 3. VERIFY LABEL ALIGNMENT
# ============================================================

print()
print("===== LABEL ALIGNMENT =====")

print(
    "Handcrafted / VGG16 train:",
    np.array_equal(
        hand_train_labels,
        vgg_train_labels
    )
)

print(
    "Handcrafted / VGG16 val:",
    np.array_equal(
        hand_val_labels,
        vgg_val_labels
    )
)

print(
    "Handcrafted / VGG16 test:",
    np.array_equal(
        hand_test_labels,
        vgg_test_labels
    )
)


# ============================================================
# 4. EVALUATION FUNCTION
# ============================================================

def run_svm_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    experiment_name
):

    print()
    print("=" * 60)
    print(experiment_name)
    print("=" * 60)

    print(
        "Original feature count:",
        X_train.shape[1]
    )


    # --------------------------------------------------------
    # ANOVA
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=30
    )

    X_train_selected = (
        selector.fit_transform(
            X_train,
            y_train
        )
    )

    X_val_selected = (
        selector.transform(X_val)
    )

    X_test_selected = (
        selector.transform(X_test)
    )

    selected_indices = (
        selector.get_support(
            indices=True
        )
    )

    print(
        "Selected features:",
        len(selected_indices)
    )


    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_selected
        )
    )

    X_val_scaled = (
        scaler.transform(
            X_val_selected
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_selected
        )
    )


    # --------------------------------------------------------
    # RBF-SVM
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = svm.predict(
        X_val_scaled
    )

    val_probabilities = (
        svm.predict_proba(
            X_val_scaled
        )[:, 0]
    )

    val_crc = (
        y_val == 0
    ).astype(int)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_precision = precision_score(
        y_val,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_predictions,
        pos_label=0,
        zero_division=0
    )

    val_auc = roc_auc_score(
        val_crc,
        val_probabilities
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = svm.predict(
        X_test_scaled
    )

    test_probabilities = (
        svm.predict_proba(
            X_test_scaled
        )[:, 0]
    )

    test_crc = (
        y_test == 0
    ).astype(int)

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        pos_label=0,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        pos_label=0,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        pos_label=0,
        zero_division=0
    )

    test_auc = roc_auc_score(
        test_crc,
        test_probabilities
    )

    test_cm = confusion_matrix(
        y_test,
        test_predictions
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()
    print("Validation:")
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


    print()
    print("Test:")
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

    print()
    print("Test Confusion Matrix:")
    print(test_cm)


    return {
        "experiment": experiment_name,
        "features": X_train.shape[1],
        "selected": 30,

        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc,

        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auc": test_auc,

        "confusion_matrix": test_cm,
        "selected_indices": selected_indices
    }


# ============================================================
# 5. HANDCRAFTED ONLY
# ============================================================

hand_results = run_svm_experiment(
    hand_train,
    hand_train_labels,
    hand_val,
    hand_val_labels,
    hand_test,
    hand_test_labels,
    "HANDCRAFTED ONLY"
)


# ============================================================
# 6. VGG16 ONLY
# ============================================================

vgg_results = run_svm_experiment(
    vgg_train,
    vgg_train_labels,
    vgg_val,
    vgg_val_labels,
    vgg_test,
    vgg_test_labels,
    "VGG16 ONLY"
)


# ============================================================
# 7. VGG16 + HANDCRAFTED
# ============================================================

fused_results = run_svm_experiment(
    fused_train,
    fused_train_labels,
    fused_val,
    fused_val_labels,
    fused_test,
    fused_test_labels,
    "VGG16 + HANDCRAFTED"
)


# ============================================================
# 8. SUMMARY
# ============================================================

print()
print("=" * 80)
print("===== VGG16 ABLATION SUMMARY =====")
print("=" * 80)

print(
    f"{'Experiment':<25}"
    f"{'Features':>10}"
    f"{'Test Acc':>12}"
    f"{'Test F1':>12}"
    f"{'Test AUC':>12}"
)

print("-" * 80)

for result in [
    hand_results,
    vgg_results,
    fused_results
]:

    print(
        f"{result['experiment']:<25}"
        f"{result['features']:>10}"
        f"{result['test_accuracy']:>12.4f}"
        f"{result['test_f1']:>12.4f}"
        f"{result['test_auc']:>12.4f}"
    )


# ============================================================
# 9. SAVE RESULTS
# ============================================================

summary_file = (
    RESULTS_DIR /
    "ablation_results.txt"
)

with open(summary_file, "w") as f:

    f.write(
        "VGG16 Ablation Study\n"
    )

    f.write(
        "====================\n\n"
    )

    for result in [
        hand_results,
        vgg_results,
        fused_results
    ]:

        f.write(
            f"{result['experiment']}\n"
        )

        f.write(
            f"Features: "
            f"{result['features']}\n"
        )

        f.write(
            f"Selected: "
            f"{result['selected']}\n"
        )

        f.write(
            f"Test Accuracy: "
            f"{result['test_accuracy']:.4f}\n"
        )

        f.write(
            f"Test Precision: "
            f"{result['test_precision']:.4f}\n"
        )

        f.write(
            f"Test Recall: "
            f"{result['test_recall']:.4f}\n"
        )

        f.write(
            f"Test F1: "
            f"{result['test_f1']:.4f}\n"
        )

        f.write(
            f"Test ROC-AUC: "
            f"{result['test_auc']:.4f}\n"
        )

        f.write(
            "Test Confusion Matrix:\n"
        )

        f.write(
            str(
                result["confusion_matrix"]
            )
        )

        f.write(
            "\n\n"
        )


# ============================================================
# 10. COMPLETE
# ============================================================

print()
print(
    "Ablation results saved to:",
    summary_file
)

print(
    "===== VGG16 ABLATION COMPLETE ====="
)