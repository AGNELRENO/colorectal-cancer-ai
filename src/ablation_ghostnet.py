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

FUSED_DIR = (
    PROJECT_DIR / "data" / "fused_ghostnet_features"
)

RESULTS_DIR = (
    PROJECT_DIR / "results" / "ghostnet_ablation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD LABELS
# ============================================================

print("=" * 60)
print("GHOSTNET ABLATION STUDY")
print("=" * 60)

print("\nLoading data...")


train_labels = np.load(
    GHOSTNET_DIR / "train_labels.npy"
)

val_labels = np.load(
    GHOSTNET_DIR / "val_labels.npy"
)

test_labels = np.load(
    GHOSTNET_DIR / "test_labels.npy"
)


# ============================================================
# LOAD GHOSTNET FEATURES
# ============================================================

train_ghostnet = np.load(
    GHOSTNET_DIR / "train_features.npy"
)

val_ghostnet = np.load(
    GHOSTNET_DIR / "val_features.npy"
)

test_ghostnet = np.load(
    GHOSTNET_DIR / "test_features.npy"
)


# ============================================================
# LOAD HANDCRAFTED FEATURES
# ============================================================

train_handcrafted = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

val_handcrafted = np.load(
    HANDCRAFTED_DIR / "val_features.npy"
)

test_handcrafted = np.load(
    HANDCRAFTED_DIR / "test_features.npy"
)


# ============================================================
# LOAD FUSED FEATURES
# ============================================================

train_fused = np.load(
    FUSED_DIR / "train_fused.npy"
)

val_fused = np.load(
    FUSED_DIR / "val_fused.npy"
)

test_fused = np.load(
    FUSED_DIR / "test_fused.npy"
)


print("\nFeature shapes:")

print(
    "GhostNet:",
    train_ghostnet.shape
)

print(
    "Handcrafted:",
    train_handcrafted.shape
)

print(
    "Fused:",
    train_fused.shape
)


# ============================================================
# FUNCTION: TRAIN + EVALUATE
# ============================================================

def run_experiment(
    experiment_name,
    train_features,
    val_features,
    test_features
):

    print("\n" + "=" * 60)

    print(
        f"EXPERIMENT: {experiment_name}"
    )

    print("=" * 60)


    # --------------------------------------------------------
    # ANOVA TOP-30
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=30
    )


    train_selected = selector.fit_transform(
        train_features,
        train_labels
    )

    val_selected = selector.transform(
        val_features
    )

    test_selected = selector.transform(
        test_features
    )


    selected_indices = (
        selector.get_support(
            indices=True
        )
    )


    print(
        "Original features:",
        train_features.shape[1]
    )

    print(
        "Selected features:",
        train_selected.shape[1]
    )


    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

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


    print("\nTraining SVM...")

    svm.fit(
        train_scaled,
        train_labels
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_predictions = svm.predict(
        val_scaled
    )

    val_probabilities = svm.predict_proba(
        val_scaled
    )[:, 1]


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_predictions = svm.predict(
        test_scaled
    )

    test_probabilities = svm.predict_proba(
        test_scaled
    )[:, 1]


    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # TEST METRICS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\nValidation:")

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


    print("\nTest:")

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


    return {
        "Model": experiment_name,
        "Features": train_features.shape[1],
        "Selected": 30,
        "Accuracy": test_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1": test_f1,
        "ROC-AUC": test_auc,
        "Confusion Matrix": test_cm,
        "Selected Indices": selected_indices
    }


# ============================================================
# EXPERIMENT 1
# ============================================================

handcrafted_result = run_experiment(
    "Handcrafted Only",
    train_handcrafted,
    val_handcrafted,
    test_handcrafted
)


# ============================================================
# EXPERIMENT 2
# ============================================================

ghostnet_result = run_experiment(
    "GhostNet Only",
    train_ghostnet,
    val_ghostnet,
    test_ghostnet
)


# ============================================================
# EXPERIMENT 3
# ============================================================

fusion_result = run_experiment(
    "GhostNet + Handcrafted",
    train_fused,
    val_fused,
    test_fused
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_path = (
    RESULTS_DIR /
    "ghostnet_ablation_results.txt"
)


with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GHOSTNET ABLATION STUDY\n"
    )

    f.write(
        "=======================\n\n"
    )


    for result in [
        handcrafted_result,
        ghostnet_result,
        fusion_result
    ]:

        f.write(
            f"{result['Model']}\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            f"Features: "
            f"{result['Features']}\n"
        )

        f.write(
            f"Selected: "
            f"{result['Selected']}\n"
        )

        f.write(
            f"Accuracy: "
            f"{result['Accuracy']:.4f}\n"
        )

        f.write(
            f"Precision: "
            f"{result['Precision']:.4f}\n"
        )

        f.write(
            f"Recall: "
            f"{result['Recall']:.4f}\n"
        )

        f.write(
            f"F1: "
            f"{result['F1']:.4f}\n"
        )

        f.write(
            f"ROC-AUC: "
            f"{result['ROC-AUC']:.4f}\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(result["Confusion Matrix"])
        )

        f.write(
            "\n\n"
        )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print(
    "GHOSTNET ABLATION STUDY COMPLETE"
)

print("=" * 60)

print(
    "Results saved to:"
)

print(
    summary_path
)