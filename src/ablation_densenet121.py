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

HANDCRAFTED_DIR = Path("data/handcrafted_features")
DENSENET_DIR = Path("data/densenet121_features")

RESULTS_DIR = Path("results/densenet121_ablation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD LABELS
# ============================================================

y_train = np.load(
    HANDCRAFTED_DIR / "train_labels.npy"
)

y_val = np.load(
    HANDCRAFTED_DIR / "val_labels.npy"
)

y_test = np.load(
    HANDCRAFTED_DIR / "test_labels.npy"
)


# ============================================================
# LOAD FEATURES
# ============================================================

hand_train = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

hand_val = np.load(
    HANDCRAFTED_DIR / "val_features.npy"
)

hand_test = np.load(
    HANDCRAFTED_DIR / "test_features.npy"
)


dense_train = np.load(
    DENSENET_DIR / "train_features.npy"
)

dense_val = np.load(
    DENSENET_DIR / "val_features.npy"
)

dense_test = np.load(
    DENSENET_DIR / "test_features.npy"
)


# ============================================================
# FUSION
# ============================================================

fusion_train = np.concatenate(
    [dense_train, hand_train],
    axis=1
)

fusion_val = np.concatenate(
    [dense_val, hand_val],
    axis=1
)

fusion_test = np.concatenate(
    [dense_test, hand_test],
    axis=1
)


# ============================================================
# FUNCTION FOR ONE ABLATION EXPERIMENT
# ============================================================

def run_experiment(
    name,
    X_train,
    X_val,
    X_test
):

    print()
    print("=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Original feature dimension: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # ANOVA / SelectKBest
    # --------------------------------------------------------

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

    selected_indices = selector.get_support(
        indices=True
    )

    print(
        f"Selected feature dimension: "
        f"{X_train_selected.shape[1]}"
    )

    # --------------------------------------------------------
    # STANDARD SCALER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RBF SVM
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

    val_pred = svm.predict(
        X_val_scaled
    )

    val_prob = svm.predict_proba(
        X_val_scaled
    )[:, 1]

    val_accuracy = accuracy_score(
        y_val,
        val_pred
    )

    val_precision = precision_score(
        y_val,
        val_pred,
        average="macro"
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        average="macro"
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        average="macro"
    )

    val_auc = roc_auc_score(
        y_val,
        val_prob
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_pred = svm.predict(
        X_test_scaled
    )

    test_prob = svm.predict_proba(
        X_test_scaled
    )[:, 1]

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_precision = precision_score(
        y_test,
        test_pred,
        average="macro"
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        average="macro"
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="macro"
    )

    test_auc = roc_auc_score(
        y_test,
        test_prob
    )

    cm = confusion_matrix(
        y_test,
        test_pred
    )

    # --------------------------------------------------------
    # PRINT RESULTS
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
        f"F1-score  : {val_f1:.4f}"
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
        f"F1-score  : {test_f1:.4f}"
    )
    print(
        f"ROC-AUC   : {test_auc:.4f}"
    )

    print()
    print("Confusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    safe_name = name.lower().replace(
        " ",
        "_"
    ).replace(
        "+",
        "plus"
    )

    np.save(
        RESULTS_DIR /
        f"{safe_name}_selected_indices.npy",
        selected_indices
    )

    np.save(
        RESULTS_DIR /
        f"{safe_name}_test_predictions.npy",
        test_pred
    )

    np.save(
        RESULTS_DIR /
        f"{safe_name}_test_probabilities.npy",
        test_prob
    )

    return {
        "name": name,
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
        "confusion_matrix": cm
    }


# ============================================================
# EXPERIMENT A — HANDCRAFTED ONLY
# ============================================================

result_handcrafted = run_experiment(
    "Handcrafted Only",
    hand_train,
    hand_val,
    hand_test
)


# ============================================================
# EXPERIMENT B — DENSENET121 ONLY
# ============================================================

result_densenet = run_experiment(
    "DenseNet121 Only",
    dense_train,
    dense_val,
    dense_test
)


# ============================================================
# EXPERIMENT C — DENSENET121 + HANDCRAFTED
# ============================================================

result_fusion = run_experiment(
    "DenseNet121 + Handcrafted",
    fusion_train,
    fusion_val,
    fusion_test
)


# ============================================================
# FINAL ABLATION SUMMARY
# ============================================================

print()
print("=" * 75)
print("DENSENET121 ABLATION SUMMARY")
print("=" * 75)

print(
    f"{'Experiment':<30}"
    f"{'Accuracy':<12}"
    f"{'F1':<12}"
    f"{'ROC-AUC':<12}"
)

print("-" * 75)

for result in [
    result_handcrafted,
    result_densenet,
    result_fusion
]:

    print(
        f"{result['name']:<30}"
        f"{result['test_accuracy']:.4f}"
        f"       "
        f"{result['test_f1']:.4f}"
        f"       "
        f"{result['test_auc']:.4f}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    RESULTS_DIR / "ablation_summary.txt",
    "w"
) as f:

    f.write(
        "DENSENET121 ABLATION STUDY\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    for result in [
        result_handcrafted,
        result_densenet,
        result_fusion
    ]:

        f.write(
            f"{result['name']}\n"
        )

        f.write(
            f"Test Accuracy : "
            f"{result['test_accuracy']:.4f}\n"
        )

        f.write(
            f"Test Precision: "
            f"{result['test_precision']:.4f}\n"
        )

        f.write(
            f"Test Recall   : "
            f"{result['test_recall']:.4f}\n"
        )

        f.write(
            f"Test F1       : "
            f"{result['test_f1']:.4f}\n"
        )

        f.write(
            f"Test ROC-AUC  : "
            f"{result['test_auc']:.4f}\n"
        )

        f.write(
            "Confusion Matrix:\n"
        )

        f.write(
            str(result["confusion_matrix"])
        )

        f.write(
            "\n\n"
        )


print()
print(
    "Ablation results saved to:"
)
print(RESULTS_DIR)

print()
print(
    "DENSENET121 ABLATION COMPLETE!"
)