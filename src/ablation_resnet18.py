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
    confusion_matrix
)


# ==================================================
# PATHS
# ==================================================

HANDCRAFTED_DIR = Path(
    "data/handcrafted_features"
)

RESNET_DIR = Path(
    "data/resnet18_features"
)

OUTPUT_DIR = Path(
    "results/ablation_resnet18"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD LABELS
# ==================================================

y_train = np.load(
    HANDCRAFTED_DIR / "train_labels.npy"
)

y_val = np.load(
    HANDCRAFTED_DIR / "val_labels.npy"
)

y_test = np.load(
    HANDCRAFTED_DIR / "test_labels.npy"
)


# ==================================================
# LOAD HANDCRAFTED FEATURES
# ==================================================

X_train_hc = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

X_val_hc = np.load(
    HANDCRAFTED_DIR / "val_features.npy"
)

X_test_hc = np.load(
    HANDCRAFTED_DIR / "test_features.npy"
)


# ==================================================
# LOAD RESNET18 FEATURES
# ==================================================

X_train_resnet = np.load(
    RESNET_DIR / "train_features.npy"
)

X_val_resnet = np.load(
    RESNET_DIR / "val_features.npy"
)

X_test_resnet = np.load(
    RESNET_DIR / "test_features.npy"
)


# ==================================================
# VERIFY LABELS
# ==================================================

assert np.array_equal(
    y_train,
    np.load(
        RESNET_DIR / "train_labels.npy"
    )
)

assert np.array_equal(
    y_val,
    np.load(
        RESNET_DIR / "val_labels.npy"
    )
)

assert np.array_equal(
    y_test,
    np.load(
        RESNET_DIR / "test_labels.npy"
    )
)

print("===== DATA LOADED =====")

print(
    "Handcrafted:",
    X_train_hc.shape
)

print(
    "ResNet18:",
    X_train_resnet.shape
)


# ==================================================
# FUNCTION TO RUN ONE EXPERIMENT
# ==================================================

def run_experiment(
    name,
    X_train,
    X_val,
    X_test
):

    print()
    print("========================================")
    print(name)
    print("========================================")

    print(
        "Original feature count:",
        X_train.shape[1]
    )


    # --------------------------------------------------
    # Pipeline
    # --------------------------------------------------

    pipeline = Pipeline([

        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif,
                k=30
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

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


    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print("Training...")

    pipeline.fit(
        X_train,
        y_train
    )

    print(
        "Training completed."
    )


    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    val_pred = pipeline.predict(
        X_val
    )

    val_prob = pipeline.predict_proba(
        X_val
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


    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    test_pred = pipeline.predict(
        X_test
    )

    test_prob = pipeline.predict_proba(
        X_test
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


    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    cm = confusion_matrix(
        y_test,
        test_pred
    )


    # --------------------------------------------------
    # Selected features
    # --------------------------------------------------

    selector = pipeline.named_steps[
        "feature_selection"
    ]

    selected_indices = (
        selector.get_support(
            indices=True
        )
    )


    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("Selected features:", 30)

    print()
    print("VALIDATION")

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
    print("TEST")

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


    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    result_file = (
        OUTPUT_DIR /
        f"{name.lower().replace(' ', '_')}.txt"
    )

    with open(
        result_file,
        "w"
    ) as f:

        f.write(
            f"{name}\n"
        )

        f.write(
            "=" * 50 + "\n\n"
        )

        f.write(
            f"Original features: "
            f"{X_train.shape[1]}\n"
        )

        f.write(
            "Selected features: 30\n"
        )

        f.write(
            "Feature selection: "
            "ANOVA F-test\n"
        )

        f.write(
            "Scaler: StandardScaler\n"
        )

        f.write(
            "SVM: RBF, C=1.0, gamma=scale\n\n"
        )

        f.write(
            "VALIDATION\n"
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
            f"F1-score: {val_f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC: {val_auc:.4f}\n\n"
        )

        f.write(
            "TEST\n"
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
            f"F1-score: {test_f1:.4f}\n"
        )

        f.write(
            f"ROC-AUC: {test_auc:.4f}\n\n"
        )

        f.write(
            "CONFUSION MATRIX\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nSELECTED FEATURE INDICES\n"
        )

        f.write(
            str(selected_indices)
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
        "test_auc": test_auc
    }


# ==================================================
# EXPERIMENT A
# ==================================================

result_hc = run_experiment(
    "Handcrafted Only",
    X_train_hc,
    X_val_hc,
    X_test_hc
)


# ==================================================
# EXPERIMENT B
# ==================================================

result_resnet = run_experiment(
    "ResNet18 Only",
    X_train_resnet,
    X_val_resnet,
    X_test_resnet
)


# ==================================================
# EXPERIMENT C
# ==================================================

X_train_fused = np.concatenate(
    [
        X_train_hc,
        X_train_resnet
    ],
    axis=1
)

X_val_fused = np.concatenate(
    [
        X_val_hc,
        X_val_resnet
    ],
    axis=1
)

X_test_fused = np.concatenate(
    [
        X_test_hc,
        X_test_resnet
    ],
    axis=1
)


result_fused = run_experiment(
    "Handcrafted + ResNet18",
    X_train_fused,
    X_val_fused,
    X_test_fused
)


# ==================================================
# SUMMARY
# ==================================================

results = [
    result_hc,
    result_resnet,
    result_fused
]


print()
print("========================================")
print("          ABLATION SUMMARY")
print("========================================")

print()

print(
    f"{'Experiment':<28}"
    f"{'Accuracy':>12}"
    f"{'F1':>12}"
    f"{'ROC-AUC':>12}"
)

print("-" * 64)


for result in results:

    print(
        f"{result['name']:<28}"
        f"{result['test_accuracy']:>12.4f}"
        f"{result['test_f1']:>12.4f}"
        f"{result['test_auc']:>12.4f}"
    )


# ==================================================
# SAVE SUMMARY
# ==================================================

with open(
    OUTPUT_DIR / "ablation_summary.txt",
    "w"
) as f:

    f.write(
        "RESNET18 ABLATION STUDY\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Experiment\tAccuracy\tPrecision\tRecall\tF1\tROC-AUC\n"
    )

    for result in results:

        f.write(
            f"{result['name']}\t"
            f"{result['test_accuracy']:.4f}\t"
            f"{result['test_precision']:.4f}\t"
            f"{result['test_recall']:.4f}\t"
            f"{result['test_f1']:.4f}\t"
            f"{result['test_auc']:.4f}\n"
        )


print()
print(
    "Ablation study completed!"
)

print(
    "Results saved to:"
)

print(
    OUTPUT_DIR
)