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

RMSCNN_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "rmscnn_features"
)

HC_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "handcrafted_features"
)

FUSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_rmscnn_features"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "rmscnn_ablation"
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("RMSCNN ABLATION STUDY")
print("========================================")


# RMSCNN
rmscnn_train = np.load(
    os.path.join(RMSCNN_DIR, "train_features.npy")
)
rmscnn_val = np.load(
    os.path.join(RMSCNN_DIR, "val_features.npy")
)
rmscnn_test = np.load(
    os.path.join(RMSCNN_DIR, "test_features.npy")
)

y_train = np.load(
    os.path.join(RMSCNN_DIR, "train_labels.npy")
)
y_val = np.load(
    os.path.join(RMSCNN_DIR, "val_labels.npy")
)
y_test = np.load(
    os.path.join(RMSCNN_DIR, "test_labels.npy")
)


# Handcrafted
hc_train = np.load(
    os.path.join(HC_DIR, "train_features.npy")
)
hc_val = np.load(
    os.path.join(HC_DIR, "val_features.npy")
)
hc_test = np.load(
    os.path.join(HC_DIR, "test_features.npy")
)


# Fused
fused_train = np.load(
    os.path.join(FUSED_DIR, "train_features.npy")
)
fused_val = np.load(
    os.path.join(FUSED_DIR, "val_features.npy")
)
fused_test = np.load(
    os.path.join(FUSED_DIR, "test_features.npy")
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    model,
    X_train,
    X_val,
    X_test,
    name
):

    print("\n----------------------------------------")
    print(name)
    print("----------------------------------------")

    model.fit(
        X_train,
        y_train
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    val_acc = accuracy_score(
        y_val,
        val_pred
    )

    val_precision = precision_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_recall = recall_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0
    )

    val_auc = roc_auc_score(
        y_val,
        val_prob
    )

    test_acc = accuracy_score(
        y_test,
        test_pred
    )

    test_precision = precision_score(
        y_test,
        test_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        zero_division=0
    )

    test_auc = roc_auc_score(
        y_test,
        test_prob
    )

    test_cm = confusion_matrix(
        y_test,
        test_pred
    )

    print(
        f"Validation Accuracy : {val_acc:.4f}"
    )
    print(
        f"Validation F1       : {val_f1:.4f}"
    )
    print(
        f"Validation ROC-AUC  : {val_auc:.4f}"
    )

    print(
        f"\nTest Accuracy : {test_acc:.4f}"
    )
    print(
        f"Test Precision: {test_precision:.4f}"
    )
    print(
        f"Test Recall   : {test_recall:.4f}"
    )
    print(
        f"Test F1       : {test_f1:.4f}"
    )
    print(
        f"Test ROC-AUC  : {test_auc:.4f}"
    )

    print("\nTest Confusion Matrix:")
    print(test_cm)

    return {
        "val_accuracy": val_acc,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1,
        "val_auc": val_auc,
        "test_accuracy": test_acc,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1,
        "test_auc": test_auc,
        "test_cm": test_cm
    }


# ============================================================
# 1. HANDCRAFTED ONLY
# ============================================================

print("\n========================================")
print("1. HANDCRAFTED FEATURES ONLY")
print("========================================")

scaler_hc = StandardScaler()

hc_train_scaled = scaler_hc.fit_transform(
    hc_train
)

hc_val_scaled = scaler_hc.transform(
    hc_val
)

hc_test_scaled = scaler_hc.transform(
    hc_test
)

svm_hc = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

hc_results = evaluate(
    svm_hc,
    hc_train_scaled,
    hc_val_scaled,
    hc_test_scaled,
    "Handcrafted Only"
)


# ============================================================
# 2. RMSCNN ONLY
# ============================================================

print("\n========================================")
print("2. RMSCNN FEATURES ONLY")
print("========================================")

selector_rmscnn = SelectKBest(
    score_func=f_classif,
    k=30
)

rmscnn_train_selected = selector_rmscnn.fit_transform(
    rmscnn_train,
    y_train
)

rmscnn_val_selected = selector_rmscnn.transform(
    rmscnn_val
)

rmscnn_test_selected = selector_rmscnn.transform(
    rmscnn_test
)

scaler_rmscnn = StandardScaler()

rmscnn_train_scaled = scaler_rmscnn.fit_transform(
    rmscnn_train_selected
)

rmscnn_val_scaled = scaler_rmscnn.transform(
    rmscnn_val_selected
)

rmscnn_test_scaled = scaler_rmscnn.transform(
    rmscnn_test_selected
)

svm_rmscnn = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

rmscnn_results = evaluate(
    svm_rmscnn,
    rmscnn_train_scaled,
    rmscnn_val_scaled,
    rmscnn_test_scaled,
    "RMSCNN Only"
)


# ============================================================
# 3. RMSCNN + HANDCRAFTED FUSION
# ============================================================

print("\n========================================")
print("3. RMSCNN + HANDCRAFTED FUSION")
print("========================================")

selector_fusion = SelectKBest(
    score_func=f_classif,
    k=30
)

fusion_train_selected = selector_fusion.fit_transform(
    fused_train,
    y_train
)

fusion_val_selected = selector_fusion.transform(
    fused_val
)

fusion_test_selected = selector_fusion.transform(
    fused_test
)

scaler_fusion = StandardScaler()

fusion_train_scaled = scaler_fusion.fit_transform(
    fusion_train_selected
)

fusion_val_scaled = scaler_fusion.transform(
    fusion_val_selected
)

fusion_test_scaled = scaler_fusion.transform(
    fusion_test_selected
)

svm_fusion = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

fusion_results = evaluate(
    svm_fusion,
    fusion_train_scaled,
    fusion_val_scaled,
    fusion_test_scaled,
    "RMSCNN + Handcrafted Fusion"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

results = {
    "Handcrafted Only": hc_results,
    "RMSCNN Only": rmscnn_results,
    "RMSCNN + Handcrafted": fusion_results
}


with open(
    os.path.join(
        RESULT_DIR,
        "ablation_results.txt"
    ),
    "w"
) as f:

    f.write(
        "RMSCNN ABLATION STUDY\n"
    )

    f.write(
        "=====================\n\n"
    )

    for name, result in results.items():

        f.write(
            f"{name}\n"
        )

        f.write(
            "----------------------------------------\n"
        )

        f.write(
            f"Validation Accuracy: "
            f"{result['val_accuracy']:.4f}\n"
        )

        f.write(
            f"Validation Precision: "
            f"{result['val_precision']:.4f}\n"
        )

        f.write(
            f"Validation Recall: "
            f"{result['val_recall']:.4f}\n"
        )

        f.write(
            f"Validation F1: "
            f"{result['val_f1']:.4f}\n"
        )

        f.write(
            f"Validation ROC-AUC: "
            f"{result['val_auc']:.4f}\n\n"
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
            f"Test Confusion Matrix:\n"
        )

        f.write(
            f"{result['test_cm']}\n\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("ABLATION SUMMARY")
print("========================================")

print(
    f"Handcrafted Only : "
    f"{hc_results['test_accuracy']:.4f}"
)

print(
    f"RMSCNN Only      : "
    f"{rmscnn_results['test_accuracy']:.4f}"
)

print(
    f"RMSCNN + HC      : "
    f"{fusion_results['test_accuracy']:.4f}"
)

print("\nResults saved to:")
print(RESULT_DIR)