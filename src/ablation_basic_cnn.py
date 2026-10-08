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

DEEP_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "basic_cnn_features"
)

HC_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "handcrafted_features"
)

FUSED_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_basic_cnn_features"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "basic_cnn_ablation"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("BASIC CNN ABLATION STUDY")
print("========================================")

# Basic CNN
X_train_deep = np.load(
    os.path.join(DEEP_DIR, "train_features.npy")
)
y_train = np.load(
    os.path.join(DEEP_DIR, "train_labels.npy")
)

X_val_deep = np.load(
    os.path.join(DEEP_DIR, "val_features.npy")
)
y_val = np.load(
    os.path.join(DEEP_DIR, "val_labels.npy")
)

X_test_deep = np.load(
    os.path.join(DEEP_DIR, "test_features.npy")
)
y_test = np.load(
    os.path.join(DEEP_DIR, "test_labels.npy")
)


# Handcrafted
X_train_hc = np.load(
    os.path.join(HC_DIR, "train_features.npy")
)

X_val_hc = np.load(
    os.path.join(HC_DIR, "val_features.npy")
)

X_test_hc = np.load(
    os.path.join(HC_DIR, "test_features.npy")
)


# Fused
X_train_fused = np.load(
    os.path.join(FUSED_DIR, "train_features.npy")
)

X_val_fused = np.load(
    os.path.join(FUSED_DIR, "val_features.npy")
)

X_test_fused = np.load(
    os.path.join(FUSED_DIR, "test_features.npy")
)


# ============================================================
# CHECK LABEL CONSISTENCY
# ============================================================

assert np.array_equal(
    y_train,
    np.load(os.path.join(HC_DIR, "train_labels.npy"))
)

assert np.array_equal(
    y_val,
    np.load(os.path.join(HC_DIR, "val_labels.npy"))
)

assert np.array_equal(
    y_test,
    np.load(os.path.join(HC_DIR, "test_labels.npy"))
)

assert np.array_equal(
    y_train,
    np.load(os.path.join(FUSED_DIR, "train_labels.npy"))
)

assert np.array_equal(
    y_val,
    np.load(os.path.join(FUSED_DIR, "val_labels.npy"))
)

assert np.array_equal(
    y_test,
    np.load(os.path.join(FUSED_DIR, "test_labels.npy"))
)


print("\nFeature dimensions:")
print("Handcrafted:", X_train_hc.shape)
print("Basic CNN  :", X_train_deep.shape)
print("Fusion     :", X_train_fused.shape)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    name
):

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)

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

    auc = roc_auc_score(
        y,
        probabilities[:, 1]
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(f"\n{name}")
    print("----------------------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "cm": cm
    }


# ============================================================
# 1. HANDCRAFTED FEATURES ONLY
# ============================================================

print("\n========================================")
print("1. HANDCRAFTED FEATURES ONLY")
print("========================================")

selector_hc = SelectKBest(
    score_func=f_classif,
    k=30
)

X_train_hc_selected = selector_hc.fit_transform(
    X_train_hc,
    y_train
)

X_val_hc_selected = selector_hc.transform(
    X_val_hc
)

X_test_hc_selected = selector_hc.transform(
    X_test_hc
)

scaler_hc = StandardScaler()

X_train_hc_scaled = scaler_hc.fit_transform(
    X_train_hc_selected
)

X_val_hc_scaled = scaler_hc.transform(
    X_val_hc_selected
)

X_test_hc_scaled = scaler_hc.transform(
    X_test_hc_selected
)

svm_hc = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_hc.fit(
    X_train_hc_scaled,
    y_train
)

hc_val_results = evaluate_model(
    svm_hc,
    X_val_hc_scaled,
    y_val,
    "HANDCRAFTED — VALIDATION"
)

hc_test_results = evaluate_model(
    svm_hc,
    X_test_hc_scaled,
    y_test,
    "HANDCRAFTED — TEST"
)


# ============================================================
# 2. BASIC CNN FEATURES ONLY
# ============================================================

print("\n========================================")
print("2. BASIC CNN FEATURES ONLY")
print("========================================")

selector_deep = SelectKBest(
    score_func=f_classif,
    k=30
)

X_train_deep_selected = selector_deep.fit_transform(
    X_train_deep,
    y_train
)

X_val_deep_selected = selector_deep.transform(
    X_val_deep
)

X_test_deep_selected = selector_deep.transform(
    X_test_deep
)

scaler_deep = StandardScaler()

X_train_deep_scaled = scaler_deep.fit_transform(
    X_train_deep_selected
)

X_val_deep_scaled = scaler_deep.transform(
    X_val_deep_selected
)

X_test_deep_scaled = scaler_deep.transform(
    X_test_deep_selected
)

svm_deep = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_deep.fit(
    X_train_deep_scaled,
    y_train
)

deep_val_results = evaluate_model(
    svm_deep,
    X_val_deep_scaled,
    y_val,
    "BASIC CNN ONLY — VALIDATION"
)

deep_test_results = evaluate_model(
    svm_deep,
    X_test_deep_scaled,
    y_test,
    "BASIC CNN ONLY — TEST"
)


# ============================================================
# 3. BASIC CNN + HANDCRAFTED FEATURES
# ============================================================

print("\n========================================")
print("3. BASIC CNN + HANDCRAFTED FEATURES")
print("========================================")

selector_fused = SelectKBest(
    score_func=f_classif,
    k=30
)

X_train_fused_selected = selector_fused.fit_transform(
    X_train_fused,
    y_train
)

X_val_fused_selected = selector_fused.transform(
    X_val_fused
)

X_test_fused_selected = selector_fused.transform(
    X_test_fused
)

scaler_fused = StandardScaler()

X_train_fused_scaled = scaler_fused.fit_transform(
    X_train_fused_selected
)

X_val_fused_scaled = scaler_fused.transform(
    X_val_fused_selected
)

X_test_fused_scaled = scaler_fused.transform(
    X_test_fused_selected
)

svm_fused = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
    random_state=42
)

svm_fused.fit(
    X_train_fused_scaled,
    y_train
)

fused_val_results = evaluate_model(
    svm_fused,
    X_val_fused_scaled,
    y_val,
    "FUSION — VALIDATION"
)

fused_test_results = evaluate_model(
    svm_fused,
    X_test_fused_scaled,
    y_test,
    "FUSION — TEST"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    RESULT_DIR,
    "basic_cnn_ablation_results.txt"
)

with open(
    results_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BASIC CNN ABLATION STUDY\n"
    )

    f.write(
        "========================\n\n"
    )

    f.write(
        "Configurations:\n"
    )

    f.write(
        "1. Handcrafted features only\n"
    )

    f.write(
        "2. Basic CNN features only\n"
    )

    f.write(
        "3. Basic CNN + Handcrafted features\n\n"
    )

    f.write(
        "Common pipeline:\n"
    )

    f.write(
        "Features -> ANOVA Top-30 -> StandardScaler -> RBF-SVM\n\n"
    )

    # Validation
    f.write(
        "VALIDATION RESULTS\n"
    )

    f.write(
        "------------------\n"
    )

    f.write(
        f"Handcrafted: "
        f"Accuracy={hc_val_results['accuracy']:.4f}, "
        f"Precision={hc_val_results['precision']:.4f}, "
        f"Recall={hc_val_results['recall']:.4f}, "
        f"F1={hc_val_results['f1']:.4f}, "
        f"AUC={hc_val_results['auc']:.4f}\n"
    )

    f.write(
        f"Basic CNN: "
        f"Accuracy={deep_val_results['accuracy']:.4f}, "
        f"Precision={deep_val_results['precision']:.4f}, "
        f"Recall={deep_val_results['recall']:.4f}, "
        f"F1={deep_val_results['f1']:.4f}, "
        f"AUC={deep_val_results['auc']:.4f}\n"
    )

    f.write(
        f"Fusion: "
        f"Accuracy={fused_val_results['accuracy']:.4f}, "
        f"Precision={fused_val_results['precision']:.4f}, "
        f"Recall={fused_val_results['recall']:.4f}, "
        f"F1={fused_val_results['f1']:.4f}, "
        f"AUC={fused_val_results['auc']:.4f}\n\n"
    )

    # Test
    f.write(
        "TEST RESULTS\n"
    )

    f.write(
        "------------\n"
    )

    f.write(
        f"Handcrafted: "
        f"Accuracy={hc_test_results['accuracy']:.4f}, "
        f"Precision={hc_test_results['precision']:.4f}, "
        f"Recall={hc_test_results['recall']:.4f}, "
        f"F1={hc_test_results['f1']:.4f}, "
        f"AUC={hc_test_results['auc']:.4f}\n"
    )

    f.write(
        f"Basic CNN: "
        f"Accuracy={deep_test_results['accuracy']:.4f}, "
        f"Precision={deep_test_results['precision']:.4f}, "
        f"Recall={deep_test_results['recall']:.4f}, "
        f"F1={deep_test_results['f1']:.4f}, "
        f"AUC={deep_test_results['auc']:.4f}\n"
    )

    f.write(
        f"Fusion: "
        f"Accuracy={fused_test_results['accuracy']:.4f}, "
        f"Precision={fused_test_results['precision']:.4f}, "
        f"Recall={fused_test_results['recall']:.4f}, "
        f"F1={fused_test_results['f1']:.4f}, "
        f"AUC={fused_test_results['auc']:.4f}\n\n"
    )

    f.write(
        "TEST CONFUSION MATRICES\n"
    )

    f.write(
        "------------------------\n"
    )

    f.write(
        f"Handcrafted:\n"
        f"{hc_test_results['cm']}\n\n"
    )

    f.write(
        f"Basic CNN:\n"
        f"{deep_test_results['cm']}\n\n"
    )

    f.write(
        f"Fusion:\n"
        f"{fused_test_results['cm']}\n"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("ABLATION STUDY COMPLETE")
print("========================================")

print("\nTEST SUMMARY")
print("----------------------------------------")

print(
    f"Handcrafted Only : "
    f"{hc_test_results['accuracy']:.4f} Accuracy | "
    f"{hc_test_results['f1']:.4f} F1 | "
    f"{hc_test_results['auc']:.4f} AUC"
)

print(
    f"Basic CNN Only   : "
    f"{deep_test_results['accuracy']:.4f} Accuracy | "
    f"{deep_test_results['f1']:.4f} F1 | "
    f"{deep_test_results['auc']:.4f} AUC"
)

print(
    f"Fusion           : "
    f"{fused_test_results['accuracy']:.4f} Accuracy | "
    f"{fused_test_results['f1']:.4f} F1 | "
    f"{fused_test_results['auc']:.4f} AUC"
)

print("\nResults saved to:")
print(results_file)