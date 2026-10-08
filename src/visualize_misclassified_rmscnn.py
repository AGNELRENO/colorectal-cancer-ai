import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

DATA_DIR = os.path.join(
    PROJECT_DIR,
    "data"
)

RMSCNN_DIR = os.path.join(
    DATA_DIR,
    "rmscnn_features"
)

HC_DIR = os.path.join(
    DATA_DIR,
    "handcrafted_features"
)

FUSED_DIR = os.path.join(
    DATA_DIR,
    "fused_rmscnn_features"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "rmscnn_xai",
    "misclassified"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD FEATURES
# ============================================================

X_train = np.load(
    os.path.join(
        FUSED_DIR,
        "train_features.npy"
    )
)

X_val = np.load(
    os.path.join(
        FUSED_DIR,
        "val_features.npy"
    )
)

X_test = np.load(
    os.path.join(
        FUSED_DIR,
        "test_features.npy"
    )
)

y_train = np.load(
    os.path.join(
        FUSED_DIR,
        "train_labels.npy"
    )
)

y_val = np.load(
    os.path.join(
        FUSED_DIR,
        "val_labels.npy"
    )
)

y_test = np.load(
    os.path.join(
        FUSED_DIR,
        "test_labels.npy"
    )
)


# ============================================================
# LOAD TEST IMAGE PATHS
# ============================================================

test_samples = np.load(
    os.path.join(
        DATA_DIR,
        "test_samples.npy"
    ),
    allow_pickle=True
)


# ============================================================
# ANOVA
# ============================================================

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


# ============================================================
# STANDARDIZATION
# ============================================================

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


# ============================================================
# TRAIN RBF-SVM
# ============================================================

print("========================================")
print("RMSCNN MISCLASSIFICATION ANALYSIS")
print("========================================")

print("\nTraining RBF-SVM...")

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
# TEST PREDICTIONS
# ============================================================

test_predictions = svm.predict(
    X_test_scaled
)

test_probabilities = svm.predict_proba(
    X_test_scaled
)

# Probability of predicted class
predicted_confidences = np.max(
    test_probabilities,
    axis=1
)


# ============================================================
# FIND MISCLASSIFICATIONS
# ============================================================

misclassified_indices = np.where(
    test_predictions != y_test
)[0]


print(
    "\nTotal test samples:",
    len(y_test)
)

print(
    "Correct predictions:",
    np.sum(
        test_predictions == y_test
    )
)

print(
    "Misclassified:",
    len(misclassified_indices)
)


# ============================================================
# SAVE MISCLASSIFICATION REPORT
# ============================================================

class_names = {
    0: "CRC",
    1: "Non-CRC"
}

report_path = os.path.join(
    RESULT_DIR,
    "misclassification_report.txt"
)


with open(
    report_path,
    "w"
) as f:

    f.write(
        "RMSCNN + Handcrafted Fusion-SVM\n"
    )

    f.write(
        "Misclassification Analysis\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        f"Total test samples: "
        f"{len(y_test)}\n"
    )

    f.write(
        f"Correct predictions: "
        f"{np.sum(test_predictions == y_test)}\n"
    )

    f.write(
        f"Misclassified samples: "
        f"{len(misclassified_indices)}\n\n"
    )


# ============================================================
# VISUALIZE EACH ERROR
# ============================================================

print("\n========================================")
print("MISCLASSIFIED IMAGES")
print("========================================")


for count, index in enumerate(
    misclassified_indices,
    start=1
):

    image_path = test_samples[index][0]

    true_label = int(
        y_test[index]
    )

    predicted_label = int(
        test_predictions[index]
    )

    confidence = float(
        predicted_confidences[index]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    plt.figure(
        figsize=(7, 7)
    )

    plt.imshow(
        image
    )

    plt.axis("off")

    plt.title(
        f"True: {class_names[true_label]} | "
        f"Predicted: {class_names[predicted_label]}\n"
        f"Confidence: {confidence:.4f}"
    )

    plt.tight_layout()

    filename = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        RESULT_DIR,
        f"{count:02d}_{filename}_misclassified.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    with open(
        report_path,
        "a"
    ) as f:

        f.write(
            f"Error {count}\n"
        )

        f.write(
            f"Image: {image_path}\n"
        )

        f.write(
            f"True class: "
            f"{class_names[true_label]}\n"
        )

        f.write(
            f"Predicted class: "
            f"{class_names[predicted_label]}\n"
        )

        f.write(
            f"Confidence: "
            f"{confidence:.4f}\n"
        )

        f.write(
            f"Visualization: "
            f"{output_path}\n\n"
        )

    print(
        f"\nError {count}"
    )

    print(
        "Image:",
        os.path.basename(image_path)
    )

    print(
        "True:",
        class_names[true_label]
    )

    print(
        "Predicted:",
        class_names[predicted_label]
    )

    print(
        f"Confidence: {confidence:.4f}"
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# SUMMARY
# ============================================================

crc_to_noncrc = np.sum(
    (y_test == 0) &
    (test_predictions == 1)
)

noncrc_to_crc = np.sum(
    (y_test == 1) &
    (test_predictions == 0)
)


with open(
    report_path,
    "a"
) as f:

    f.write(
        "ERROR DIRECTION SUMMARY\n"
    )

    f.write(
        "=======================\n"
    )

    f.write(
        f"CRC -> Non-CRC: "
        f"{crc_to_noncrc}\n"
    )

    f.write(
        f"Non-CRC -> CRC: "
        f"{noncrc_to_crc}\n"
    )


print("\n========================================")
print("MISCLASSIFICATION ANALYSIS COMPLETE")
print("========================================")

print(
    "CRC -> Non-CRC:",
    crc_to_noncrc
)

print(
    "Non-CRC -> CRC:",
    noncrc_to_crc
)

print("\nResults saved to:")
print(RESULT_DIR)

print("\nReport saved to:")
print(report_path)