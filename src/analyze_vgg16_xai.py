import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# PATHS
# ============================================================

VGG16_DIR = Path("data/vgg16_features")
FUSED_DIR = Path("data/fused_vgg16_features")
HANDCRAFTED_DIR = Path("data/handcrafted_features")

RESULTS_DIR = Path("results/vgg16_xai")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FEATURES
# ============================================================

print("===== LOADING VGG16 FEATURES =====")

vgg_train = np.load(
    VGG16_DIR / "train_features.npy"
)

vgg_labels = np.load(
    VGG16_DIR / "train_labels.npy"
)

fused_train = np.load(
    FUSED_DIR / "train_features.npy"
)

hand_train = np.load(
    HANDCRAFTED_DIR / "train_features.npy"
)

print("VGG16 feature shape:")
print(vgg_train.shape)

print("Handcrafted feature shape:")
print(hand_train.shape)

print("Fused feature shape:")
print(fused_train.shape)


# ============================================================
# ANOVA FEATURE SELECTION
# ============================================================

print()
print("===== ANOVA FEATURE ANALYSIS =====")

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

selector.fit(
    fused_train,
    fused_train_labels := vgg_labels
)

selected_indices = selector.get_support(
    indices=True
)

scores = selector.scores_

print()
print("Top 30 selected feature indices:")

for rank, index in enumerate(
    selected_indices,
    start=1
):
    print(
        f"{rank:02d}. Feature {index} "
        f"F-score = {scores[index]:.6f}"
    )


# ============================================================
# MAP FEATURE SOURCE
# ============================================================

print()
print("===== FEATURE SOURCE ANALYSIS =====")

VGG16_FEATURE_COUNT = 25088
HANDCRAFTED_FEATURE_COUNT = 44

vgg_features = []
handcrafted_features = []

for index in selected_indices:

    if index < VGG16_FEATURE_COUNT:

        vgg_features.append(index)

    else:

        handcrafted_features.append(
            index - VGG16_FEATURE_COUNT
        )


print(
    "VGG16 features selected:",
    len(vgg_features)
)

print(
    "Handcrafted features selected:",
    len(handcrafted_features)
)


# ============================================================
# HANDCRAFTED FEATURE NAMES
# ============================================================

handcrafted_names = [

    # RGB
    "RGB_R_mean",
    "RGB_R_std",
    "RGB_G_mean",
    "RGB_G_std",
    "RGB_B_mean",
    "RGB_B_std",

    # HSV
    "HSV_H_mean",
    "HSV_H_std",
    "HSV_S_mean",
    "HSV_S_std",
    "HSV_V_mean",
    "HSV_V_std",

    # Lab
    "Lab_L_mean",
    "Lab_L_std",
    "Lab_a_mean",
    "Lab_a_std",
    "Lab_b_mean",
    "Lab_b_std",

    # YCrCb
    "YCrCb_Y_mean",
    "YCrCb_Y_std",
    "YCrCb_Cr_mean",
    "YCrCb_Cr_std",
    "YCrCb_Cb_mean",
    "YCrCb_Cb_std",

    # GLCM
    "GLCM_contrast",
    "GLCM_dissimilarity",
    "GLCM_homogeneity",
    "GLCM_energy",
    "GLCM_correlation",
    "GLCM_ASM",

    # LBP
    "LBP_bin_0",
    "LBP_bin_1",
    "LBP_bin_2",
    "LBP_bin_3",
    "LBP_bin_4",
    "LBP_bin_5",
    "LBP_bin_6",
    "LBP_bin_7",
    "LBP_bin_8",
    "LBP_bin_9",

    # Shape
    "Shape_area",
    "Shape_perimeter",
    "Shape_circularity",
    "Shape_aspect_ratio"
]


# ============================================================
# SAVE REPORT
# ============================================================

report_file = (
    RESULTS_DIR /
    "anova_feature_ranking.txt"
)

with open(report_file, "w") as f:

    f.write(
        "VGG16 ANOVA FEATURE ANALYSIS\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        "Total fused features: 25132\n"
    )

    f.write(
        "VGG16 features: 25088\n"
    )

    f.write(
        "Handcrafted features: 44\n"
    )

    f.write(
        "Selected features: 30\n\n"
    )

    f.write(
        "TOP 30 FEATURES\n"
    )

    f.write(
        "----------------\n"
    )

    for rank, index in enumerate(
        selected_indices,
        start=1
    ):

        if index < VGG16_FEATURE_COUNT:

            source = "VGG16"

            feature_name = (
                f"VGG16_feature_{index}"
            )

        else:

            handcrafted_index = (
                index -
                VGG16_FEATURE_COUNT
            )

            source = "Handcrafted"

            if (
                handcrafted_index <
                len(handcrafted_names)
            ):

                feature_name = (
                    handcrafted_names[
                        handcrafted_index
                    ]
                )

            else:

                feature_name = (
                    f"Handcrafted_feature_"
                    f"{handcrafted_index}"
                )

        f.write(
            f"{rank:02d}. "
            f"Index={index}, "
            f"Source={source}, "
            f"Feature={feature_name}, "
            f"F-score={scores[index]:.6f}\n"
        )

    f.write("\n")

    f.write(
        "FEATURE SOURCE SUMMARY\n"
    )

    f.write(
        "-----------------------\n"
    )

    f.write(
        f"VGG16 selected: "
        f"{len(vgg_features)}\n"
    )

    f.write(
        f"Handcrafted selected: "
        f"{len(handcrafted_features)}\n"
    )


# ============================================================
# SAVE SELECTED INDICES
# ============================================================

np.save(
    RESULTS_DIR /
    "selected_feature_indices.npy",
    selected_indices
)

np.save(
    RESULTS_DIR /
    "anova_scores.npy",
    scores
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("===== FEATURE SOURCE SUMMARY =====")

print(
    f"VGG16 selected       : "
    f"{len(vgg_features)}"
)

print(
    f"Handcrafted selected : "
    f"{len(handcrafted_features)}"
)

print()
print(
    "Report saved to:",
    report_file
)

print(
    "===== VGG16 ANOVA XAI COMPLETE ====="
)