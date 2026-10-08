import numpy as np
from pathlib import Path

from sklearn.feature_selection import SelectKBest, f_classif


# ==================================================
# PATHS
# ==================================================

DATA_DIR = Path(
    "data/fused_features"
)

OUTPUT_DIR = Path(
    "results/resnet18_xai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# LOAD TRAINING DATA
# ==================================================

print("===== LOADING TRAINING DATA =====")

X_train = np.load(
    DATA_DIR / "train_features.npy"
)

y_train = np.load(
    DATA_DIR / "train_labels.npy"
)

print(
    "Training feature shape:",
    X_train.shape
)

print(
    "Training label shape:",
    y_train.shape
)


# ==================================================
# ANOVA FEATURE SELECTION
# ==================================================

print()
print("===== ANOVA FEATURE ANALYSIS =====")

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

selector.fit(
    X_train,
    y_train
)


# ==================================================
# GET FEATURE SCORES
# ==================================================

all_scores = selector.scores_

selected_indices = selector.get_support(
    indices=True
)

selected_scores = all_scores[
    selected_indices
]


# ==================================================
# SORT BY ANOVA SCORE
# ==================================================

ranking_order = np.argsort(
    selected_scores
)[::-1]

ranked_indices = selected_indices[
    ranking_order
]

ranked_scores = selected_scores[
    ranking_order
]


# ==================================================
# FEATURE NAME FUNCTION
# ==================================================

def get_feature_name(index):

    # ----------------------------------------------
    # Handcrafted features
    # ----------------------------------------------

    if index < 44:

        handcrafted_names = [

            "RGB_R_mean",
            "RGB_R_std",
            "RGB_G_mean",
            "RGB_G_std",
            "RGB_B_mean",
            "RGB_B_std",

            "HSV_H_mean",
            "HSV_H_std",
            "HSV_S_mean",
            "HSV_S_std",
            "HSV_V_mean",
            "HSV_V_std",

            "Lab_L_mean",
            "Lab_L_std",
            "Lab_a_mean",
            "Lab_a_std",
            "Lab_b_mean",
            "Lab_b_std",

            "YCbCr_Y_mean",
            "YCbCr_Y_std",
            "YCbCr_Cr_mean",
            "YCbCr_Cr_std",
            "YCbCr_Cb_mean",
            "YCbCr_Cb_std",

            "GLCM_contrast",
            "GLCM_dissimilarity",
            "GLCM_homogeneity",
            "GLCM_energy",
            "GLCM_correlation",
            "GLCM_ASM",

            "LBP_0",
            "LBP_1",
            "LBP_2",
            "LBP_3",
            "LBP_4",
            "LBP_5",
            "LBP_6",
            "LBP_7",
            "LBP_8",
            "LBP_9",

            "Shape_area",
            "Shape_perimeter",
            "Shape_circularity",
            "Shape_aspect_ratio"
        ]

        return handcrafted_names[index]

    # ----------------------------------------------
    # ResNet18 features
    # ----------------------------------------------

    else:

        resnet_index = index - 44

        return (
            f"ResNet18_feature_{resnet_index}"
        )


# ==================================================
# PRINT FEATURE RANKING
# ==================================================

print()
print(
    "===== TOP 30 ANOVA FEATURES ====="
)

print()

for rank, (index, score) in enumerate(
    zip(
        ranked_indices,
        ranked_scores
    ),
    start=1
):

    feature_name = get_feature_name(
        index
    )

    print(
        f"{rank:2d}. "
        f"Index={index:3d} | "
        f"{feature_name:<30} | "
        f"F-score={score:.6f}"
    )


# ==================================================
# SAVE FEATURE RANKING
# ==================================================

ranking_file = (
    OUTPUT_DIR /
    "anova_feature_ranking.txt"
)

with open(
    ranking_file,
    "w"
) as f:

    f.write(
        "RESNET18 + HANDCRAFTED FEATURE XAI\n"
    )

    f.write(
        "====================================\n\n"
    )

    f.write(
        "Feature selection: SelectKBest\n"
    )

    f.write(
        "Scoring method: ANOVA F-test\n"
    )

    f.write(
        "Number of selected features: 30\n\n"
    )

    f.write(
        "RANK | INDEX | FEATURE | F-SCORE\n"
    )

    f.write(
        "-" * 70 + "\n"
    )