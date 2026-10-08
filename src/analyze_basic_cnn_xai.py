import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = r"E:\colorectal-cancer-project"

FEATURE_DIR = os.path.join(
    PROJECT_DIR,
    "data",
    "fused_basic_cnn_features"
)

RESULT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "basic_cnn_xai"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD FUSED FEATURES
# ============================================================

print("========================================")
print("BASIC CNN ANOVA FEATURE ANALYSIS")
print("========================================")

X_train = np.load(
    os.path.join(
        FEATURE_DIR,
        "train_features.npy"
    )
)

y_train = np.load(
    os.path.join(
        FEATURE_DIR,
        "train_labels.npy"
    )
)


print("\nInput feature shape:")
print(X_train.shape)

print(
    "\nFeature composition:"
)

print(
    "Basic CNN features     : 128"
)

print(
    "Handcrafted features   : 44"
)

print(
    "Total fused features   : 172"
)


# ============================================================
# ANOVA FEATURE SELECTION
# ============================================================

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

selector.fit(
    X_train,
    y_train
)


# ============================================================
# GET FEATURE INFORMATION
# ============================================================

selected_indices = selector.get_support(
    indices=True
)

anova_scores = selector.scores_

selected_scores = anova_scores[
    selected_indices
]

ranking_order = np.argsort(
    selected_scores
)[::-1]

ranked_indices = selected_indices[
    ranking_order
]

ranked_scores = selected_scores[
    ranking_order
]


# ============================================================
# DETERMINE FEATURE SOURCE
# ============================================================

basic_cnn_features = []
handcrafted_features = []

for index in ranked_indices:

    if index < 128:
        basic_cnn_features.append(index)

    else:
        handcrafted_features.append(
            index - 128
        )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("TOP 30 ANOVA FEATURES")
print("========================================")

for rank, (index, score) in enumerate(
    zip(ranked_indices, ranked_scores),
    start=1
):

    if index < 128:

        source = "Basic CNN"

        feature_number = index

    else:

        source = "Handcrafted"

        feature_number = index - 128

    print(
        f"Rank {rank:02d} | "
        f"Feature {feature_number:03d} | "
        f"{source:12s} | "
        f"ANOVA F-score: {score:.4f}"
    )


# ============================================================
# COUNT FEATURE SOURCES
# ============================================================

num_basic_cnn = len(
    basic_cnn_features
)

num_handcrafted = len(
    handcrafted_features
)

print("\n========================================")
print("FEATURE SOURCE SUMMARY")
print("========================================")

print(
    f"Basic CNN features selected : "
    f"{num_basic_cnn}/30"
)

print(
    f"Handcrafted features selected: "
    f"{num_handcrafted}/30"
)


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

np.save(
    os.path.join(
        RESULT_DIR,
        "selected_feature_indices.npy"
    ),
    ranked_indices
)

np.save(
    os.path.join(
        RESULT_DIR,
        "selected_feature_scores.npy"
    ),
    ranked_scores
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = os.path.join(
    RESULT_DIR,
    "anova_feature_analysis.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "BASIC CNN + HANDCRAFTED ANOVA FEATURE ANALYSIS\n"
    )

    f.write(
        "==============================================\n\n"
    )

    f.write(
        "Total fused features: 172\n"
    )

    f.write(
        "Basic CNN features: 128\n"
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
        "---------------\n"
    )

    for rank, (index, score) in enumerate(
        zip(ranked_indices, ranked_scores),
        start=1
    ):

        if index < 128:

            source = "Basic CNN"

            feature_number = index

        else:

            source = "Handcrafted"

            feature_number = index - 128

        f.write(
            f"Rank {rank:02d}: "
            f"Feature {feature_number:03d} | "
            f"{source} | "
            f"F-score = {score:.4f}\n"
        )

    f.write(
        "\nFEATURE SOURCE SUMMARY\n"
    )

    f.write(
        "-----------------------\n"
    )

    f.write(
        f"Basic CNN selected: "
        f"{num_basic_cnn}/30\n"
    )

    f.write(
        f"Handcrafted selected: "
        f"{num_handcrafted}/30\n"
    )


# ============================================================
# VISUALIZATION
# ============================================================

labels = []

for index in ranked_indices:

    if index < 128:

        labels.append(
            f"CNN-{index}"
        )

    else:

        labels.append(
            f"HC-{index - 128}"
        )


plt.figure(
    figsize=(14, 7)
)

plt.bar(
    range(1, 31),
    ranked_scores
)

plt.xlabel(
    "ANOVA Feature Rank"
)

plt.ylabel(
    "F-score"
)

plt.title(
    "Basic CNN + Handcrafted Features: "
    "Top 30 ANOVA Features"
)

plt.xticks(
    range(1, 31),
    labels,
    rotation=75
)

plt.tight_layout()

plot_path = os.path.join(
    RESULT_DIR,
    "basic_cnn_anova_top30.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("ANOVA ANALYSIS COMPLETE")
print("========================================")

print(
    f"Basic CNN selected     : "
    f"{num_basic_cnn}/30"
)

print(
    f"Handcrafted selected   : "
    f"{num_handcrafted}/30"
)

print(
    "\nReport saved to:"
)

print(
    report_path
)

print(
    "\nPlot saved to:"
)

print(
    plot_path
)