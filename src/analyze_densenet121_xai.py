import numpy as np
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/fused_densenet121_features")
RESULTS_DIR = Path("results/densenet121_xai")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD TRAINING FEATURES
# ============================================================

X_train = np.load(
    DATA_DIR / "train_features.npy"
)

y_train = np.load(
    DATA_DIR / "train_labels.npy"
)


# ============================================================
# FEATURE INFORMATION
# ============================================================

DENSENET_FEATURES = 1024
HANDCRAFTED_FEATURES = 44

TOTAL_FEATURES = (
    DENSENET_FEATURES +
    HANDCRAFTED_FEATURES
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
# GET FEATURE SCORES
# ============================================================

scores = selector.scores_

selected_indices = selector.get_support(
    indices=True
)


# ============================================================
# SORT ALL FEATURES BY ANOVA SCORE
# ============================================================

valid_indices = np.where(
    np.isfinite(scores)
)[0]

sorted_indices = valid_indices[
    np.argsort(
        scores[valid_indices]
    )[::-1]
]


# ============================================================
# FEATURE NAME FUNCTION
# ============================================================

def get_feature_name(index):

    if index < DENSENET_FEATURES:

        return (
            f"DenseNet121_feature_{index}"
        )

    handcrafted_index = (
        index - DENSENET_FEATURES
    )

    return (
        f"Handcrafted_feature_"
        f"{handcrafted_index}"
    )


# ============================================================
# DISPLAY TOP 30
# ============================================================

print("=" * 70)
print("DENSENET121 XAI / ANOVA FEATURE ANALYSIS")
print("=" * 70)

print()
print("Total fused features:", TOTAL_FEATURES)
print("DenseNet121 features:", DENSENET_FEATURES)
print("Handcrafted features:", HANDCRAFTED_FEATURES)
print("Selected features:", len(selected_indices))

print()
print("TOP 30 FEATURES")
print("-" * 70)

dense_count = 0
handcrafted_count = 0

for rank, index in enumerate(
    selected_indices,
    start=1
):

    score = scores[index]

    feature_name = get_feature_name(
        index
    )

    if index < DENSENET_FEATURES:
        feature_type = "DenseNet121"
        dense_count += 1
    else:
        feature_type = "Handcrafted"
        handcrafted_count += 1

    print(
        f"{rank:2d}. "
        f"Index={index:4d} | "
        f"{feature_name:<35} | "
        f"Type={feature_type:<12} | "
        f"F-score={score:.4f}"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

output_file = (
    RESULTS_DIR /
    "anova_feature_ranking.txt"
)


with open(
    output_file,
    "w"
) as f:

    f.write(
        "DENSENET121 ANOVA FEATURE ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Total fused features: "
        f"{TOTAL_FEATURES}\n"
    )

    f.write(
        f"DenseNet121 features: "
        f"{DENSENET_FEATURES}\n"
    )

    f.write(
        f"Handcrafted features: "
        f"{HANDCRAFTED_FEATURES}\n"
    )

    f.write(
        f"Selected top features: "
        f"{len(selected_indices)}\n\n"
    )

    f.write(
        "TOP 30 FEATURES\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for rank, index in enumerate(
        selected_indices,
        start=1
    ):

        score = scores[index]

        feature_name = get_feature_name(
            index
        )

        if index < DENSENET_FEATURES:
            feature_type = "DenseNet121"
        else:
            feature_type = "Handcrafted"

        f.write(
            f"{rank:2d}. "
            f"Index={index:4d} | "
            f"{feature_name:<35} | "
            f"Type={feature_type:<12} | "
            f"F-score={score:.4f}\n"
        )

    f.write("\n")
    f.write("FEATURE SOURCE SUMMARY\n")
    f.write("-" * 40 + "\n")

    f.write(
        f"DenseNet121 selected: "
        f"{dense_count}\n"
    )

    f.write(
        f"Handcrafted selected: "
        f"{handcrafted_count}\n"
    )


# ============================================================
# SAVE SELECTED INDICES
# ============================================================

np.save(
    RESULTS_DIR /
    "selected_feature_indices.npy",
    selected_indices
)


print()
print("=" * 70)
print("FEATURE SOURCE SUMMARY")
print("=" * 70)

print(
    "DenseNet121 selected:",
    dense_count
)

print(
    "Handcrafted selected:",
    handcrafted_count
)

print()
print("Saved:")
print(output_file)

print()
print(
    "DENSENET121 ANOVA ANALYSIS COMPLETE!"
)