import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

FUSED_DIR = (
    PROJECT_DIR
    / "data"
    / "fused_ghostnet_features"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "results"
    / "ghostnet_xai"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_DEEP_FEATURES = 960
NUM_HANDCRAFTED_FEATURES = 44
TOTAL_FEATURES = (
    NUM_DEEP_FEATURES
    + NUM_HANDCRAFTED_FEATURES
)

TOP_K = 30


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("GHOSTNET ANOVA / XAI ANALYSIS")
print("=" * 60)

print("\nLoading fused features...")

train_features = np.load(
    FUSED_DIR / "train_fused.npy"
)

train_labels = np.load(
    PROJECT_DIR
    / "data"
    / "ghostnet_features"
    / "train_labels.npy"
)


print(
    "Training features:",
    train_features.shape
)

print(
    "Training labels:",
    train_labels.shape
)


# ============================================================
# VERIFY FEATURE DIMENSIONS
# ============================================================

if train_features.shape[1] != TOTAL_FEATURES:

    raise ValueError(
        f"Expected {TOTAL_FEATURES} features, "
        f"but found {train_features.shape[1]}"
    )


print()
print(
    f"Deep features       : {NUM_DEEP_FEATURES}"
)

print(
    f"Handcrafted features: {NUM_HANDCRAFTED_FEATURES}"
)

print(
    f"Total features      : {TOTAL_FEATURES}"
)


# ============================================================
# ANOVA FEATURE ANALYSIS
# ============================================================

print("\n===== ANOVA FEATURE ANALYSIS =====")

scores, p_values = f_classif(
    train_features,
    train_labels
)


# Replace invalid values if any

scores = np.nan_to_num(
    scores,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

p_values = np.nan_to_num(
    p_values,
    nan=1.0,
    posinf=1.0,
    neginf=1.0
)


# ============================================================
# SORT FEATURES BY ANOVA SCORE
# ============================================================

ranked_indices = np.argsort(
    scores
)[::-1]


top_indices = ranked_indices[
    :TOP_K
]

top_scores = scores[
    top_indices
]

top_p_values = p_values[
    top_indices
]


# ============================================================
# DETERMINE FEATURE TYPE
# ============================================================

def get_feature_type(index):

    if index < NUM_DEEP_FEATURES:

        return "GhostNet"

    return "Handcrafted"


feature_types = [
    get_feature_type(index)
    for index in top_indices
]


# ============================================================
# PRINT TOP FEATURES
# ============================================================

print("\nTop 30 ANOVA features:")

print(
    "-" * 60
)

for rank, (
    index,
    score,
    p_value,
    feature_type
) in enumerate(
    zip(
        top_indices,
        top_scores,
        top_p_values,
        feature_types
    ),
    start=1
):

    print(
        f"{rank:2d}. "
        f"Feature {index:4d} | "
        f"Type: {feature_type:12s} | "
        f"F-score: {score:.6f} | "
        f"p-value: {p_value:.6e}"
    )


# ============================================================
# COUNT FEATURE TYPES
# ============================================================

ghostnet_count = sum(
    feature_type == "GhostNet"
    for feature_type in feature_types
)

handcrafted_count = sum(
    feature_type == "Handcrafted"
    for feature_type in feature_types
)


print("\n===== FEATURE TYPE SUMMARY =====")

print(
    "Top-30 GhostNet features:",
    ghostnet_count
)

print(
    "Top-30 handcrafted features:",
    handcrafted_count
)


# ============================================================
# SAVE FEATURE INDICES
# ============================================================

np.save(
    OUTPUT_DIR / "top30_indices.npy",
    top_indices
)

np.save(
    OUTPUT_DIR / "top30_scores.npy",
    top_scores
)

np.save(
    OUTPUT_DIR / "top30_p_values.npy",
    top_p_values
)


# ============================================================
# SAVE TEXT REPORT
# ============================================================

report_path = (
    OUTPUT_DIR
    / "anova_feature_ranking.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GHOSTNET ANOVA FEATURE ANALYSIS\n"
    )

    f.write(
        "================================\n\n"
    )

    f.write(
        f"Deep features: "
        f"{NUM_DEEP_FEATURES}\n"
    )

    f.write(
        f"Handcrafted features: "
        f"{NUM_HANDCRAFTED_FEATURES}\n"
    )

    f.write(
        f"Total fused features: "
        f"{TOTAL_FEATURES}\n\n"
    )

    f.write(
        "TOP 30 FEATURES\n"
    )

    f.write(
        "---------------\n"
    )

    for rank, (
        index,
        score,
        p_value,
        feature_type
    ) in enumerate(
        zip(
            top_indices,
            top_scores,
            top_p_values,
            feature_types
        ),
        start=1
    ):

        f.write(
            f"{rank:2d}. "
            f"Feature {index:4d} | "
            f"{feature_type:12s} | "
            f"F-score = {score:.6f} | "
            f"p-value = {p_value:.6e}\n"
        )

    f.write("\n")

    f.write(
        f"Top-30 GhostNet features: "
        f"{ghostnet_count}\n"
    )

    f.write(
        f"Top-30 handcrafted features: "
        f"{handcrafted_count}\n"
    )


# ============================================================
# VISUALIZATION
# ============================================================

print("\nCreating ANOVA visualization...")


# Reverse order so the highest score appears at the top

plot_indices = top_indices[::-1]

plot_scores = top_scores[::-1]

plot_labels = [
    f"{idx} ({get_feature_type(idx)})"
    for idx in plot_indices
]


plt.figure(
    figsize=(10, 12)
)

plt.barh(
    plot_labels,
    plot_scores
)

plt.xlabel(
    "ANOVA F-score"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "GhostNet + Handcrafted Features: Top 30 ANOVA Features"
)

plt.tight_layout()


plot_path = (
    OUTPUT_DIR
    / "top30_anova_features.png"
)

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 60)

print(
    "GHOSTNET ANOVA / XAI ANALYSIS COMPLETE"
)

print("=" * 60)

print(
    "Top-30 GhostNet features:",
    ghostnet_count
)

print(
    "Top-30 handcrafted features:",
    handcrafted_count
)

print()
print(
    "Report saved to:"
)

print(
    report_path
)

print()
print(
    "Visualization saved to:"
)

print(
    plot_path
)