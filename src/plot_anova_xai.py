import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif


# ==================================================
# PATHS
# ==================================================

DATA_DIR = Path("data/fused_features")

OUTPUT_DIR = Path("results/resnet18_xai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD TRAINING DATA
# ==================================================

X_train = np.load(
    DATA_DIR / "train_features.npy"
)

y_train = np.load(
    DATA_DIR / "train_labels.npy"
)


# ==================================================
# ANOVA
# ==================================================

selector = SelectKBest(
    score_func=f_classif,
    k=30
)

selector.fit(
    X_train,
    y_train
)


selected_indices = selector.get_support(
    indices=True
)

selected_scores = selector.scores_[
    selected_indices
]


# ==================================================
# SORT FEATURES
# ==================================================

order = np.argsort(
    selected_scores
)[::-1]

selected_indices = selected_indices[
    order
]

selected_scores = selected_scores[
    order
]


# ==================================================
# FEATURE NAMES
# ==================================================

feature_names = [
    f"ResNet18_{index - 44}"
    if index >= 44
    else f"Handcrafted_{index}"
    for index in selected_indices
]


# ==================================================
# PLOT
# ==================================================

plt.figure(
    figsize=(12, 8)
)

y_positions = np.arange(
    len(feature_names)
)

plt.barh(
    y_positions,
    selected_scores
)

plt.yticks(
    y_positions,
    feature_names
)

plt.gca().invert_yaxis()

plt.xlabel(
    "ANOVA F-score"
)

plt.ylabel(
    "Selected ResNet18 Feature"
)

plt.title(
    "Top 30 ANOVA-Selected Features"
)

plt.tight_layout()


# ==================================================
# SAVE
# ==================================================

output_file = (
    OUTPUT_DIR /
    "top30_anova_features.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print()
print("========================================")
print("      ANOVA XAI VISUALIZATION")
print("========================================")

print(
    "Selected features:",
    len(selected_indices)
)

print(
    "Visualization saved to:"
)

print(
    output_file
)