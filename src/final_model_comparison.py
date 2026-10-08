from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# FINAL MODEL COMPARISON
# Fusion Features -> ANOVA Top-30 -> StandardScaler -> RBF-SVM
# ============================================================

models = [
    "Basic CNN",
    "VGG16",
    "ResNet18",
    "DenseNet121",
    "RMSCNN",
    "GhostNet"
]

accuracy = [
    0.9409,
    0.9660,
    0.9872,
    0.9949,
    0.9968,
    0.9994
]

precision = [
    0.9589,
    0.9854,
    0.9873,
    0.9949,
    0.9936,
    1.0000
]

recall = [
    0.9223,
    0.9465,
    0.9872,
    0.9987,
    1.0000,
    0.9987
]

f1 = [
    0.9403,
    0.9656,
    0.9872,
    0.9949,
    0.9968,
    0.9994
]

roc_auc = [
    0.9781,
    0.9930,
    0.9968,
    1.0000,
    1.0000,
    1.0000
]


# ============================================================
# Output directory
# ============================================================

output_dir = Path("results/final_comparison")
output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================
# Print comparison table
# ============================================================

print("\n" + "=" * 75)
print("FINAL SIX-MODEL COMPARISON")
print("=" * 75)

print(
    f"{'Model':<15}"
    f"{'Accuracy':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
    f"{'ROC-AUC':>12}"
)

print("-" * 75)

for i, model in enumerate(models):
    print(
        f"{model:<15}"
        f"{accuracy[i]:>12.4f}"
        f"{precision[i]:>12.4f}"
        f"{recall[i]:>12.4f}"
        f"{f1[i]:>12.4f}"
        f"{roc_auc[i]:>12.4f}"
    )

print("=" * 75)


# ============================================================
# Find best models
# ============================================================

best_accuracy_index = max(range(len(accuracy)), key=lambda i: accuracy[i])
best_f1_index = max(range(len(f1)), key=lambda i: f1[i])
best_auc_index = max(range(len(roc_auc)), key=lambda i: roc_auc[i])

print("\nBest Test Accuracy:")
print(
    f"{models[best_accuracy_index]} "
    f"= {accuracy[best_accuracy_index]:.4f}"
)

print("\nBest Test F1-score:")
print(
    f"{models[best_f1_index]} "
    f"= {f1[best_f1_index]:.4f}"
)

print("\nBest ROC-AUC:")
print(
    f"{models[best_auc_index]} "
    f"= {roc_auc[best_auc_index]:.4f}"
)


# ============================================================
# Save text report
# ============================================================

report_path = output_dir / "final_model_comparison.txt"

with open(report_path, "w") as f:

    f.write("FINAL SIX-MODEL COMPARISON\n")
    f.write("=" * 75 + "\n\n")

    f.write(
        "Evaluation pipeline:\n"
        "Deep Features + Handcrafted Features\n"
        "-> ANOVA SelectKBest (Top 30)\n"
        "-> StandardScaler\n"
        "-> RBF-SVM\n\n"
    )

    f.write(
        f"{'Model':<15}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'ROC-AUC':>12}\n"
    )

    f.write("-" * 75 + "\n")

    for i, model in enumerate(models):
        f.write(
            f"{model:<15}"
            f"{accuracy[i]:>12.4f}"
            f"{precision[i]:>12.4f}"
            f"{recall[i]:>12.4f}"
            f"{f1[i]:>12.4f}"
            f"{roc_auc[i]:>12.4f}\n"
        )

    f.write("\n")
    f.write(
        f"Best Accuracy: {models[best_accuracy_index]} "
        f"({accuracy[best_accuracy_index]:.4f})\n"
    )

    f.write(
        f"Best F1-score: {models[best_f1_index]} "
        f"({f1[best_f1_index]:.4f})\n"
    )

    f.write(
        f"Best ROC-AUC: {models[best_auc_index]} "
        f"({roc_auc[best_auc_index]:.4f})\n"
    )

    f.write("\n")
    f.write(
        "Note: Results are based on the leakage-aware group-based "
        "train/validation/test split used in the project.\n"
    )


# ============================================================
# Accuracy comparison plot
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(models, accuracy)

plt.ylabel("Test Accuracy")
plt.xlabel("Model")
plt.title("Final Model Comparison - Test Accuracy")
plt.ylim(0.90, 1.01)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.002,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

accuracy_plot = output_dir / "final_model_accuracy_comparison.png"
plt.savefig(accuracy_plot, dpi=300)
plt.close()


# ============================================================
# F1-score comparison plot
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(models, f1)

plt.ylabel("Test F1-score")
plt.xlabel("Model")
plt.title("Final Model Comparison - F1-score")
plt.ylim(0.90, 1.01)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, f1):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.002,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

f1_plot = output_dir / "final_model_f1_comparison.png"
plt.savefig(f1_plot, dpi=300)
plt.close()


# ============================================================
# ROC-AUC comparison plot
# ============================================================

plt.figure(figsize=(10, 6))

bars = plt.bar(models, roc_auc)

plt.ylabel("ROC-AUC")
plt.xlabel("Model")
plt.title("Final Model Comparison - ROC-AUC")
plt.ylim(0.95, 1.01)
plt.grid(axis="y", alpha=0.3)

for bar, value in zip(bars, roc_auc):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.001,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

auc_plot = output_dir / "final_model_auc_comparison.png"
plt.savefig(auc_plot, dpi=300)
plt.close()


# ============================================================
# Complete
# ============================================================

print("\nFinal comparison completed successfully.")

print(f"Report: {report_path}")
print(f"Accuracy plot: {accuracy_plot}")
print(f"F1 plot: {f1_plot}")
print(f"ROC-AUC plot: {auc_plot}")

print("\nBest overall model:")
print("GhostNet + Fusion + ANOVA Top-30 + RBF-SVM")
print("Test Accuracy = 99.94%")
print("Test F1-score = 99.94%")
print("Test ROC-AUC = 1.0000")