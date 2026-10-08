import sys
from pathlib import Path

import torch
from torchinfo import summary

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
OUTPUT_DIR = PROJECT_ROOT / "results" / "model_summaries"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("MODEL SUMMARY GENERATION")
print("=" * 70)
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("=" * 70)


# ------------------------------------------------------------
# Helper function
# ------------------------------------------------------------

def generate_summary(model, model_name):
    """
    Generate and save a torchinfo summary.
    """

    print("\n" + "=" * 70)
    print(f"{model_name} MODEL SUMMARY")
    print("=" * 70)

    model = model.to(DEVICE)
    model.eval()

    result = summary(
        model,
        input_size=(1, 3, 224, 224),
        device=DEVICE,
        col_names=(
            "input_size",
            "output_size",
            "num_params",
            "trainable",
        ),
        depth=10,
        verbose=1,
    )

    output_file = OUTPUT_DIR / f"{model_name.lower().replace(' ', '_')}_summary.txt"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"{model_name} MODEL SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        f.write(str(result))
        f.write("\n")

    print(f"\nSaved: {output_file}")


# ------------------------------------------------------------
# 1. Basic CNN
# ------------------------------------------------------------

from model_basic_cnn import get_model as get_basic_cnn

basic_cnn = get_basic_cnn(
    num_classes=2,
    dropout=0.3
)

generate_summary(
    basic_cnn,
    "Basic CNN"
)


# ------------------------------------------------------------
# 2. ResNet18
# ------------------------------------------------------------

from model import get_model as get_resnet18

resnet18 = get_resnet18(num_classes=2)

generate_summary(
    resnet18,
    "ResNet18"
)


# ------------------------------------------------------------
# 3. DenseNet121
# ------------------------------------------------------------

from model_densenet121 import get_model as get_densenet121

densenet121 = get_densenet121(num_classes=2)

generate_summary(
    densenet121,
    "DenseNet121"
)


# ------------------------------------------------------------
# 4. VGG16
# ------------------------------------------------------------

from model_vgg16 import get_model as get_vgg16

vgg16 = get_vgg16(num_classes=2)

generate_summary(
    vgg16,
    "VGG16"
)


# ------------------------------------------------------------
# 5. GhostNet
# ------------------------------------------------------------

from model_ghostnet import get_model as get_ghostnet

ghostnet = get_ghostnet(num_classes=2)

generate_summary(
    ghostnet,
    "GhostNet"
)


# ------------------------------------------------------------
# 6. RMSCNN
# ------------------------------------------------------------

from model_rmscnn import get_model as get_rmscnn

rmscnn = get_rmscnn(num_classes=2)

generate_summary(
    rmscnn,
    "RMSCNN"
)


# ------------------------------------------------------------
# Complete
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ALL MODEL SUMMARIES GENERATED SUCCESSFULLY")
print("=" * 70)

print(f"\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(OUTPUT_DIR.glob("*_summary.txt")):
    print(f"  - {file.name}")