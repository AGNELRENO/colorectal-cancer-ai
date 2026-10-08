import subprocess
import sys
from pathlib import Path


# ============================================================
# MedGemma configuration
# ============================================================

MODEL_PATH = Path(
    r"E:\medgemma\medgemma-1.5-4b-it-q4_0.gguf"
)

MMPROJ_PATH = Path(
    r"E:\medgemma\mmproj-medgemma-1.5-4b-it-q4_0.gguf"
)

DEFAULT_IMAGE = Path(
    r"E:\colorectal-cancer-project\results\rmscnn_xai"
    r"\gradcam_results\colonca1207_gradcam.png"
)


# ============================================================
# Research prompt
# ============================================================

PROMPT = """
Analyze the provided RMSCNN Grad-CAM visualization for a research
project on colorectal cancer histopathology image classification.

The RMSCNN classification mapping is:
0 = CRC
1 = Non-CRC

For this image:
True label = CRC
Predicted label = CRC
Model confidence = 69.56%

Give ONLY these four sections:

1. VISUAL OBSERVATIONS
Describe only directly observable tissue, cellular, color,
texture, and architectural patterns.

2. GRAD-CAM INTERPRETATION
Describe where the heatmap is concentrated and what visual
patterns the model may be attending to.

3. RELATION TO MODEL PREDICTION
Explain how the highlighted visual patterns may have contributed
to the RMSCNN CRC prediction. Do not imply that these patterns
prove cancer.

4. LIMITATION
State that Grad-CAM represents model attention and does not
provide confirmed tumor boundaries, segmentation, or a clinical
diagnosis.

Do not claim that a highlighted region is definitely a tumor,
cancer, nucleus, gland, necrosis, or any other specific
pathological structure.

Do not provide a clinical diagnosis.

Do not describe reasoning or thinking.

Do not add additional sections.
"""


# ============================================================
# Check required files
# ============================================================

def check_files(image_path):

    if not MODEL_PATH.exists():
        print("ERROR: MedGemma model not found:")
        print(MODEL_PATH)
        sys.exit(1)

    if not MMPROJ_PATH.exists():
        print("ERROR: MedGemma mmproj not found:")
        print(MMPROJ_PATH)
        sys.exit(1)

    if not image_path.exists():
        print("ERROR: Grad-CAM image not found:")
        print(image_path)
        sys.exit(1)


# ============================================================
# Run MedGemma
# ============================================================

def analyze_image(image_path):

    check_files(image_path)

    command = [
        "llama-mtmd-cli",

        "-m",
        str(MODEL_PATH),

        "--mmproj",
        str(MMPROJ_PATH),

        "--image",
        str(image_path),

        "-ngl",
        "20",

        "-c",
        "4096",

        "-n",
        "350",

        "-p",
        PROMPT,
    ]

    print("=" * 70)
    print("MedGemma RMSCNN Grad-CAM Analysis")
    print("=" * 70)
    print(f"Image : {image_path}")
    print(f"Model : {MODEL_PATH}")
    print("=" * 70)

    print("\nRunning MedGemma...")
    print("Please wait...\n")

    try:

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )

    except subprocess.TimeoutExpired:

        print("\nERROR: MedGemma timed out.")
        sys.exit(1)

    except FileNotFoundError:

        print("\nERROR: llama-mtmd-cli was not found.")
        print("Make sure llama.cpp is installed and available in PATH.")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("MEDGEMMA OUTPUT")
    print("=" * 70)

    print(result.stdout)

    if result.stderr:

        print("\n" + "=" * 70)
        print("LLAMA.CPP LOG")
        print("=" * 70)

        print(result.stderr)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETED")
    print("=" * 70)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    image_path = DEFAULT_IMAGE

    # Optional custom image
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])

    analyze_image(image_path)