import cv2
from pathlib import Path

from handcrafted_features import extract_handcrafted_features


# --------------------------------------------------
# Select one image
# --------------------------------------------------

image_path = list(
    Path("data/colon_image_sets/colon_aca").glob("*.jpeg")
)[0]


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(
    str(image_path)
)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# Extract features
# --------------------------------------------------

features = extract_handcrafted_features(
    image
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("Image:", image_path)

print(
    "Image shape:",
    image.shape
)

print(
    "Feature vector shape:",
    features.shape
)

print(
    "Number of features:",
    len(features)
)

print(
    "Feature vector:"
)

print(features)

print()
print(
    "Handcrafted feature extraction successful!"
)