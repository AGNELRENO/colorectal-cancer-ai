import cv2
import numpy as np

from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ==================================================
# COLOR FEATURES
# ==================================================

def extract_color_features(image):
    """
    Extract color statistics from:
    RGB, HSV, Lab, and YCbCr.
    """

    features = []

    # --------------------------------------------------
    # RGB
    # --------------------------------------------------

    rgb = image

    for channel in cv2.split(rgb):

        features.append(
            np.mean(channel)
        )

        features.append(
            np.std(channel)
        )

    # --------------------------------------------------
    # HSV
    # --------------------------------------------------

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV
    )

    for channel in cv2.split(hsv):

        features.append(
            np.mean(channel)
        )

        features.append(
            np.std(channel)
        )

    # --------------------------------------------------
    # Lab
    # --------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    for channel in cv2.split(lab):

        features.append(
            np.mean(channel)
        )

        features.append(
            np.std(channel)
        )

    # --------------------------------------------------
    # YCbCr
    # --------------------------------------------------

    ycbcr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2YCrCb
    )

    for channel in cv2.split(ycbcr):

        features.append(
            np.mean(channel)
        )

        features.append(
            np.std(channel)
        )

    return features


# ==================================================
# GLCM FEATURES
# ==================================================

def extract_glcm_features(image):
    """
    Extract GLCM texture features.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce gray levels
    gray = gray // 32

    gray = gray.astype(
        np.uint8
    )

    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0],
        levels=8,
        symmetric=True,
        normed=True
    )

    features = []

    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM"
    ]

    for prop in properties:

        value = graycoprops(
            glcm,
            prop
        )[0, 0]

        features.append(
            value
        )

    return features


# ==================================================
# LBP FEATURES
# ==================================================

def extract_lbp_features(image):
    """
    Extract Local Binary Pattern histogram.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    radius = 1

    points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        points,
        radius,
        method="uniform"
    )

    n_bins = points + 2

    histogram, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(
            0,
            n_bins + 1
        ),
        range=(
            0,
            n_bins
        )
    )

    histogram = histogram.astype(
        np.float32
    )

    histogram /= (
        histogram.sum() + 1e-7
    )

    return histogram.tolist()


# ==================================================
# SHAPE FEATURES
# ==================================================

def extract_shape_features(image):
    """
    Extract basic shape-related features.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    features = []

    if len(contours) == 0:

        return [
            0.0,
            0.0,
            0.0,
            0.0
        ]

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(
        largest_contour
    )

    perimeter = cv2.arcLength(
        largest_contour,
        True
    )

    if perimeter > 0:

        circularity = (
            4 *
            np.pi *
            area /
            (perimeter ** 2)
        )

    else:

        circularity = 0.0


    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    aspect_ratio = (
        w / h
        if h > 0
        else 0.0
    )


    features.extend([
        area,
        perimeter,
        circularity,
        aspect_ratio
    ])

    return features


# ==================================================
# COMPLETE HANDCRAFTED FEATURE EXTRACTION
# ==================================================

def extract_handcrafted_features(image):
    """
    Extract all handcrafted features.
    """

    features = []

    # Color
    features.extend(
        extract_color_features(image)
    )

    # GLCM
    features.extend(
        extract_glcm_features(image)
    )

    # LBP
    features.extend(
        extract_lbp_features(image)
    )

    # Shape
    features.extend(
        extract_shape_features(image)
    )

    return np.array(
        features,
        dtype=np.float32
    )