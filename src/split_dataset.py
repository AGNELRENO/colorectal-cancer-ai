from pathlib import Path
import csv
import numpy as np

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data" / "colon_image_sets"

CSV_FILE = (
    PROJECT_DIR
    / "data"
    / "lc25000_image_groups.csv"
)

SEED = 42


# ============================================================
# 1. READ THE CSV
# ============================================================

samples = []

with open(
    CSV_FILE,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        # Only use colon images
        if row["tissue"] != "colon":
            continue

        filename = row["filename"]

        group_id = row["group_id"]


        # CRC
        if row["label"] == "colon_aca":

            label = 0

            image_path = (
                DATA_DIR
                / "colon_aca"
                / filename
            )


        # Non-CRC
        elif row["label"] == "colon_n":

            label = 1

            image_path = (
                DATA_DIR
                / "colon_n"
                / filename
            )


        else:
            continue


        # Check image exists
        if image_path.exists():

            samples.append(
                (
                    image_path,
                    label,
                    group_id
                )
            )


print("========================================")
print("DATASET")
print("========================================")

print(
    f"Total images found: {len(samples)}"
)


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def class_counts(indices):

    crc = 0

    non_crc = 0

    for i in indices:

        if samples[i][1] == 0:

            crc += 1

        else:

            non_crc += 1

    return crc, non_crc


# ============================================================
# 3. FIND GOOD 70/15/15 GROUP SPLIT
# ============================================================

best_split = None

best_score = float("inf")


all_indices = list(
    range(len(samples))
)


all_groups = [
    samples[i][2]
    for i in all_indices
]


# Try multiple random group splits
for seed in range(42, 142):

    # --------------------------------------------------------
    # First split:
    # Approximately 70% training
    # Approximately 30% temporary
    # --------------------------------------------------------

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=seed
    )


    train_indices, temp_indices = next(
        splitter_1.split(
            all_indices,
            groups=all_groups
        )
    )


    train_indices = list(
        train_indices
    )

    temp_indices = list(
        temp_indices
    )


    # --------------------------------------------------------
    # Second split:
    # Approximately 15% validation
    # Approximately 15% test
    # --------------------------------------------------------

    temp_groups = [
        all_groups[i]
        for i in temp_indices
    ]


    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=seed
    )


    val_relative, test_relative = next(
        splitter_2.split(
            temp_indices,
            groups=temp_groups
        )
    )


    val_indices = [
        temp_indices[i]
        for i in val_relative
    ]


    test_indices = [
        temp_indices[i]
        for i in test_relative
    ]


    # --------------------------------------------------------
    # Calculate class balance
    # --------------------------------------------------------

    train_crc, train_non_crc = (
        class_counts(train_indices)
    )

    val_crc, val_non_crc = (
        class_counts(val_indices)
    )

    test_crc, test_non_crc = (
        class_counts(test_indices)
    )


    # --------------------------------------------------------
    # Calculate CRC ratios
    # --------------------------------------------------------

    train_ratio = (
        train_crc / len(train_indices)
    )

    val_ratio = (
        val_crc / len(val_indices)
    )

    test_ratio = (
        test_crc / len(test_indices)
    )


    # --------------------------------------------------------
    # Balance score
    # --------------------------------------------------------

    score = (
        abs(train_ratio - 0.50)
        +
        abs(val_ratio - 0.50)
        +
        abs(test_ratio - 0.50)
    )


    # --------------------------------------------------------
    # Keep best split
    # --------------------------------------------------------

    if score < best_score:

        best_score = score

        best_split = (
            train_indices,
            val_indices,
            test_indices
        )


# ============================================================
# 4. USE BEST SPLIT
# ============================================================

train_indices, val_indices, test_indices = (
    best_split
)


# ============================================================
# 5. CREATE PYTORCH SAMPLE LISTS
# ============================================================

train_samples = [
    (
        samples[i][0],
        samples[i][1]
    )
    for i in train_indices
]


val_samples = [
    (
        samples[i][0],
        samples[i][1]
    )
    for i in val_indices
]


test_samples = [
    (
        samples[i][0],
        samples[i][1]
    )
    for i in test_indices
]


# ============================================================
# 6. COUNT CLASSES
# ============================================================

train_crc, train_non_crc = (
    class_counts(train_indices)
)

val_crc, val_non_crc = (
    class_counts(val_indices)
)

test_crc, test_non_crc = (
    class_counts(test_indices)
)


# ============================================================
# 7. PRINT SPLIT RESULTS
# ============================================================

print()
print("========================================")
print("GROUP-AWARE BALANCED SPLIT")
print("========================================")

print(
    f"Training   : {len(train_samples)}"
)

print(
    f"Validation : {len(val_samples)}"
)

print(
    f"Testing    : {len(test_samples)}"
)


print()
print("Class distribution:")


print(
    f"Train -> CRC: {train_crc}, "
    f"Non-CRC: {train_non_crc}"
)


print(
    f"Val   -> CRC: {val_crc}, "
    f"Non-CRC: {val_non_crc}"
)


print(
    f"Test  -> CRC: {test_crc}, "
    f"Non-CRC: {test_non_crc}"
)


# ============================================================
# 8. VERIFY GROUP LEAKAGE
# ============================================================

train_groups = set(
    samples[i][2]
    for i in train_indices
)


val_groups = set(
    samples[i][2]
    for i in val_indices
)


test_groups = set(
    samples[i][2]
    for i in test_indices
)


print()
print("========================================")
print("GROUP OVERLAP CHECK")
print("========================================")


train_val_overlap = (
    train_groups & val_groups
)

train_test_overlap = (
    train_groups & test_groups
)

val_test_overlap = (
    val_groups & test_groups
)


print(
    "Train ∩ Validation:",
    len(train_val_overlap)
)


print(
    "Train ∩ Test      :",
    len(train_test_overlap)
)


print(
    "Validation ∩ Test :",
    len(val_test_overlap)
)


# ============================================================
# 9. VERIFY NO LEAKAGE
# ============================================================

if (
    len(train_val_overlap) == 0
    and
    len(train_test_overlap) == 0
    and
    len(val_test_overlap) == 0
):

    print()
    print(
        "SUCCESS: No group leakage detected!"
    )

else:

    print()
    print(
        "WARNING: Group overlap detected!"
    )


# ============================================================
# 10. SAVE SPLIT FILES
# ============================================================

TRAIN_SPLIT_FILE = (
    PROJECT_DIR
    / "data"
    / "train_samples.npy"
)


VAL_SPLIT_FILE = (
    PROJECT_DIR
    / "data"
    / "val_samples.npy"
)


TEST_SPLIT_FILE = (
    PROJECT_DIR
    / "data"
    / "test_samples.npy"
)


np.save(
    TRAIN_SPLIT_FILE,
    np.array(
        train_samples,
        dtype=object
    )
)


np.save(
    VAL_SPLIT_FILE,
    np.array(
        val_samples,
        dtype=object
    )
)


np.save(
    TEST_SPLIT_FILE,
    np.array(
        test_samples,
        dtype=object
    )
)


# ============================================================
# 11. VERIFY SAVED FILES
# ============================================================

print()
print("========================================")
print("SPLIT FILES SAVED")
print("========================================")


print(
    "Train:",
    TRAIN_SPLIT_FILE
)


print(
    "Validation:",
    VAL_SPLIT_FILE
)


print(
    "Test:",
    TEST_SPLIT_FILE
)


print()
print(
    "Split files successfully created!"
)