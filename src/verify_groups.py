from pathlib import Path
import csv

PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = PROJECT_DIR / "data" / "lc25000_image_groups.csv"
DATA_DIR = PROJECT_DIR / "data" / "colon_image_sets"

print("===== GROUP CSV VERIFICATION =====")

# Check CSV exists
if not CSV_FILE.exists():
    print("ERROR: CSV file not found!")
    print(CSV_FILE)
    exit()

print("CSV found:", CSV_FILE)

# Read CSV
with open(CSV_FILE, "r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

print("CSV rows:", len(rows))

# Show column names
print("Columns:", list(rows[0].keys()))

# Keep only colon images
colon_rows = [
    row for row in rows
    if row["tissue"] == "colon"
]

print("Colon rows:", len(colon_rows))

# Count actual images
crc_images = list((DATA_DIR / "colon_aca").glob("*.jpeg"))
non_crc_images = list((DATA_DIR / "colon_n").glob("*.jpeg"))

print("Actual CRC images:", len(crc_images))
print("Actual Non-CRC images:", len(non_crc_images))
print("Actual total images:", len(crc_images) + len(non_crc_images))

# Count groups
group_ids = set(row["group_id"] for row in colon_rows)

print("Unique colon groups:", len(group_ids))

print()
print("===== VERIFICATION COMPLETE =====")