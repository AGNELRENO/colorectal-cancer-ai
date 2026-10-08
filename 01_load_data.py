from pathlib import Path

DATA_DIR = Path("data/colon_image_sets")

crc_dir = DATA_DIR / "colon_aca"
non_crc_dir = DATA_DIR / "colon_n"

crc_images = list(crc_dir.glob("*.jpeg"))
non_crc_images = list(non_crc_dir.glob("*.jpeg"))

print("===== DATASET SUMMARY =====")
print(f"CRC images     : {len(crc_images)}")
print(f"Non-CRC images : {len(non_crc_images)}")
print(f"Total images   : {len(crc_images) + len(non_crc_images)}")