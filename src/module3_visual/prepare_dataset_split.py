import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "data/annotations/module3/labels.csv"
OUTPUT_DIR = "data/datasets/module3"

LABEL_FOLDER_MAP = {
    "Critical": "critical",
    "Important": "important",
    "Skip": "skip"
}

# Remove old dataset split if it already exists
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Create train/val/test folders
for split in ["train", "val", "test"]:
    for folder in LABEL_FOLDER_MAP.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, folder), exist_ok=True)

# Read labels CSV
df = pd.read_csv(CSV_PATH)

print("Total images:", len(df))
print("\nOriginal label distribution:")
print(df["label"].value_counts())

print("\nLecture distribution:")
print(df["lecture_id"].value_counts())

# Prototype split:
# 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


def copy_images(split_name, split_df):
    for _, row in split_df.iterrows():
        src_path = row["image_path"]
        label = row["label"]
        label_folder = LABEL_FOLDER_MAP[label]

        filename = os.path.basename(src_path)
        dst_path = os.path.join(OUTPUT_DIR, split_name, label_folder, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Missing file: {src_path}")


copy_images("train", train_df)
copy_images("val", val_df)
copy_images("test", test_df)

print("\nDataset split completed.")

for split_name, split_df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df) 
]:
    print(f"\n{split_name.upper()} set: {len(split_df)} images")
    print(split_df["label"].value_counts())