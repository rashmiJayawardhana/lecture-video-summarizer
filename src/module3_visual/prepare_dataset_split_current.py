import os
import shutil
import pandas as pd

CSV_PATH = "data/annotations/module3/labels.csv"
OUTPUT_DIR = "data/datasets/module3"

LABEL_FOLDER_MAP = {
    "Critical": "critical",
    "Important": "important",
    "Skip": "skip"
}

TRAIN_LECTURES = [
    "lecture_001",
    "lecture_002",
    "lecture_005",
    "lecture_006",
    "lecture_010",
    "lecture_012",
    "lecture_013"
]

VAL_LECTURES = [
    "lecture_003",
    "lecture_004"
]

TEST_LECTURES = [
    "lecture_009",
    "lecture_014"
]

IGNORE_LECTURES = [
    "lecture_007",
    "lecture_008"
]


def create_folders():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ["train", "val", "test"]:
        for folder in LABEL_FOLDER_MAP.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, folder), exist_ok=True)


def copy_images(split_name, split_df):
    for _, row in split_df.iterrows():
        src_path = row["image_path"]
        label_folder = LABEL_FOLDER_MAP[row["label"]]

        filename = os.path.basename(src_path)
        dst_path = os.path.join(OUTPUT_DIR, split_name, label_folder, filename)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Missing file: {src_path}")


def print_report(split_name, split_df):
    print(f"\n{split_name.upper()} SET")
    print("Total:", len(split_df))

    print("\nLabel distribution:")
    print(split_df["label"].value_counts())

    print("\nLecture distribution:")
    print(split_df["lecture_id"].value_counts().sort_index())


df = pd.read_csv(CSV_PATH)

print("Total labelled images:", len(df))

print("\nOverall label distribution:")
print(df["label"].value_counts())

print("\nOverall lecture distribution:")
print(df["lecture_id"].value_counts().sort_index())

print("\nLabel distribution per lecture:")
print(pd.crosstab(df["lecture_id"], df["label"]))

create_folders()

train_df = df[df["lecture_id"].isin(TRAIN_LECTURES)]
val_df = df[df["lecture_id"].isin(VAL_LECTURES)]
test_df = df[df["lecture_id"].isin(TEST_LECTURES)]
ignored_df = df[df["lecture_id"].isin(IGNORE_LECTURES)]

copy_images("train", train_df)
copy_images("val", val_df)
copy_images("test", test_df)

print("\nDataset split completed.")

print_report("train", train_df)
print_report("val", val_df)
print_report("test", test_df)

print("\nIGNORED LECTURES")
print("Total ignored:", len(ignored_df))
print(ignored_df["lecture_id"].value_counts().sort_index())

total_copied = 0
for root, dirs, files in os.walk(OUTPUT_DIR):
    total_copied += len([
        f for f in files
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

print("\nTotal copied images:", total_copied)