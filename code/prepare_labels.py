import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path("/linux/antoimartin/v2/Dataset")
PARQUET_FILE = DATASET_DIR / "labels.parquet"

# Dossiers d'images existants (apres execution de split_dataset.py)
TRAIN_IMG_DIR = DATASET_DIR / "Images" / "Train"
VAL_IMG_DIR = DATASET_DIR / "Images" / "Val"
TEST_IMG_DIR = DATASET_DIR / "Images" / "Test"

# Dossiers de labels a creer
TRAIN_LBL_DIR = DATASET_DIR / "labels" / "Train"
VAL_LBL_DIR = DATASET_DIR / "labels" / "Val"
TEST_LBL_DIR = DATASET_DIR / "labels" / "Test"

TRAIN_LBL_DIR.mkdir(parents=True, exist_ok=True)
VAL_LBL_DIR.mkdir(parents=True, exist_ok=True)
TEST_LBL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

def main():
    print(f"Chargement du fichier {PARQUET_FILE}...")
    df = pd.read_parquet(PARQUET_FILE)

    df['filename'] = df['FilePath'].apply(lambda x: Path(x).name)
    grouped = df.groupby('filename')

    train_images = set([f.name for f in TRAIN_IMG_DIR.iterdir() if f.is_file()])
    val_images = set([f.name for f in VAL_IMG_DIR.iterdir() if f.is_file()])
    test_images = set([f.name for f in TEST_IMG_DIR.iterdir() if f.is_file()])

    print(f"Images sur disque : Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

    processed = {"Train": 0, "Val": 0, "Test": 0}

    for img_name, group in tqdm(grouped, desc="Generation YOLO .txt"):
        if img_name in train_images:
            out_dir = TRAIN_LBL_DIR
            processed["Train"] += 1
        elif img_name in val_images:
            out_dir = VAL_LBL_DIR
            processed["Val"] += 1
        elif img_name in test_images:
            out_dir = TEST_LBL_DIR
            processed["Test"] += 1
        else:
            continue

        txt_path = out_dir / f"{Path(img_name).stem}.txt"

        lines = []
        for _, row in group.iterrows():
            cls_name = row['Category']
            if cls_name not in CLASS_NAMES:
                continue

            cls_id = CLASS_NAMES.index(cls_name)

            try:
                img_w = float(row['ImageWidth'])
                img_h = float(row['ImageHeight'])
            except:
                img_w, img_h = 1000.0, 1000.0

            if img_w <= 0:
                img_w = 1000.0
            if img_h <= 0:
                img_h = 1000.0

            xc = ((float(row['x_min']) + float(row['x_max'])) / 2.0) / img_w
            yc = ((float(row['y_min']) + float(row['y_max'])) / 2.0) / img_h
            w = (float(row['x_max']) - float(row['x_min'])) / img_w
            h = (float(row['y_max']) - float(row['y_min'])) / img_h

            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.0, min(1.0, w))
            h = max(0.0, min(1.0, h))

            if w <= 0 or h <= 0:
                continue

            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(txt_path, "w") as f:
            if lines:
                f.write("\n".join(lines))

    print("\n--- Bilan ---")
    for k, v in processed.items():
        print(f"{k} : {v} labels crees.")

if __name__ == "__main__":
    main()