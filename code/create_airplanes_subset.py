import os
from pathlib import Path
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

# Ancienne liste complete
ORIGINAL_CLASSES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

# Nouvelle selection (equilibree)
TARGET_CLASSES = [
    "Boeing747", "Boeing777", "Boeing787", 
    "A321", "A330", "A350"
]

# Mapping des anciens ID vers les nouveaux ID (0 a 5)
ID_MAPPING = {ORIGINAL_CLASSES.index(c): TARGET_CLASSES.index(c) for c in TARGET_CLASSES}

def filter_labels(src_dir, dst_dir):
    """Read old labels, keep target classes, and remap IDs."""
    if not src_dir.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    txt_files = list(src_dir.glob("*.txt"))
    
    valid_images = 0
    for txt_path in tqdm(txt_files, desc=f"Filtering {src_dir.name}"):
        new_lines = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                orig_id = int(parts[0])
                
                # Keep only selected classes
                if orig_id in ID_MAPPING:
                    new_id = ID_MAPPING[orig_id]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
        
        # Save only if at least one valid object remains
        if new_lines:
            with open(dst_dir / txt_path.name, 'w') as f:
                f.write("\n".join(new_lines))
            valid_images += 1
            
    print(f"-> {valid_images} images kept in {dst_dir.name} (with target airplanes).")

def main():
    print(f"Target classes : {TARGET_CLASSES}")
    print(f"ID mapping     : {ID_MAPPING}\n")

    # Filter all stratified folders
    folders_to_filter = [
        ("labels/Train_Init_Stratified", "labels_airplanes/Train_Init_Stratified"),
        ("labels/Unlabeled_Pool_Stratified", "labels_airplanes/Unlabeled_Pool_Stratified"),
        ("labels/Val", "labels_airplanes/Val"),
        ("labels/Test", "labels_airplanes/Test")
    ]

    for src, dst in folders_to_filter:
        filter_labels(BASE_DIR / "dataset" / src, BASE_DIR / "dataset" / dst)

    # Create new YAML
    yaml_path = BASE_DIR / "data_airplanes.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {BASE_DIR}/dataset\n")
        f.write(f"train: images/Train_Init_Stratified\n")
        f.write(f"val: images/Val\n")
        f.write(f"test: images/Test\n")
        f.write(f"nc: {len(TARGET_CLASSES)}\n")
        f.write(f"names: {TARGET_CLASSES}\n")
        
    print(f"\nNew dataset ready! YAML created: {yaml_path}")

if __name__ == "__main__":
    main()