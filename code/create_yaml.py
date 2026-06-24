import os
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

def create_yaml(dataset_name, yaml_filename):
    yaml_path = BASE_DIR / yaml_filename
    dataset_dir = BASE_DIR / dataset_name
    
    with open(yaml_path, "w") as f:
        f.write(f"path: {dataset_dir}\n")
        f.write("train: images/Train_Init_Stratified\n") # On utilise la Baseline 20%
        f.write("val: images/Val\n")
        f.write("test: images/Test\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write("names:\n")
        for i, cls_name in enumerate(CLASS_NAMES):
            f.write(f"  {i}: {cls_name}\n")
            
    print(f"[SUCCES] Cree : {yaml_path}")

if __name__ == "__main__":
    create_yaml("dataset_split4", "data_split4.yaml")
    create_yaml("dataset_split16", "data_split16.yaml")