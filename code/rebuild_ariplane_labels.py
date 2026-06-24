import os
from pathlib import Path
from tqdm import tqdm
import shutil

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

ORIGINAL_CLASSES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

TARGET_CLASSES = [
    "Boeing747", "Boeing777", "Boeing787", 
    "A321", "A330", "A350"
]

# Mapping : Ancien ID (0-36) -> Nouvel ID (0-5)
ID_MAPPING = {ORIGINAL_CLASSES.index(c): TARGET_CLASSES.index(c) for c in TARGET_CLASSES}

def build_airplane_dataset(split_name, src_img_dir, src_lbl_dir):
    """
    Cree un dossier propre contenant uniquement les images ayant des avions,
    et les labels re-mappes de 0 a 5.
    """
    # Nouveaux dossiers
    dst_img_dir = BASE_DIR / "dataset_airplanes" / "images" / split_name
    dst_lbl_dir = BASE_DIR / "dataset_airplanes" / "labels" / split_name
    
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    txt_files = list(src_lbl_dir.glob("*.txt"))
    valid_images = 0

    print(f"\nTraitement du split : {split_name}...")
    for txt_path in tqdm(txt_files):
        new_lines = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: 
                    continue
                orig_id = int(parts[0])
                
                # Si c'est un avion cible, on re-map l'ID
                if orig_id in ID_MAPPING:
                    new_id = ID_MAPPING[orig_id]
                    # On s'assure que l'ID est bien entre 0 et 5
                    assert 0 <= new_id < 6, f"ERROR: New ID {new_id} invalid."
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
        
        # S'il y a au moins un avion sur l'image
        if new_lines:
            # 1. On sauvegarde le nouveau label
            new_txt_path = dst_lbl_dir / txt_path.name
            with open(new_txt_path, 'w') as f:
                f.write("\n".join(new_lines))
            
            # 2. On copie (ou lie) l'image correspondante
            img_path = src_img_dir / (txt_path.stem + ".jpg")
            if img_path.exists():
                dst_img_path = dst_img_dir / img_path.name
                if not dst_img_path.exists():
                    shutil.copy(str(img_path), str(dst_img_path))
            
            valid_images += 1

    print(f"-> {valid_images} airplane images prepared for {split_name}.")

def main():
    print(f"ID Mapping: {ID_MAPPING}")
    
    # On nettoie l'ancien dataset d'avions s'il existe
    ds_airplanes = BASE_DIR / "dataset_airplanes"
    if ds_airplanes.exists():
        shutil.rmtree(ds_airplanes)

    # 1. Train_Init (Les 20%)
    build_airplane_dataset(
        "train", 
        BASE_DIR / "dataset/images/Train_Init_Stratified", 
        BASE_DIR / "dataset/labels/Train_Init_Stratified"
    )
    
    # 2. Unlabeled_Pool (Les 80% pour l'AL plus tard)
    build_airplane_dataset(
        "unlabeled_pool", 
        BASE_DIR / "dataset/images/Unlabeled_Pool_Stratified", 
        BASE_DIR / "dataset/labels/Unlabeled_Pool_Stratified"
    )
    
    # 3. Validation
    build_airplane_dataset(
        "val", 
        BASE_DIR / "dataset/images/val", 
        BASE_DIR / "dataset/labels/val"
    )
    
    # 4. Test
    build_airplane_dataset(
        "test", 
        BASE_DIR / "dataset/images/test", 
        BASE_DIR / "dataset/labels/test"
    )

    # --- GENERATION DU YML PARFAIT ---
    yaml_path = BASE_DIR / "data_airplanes.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {BASE_DIR}/dataset_airplanes\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"test: images/test\n\n")
        f.write(f"nc: 6\n")
        f.write("names:\n")
        for i, cls_name in enumerate(TARGET_CLASSES):
            f.write(f"  {i}: {cls_name}\n")
            
    print(f"\n[SUCCESS] New dataset created in {ds_airplanes}")
    print(f"[SUCCESS] YAML updated: {yaml_path}")

if __name__ == "__main__":
    main()