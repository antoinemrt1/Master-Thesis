import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path("/linux/antoimartin/v2/dataset").resolve()
TRAIN_IMG_DIR = BASE_DIR / "images" / "train"
TRAIN_LBL_DIR = BASE_DIR / "labels" / "train"

# Nouveaux dossiers (On utilise 'Stratified' pour les differencier)
INIT_IMG_DIR = BASE_DIR / "images" / "Train_Init_Stratified"
POOL_IMG_DIR = BASE_DIR / "images" / "Unlabeled_Pool_Stratified"
INIT_LBL_DIR = BASE_DIR / "labels" / "Train_Init_Stratified"
POOL_LBL_DIR = BASE_DIR / "labels" / "Unlabeled_Pool_Stratified"

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

def analyze_dataset():
    """Analyse la distribution des classes dans le dossier d'entrainement."""
    print("--- 1. Analyse de la Distribution des Classes ---")
    class_counts = defaultdict(int)
    image_class_mapping = defaultdict(list) # Pour savoir quelles classes sont dans quelle image

    label_files = list(TRAIN_LBL_DIR.glob("*.txt"))
    if not label_files:
        print("[ERREUR] Aucun label trouve dans", TRAIN_LBL_DIR)
        return None, None

    for lbl_file in tqdm(label_files, desc="Analyse des labels"):
        img_name = lbl_file.stem + ".jpg"
        classes_in_img = set() # Classes uniques dans cette image
        
        with open(lbl_file, 'r') as f:
            for line in f:
                try:
                    cls_id = int(line.split()[0])
                    class_counts[cls_id] += 1
                    classes_in_img.add(cls_id)
                except ValueError:
                    pass
        
        if classes_in_img:
            image_class_mapping[img_name] = list(classes_in_img)

    # Plot
    classes = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    counts = [class_counts.get(i, 0) for i in range(len(CLASS_NAMES))]

    plt.figure(figsize=(16, 8))
    plt.bar(classes, counts, color='skyblue', edgecolor='black')
    plt.xticks(rotation=90)
    plt.yscale('log') # Echelle log car de fortes disparites
    plt.ylabel("Nombre d'instances (Log Scale)")
    plt.title("Distribution des classes dans le Dataset FAIR1M (Train Set)")
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300)
    print("[INFO] Graphique sauvegarde : class_distribution.png")

    return class_counts, image_class_mapping

def stratified_split(class_counts, image_class_mapping, init_ratio=0.20):
    """
    Cree un split stratifie en COPIANT les fichiers.
    Garantit (au mieux) la presence de chaque classe dans le set initial.
    """
    print(f"\n--- 2. Split Stratifie (Ratio: {init_ratio*100}%) ---")
    
    # Preparation des dossiers
    for d in [INIT_IMG_DIR, POOL_IMG_DIR, INIT_LBL_DIR, POOL_LBL_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Objectifs de nombre d'images par classe pour le set initial
    target_images_per_class = defaultdict(int)
    class_to_images = defaultdict(list)
    
    for img, classes in image_class_mapping.items():
        for c in classes:
            class_to_images[c].append(img)

    for c, imgs in class_to_images.items():
        target = max(1, int(len(imgs) * init_ratio)) 
        target_images_per_class[c] = target

    init_images = set()
    random.seed(42)

    # Strategie gloutonne
    sorted_classes = sorted(class_counts.keys(), key=lambda k: class_counts[k])

    for c in sorted_classes:
        imgs = class_to_images[c]
        random.shuffle(imgs)
        
        needed = target_images_per_class[c]
        current_count = sum(1 for img in imgs if img in init_images)
        
        while current_count < needed and imgs:
            img = imgs.pop()
            if img not in init_images:
                init_images.add(img)
                current_count += 1

    all_images = list(image_class_mapping.keys())
    total_images = len(all_images)
    target_total_init = int(total_images * init_ratio)
    
    remaining_images = list(set(all_images) - init_images)
    random.shuffle(remaining_images)
    
    while len(init_images) < target_total_init and remaining_images:
        init_images.add(remaining_images.pop())

    print(f"Copie des fichiers... (Init: {len(init_images)}, Pool: {total_images - len(init_images)})")
    
    for img_name in tqdm(all_images, desc="Copie des fichiers (Images & Labels)"):
        img_src = TRAIN_IMG_DIR / img_name
        lbl_src = TRAIN_LBL_DIR / (Path(img_name).stem + ".txt")
        
        if not img_src.exists(): continue

        if img_name in init_images:
            img_dst = INIT_IMG_DIR / img_name
            lbl_dst = INIT_LBL_DIR / lbl_src.name
        else:
            img_dst = POOL_IMG_DIR / img_name
            lbl_dst = POOL_LBL_DIR / lbl_src.name

        shutil.copy(str(img_src), str(img_dst))
        if lbl_src.exists():
            shutil.copy(str(lbl_src), str(lbl_dst))

    print(f"\n[SUCCESS] Split termine !")
    print(f"Dossier Initial : {INIT_IMG_DIR} ({len(list(INIT_IMG_DIR.glob('*.jpg')))} images)")
    print(f"Dossier Pool    : {POOL_IMG_DIR} ({len(list(POOL_IMG_DIR.glob('*.jpg')))} images)")

if __name__ == "__main__":
    class_counts, img_mapping = analyze_dataset()
    if class_counts:
        stratified_split(class_counts, img_mapping, init_ratio=0.20)