import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()
SRC_DATASET = BASE_DIR / "dataset"  # Le dataset source contenant les dossiers _Stratified

# Les nouveaux datasets
SPLIT4_DATASET = BASE_DIR / "dataset_split4"
SPLIT16_DATASET = BASE_DIR / "dataset_split16"

# Les dossiers a traiter pour garder la structure exacte
SUBFOLDERS = [
    "Train_Init_Stratified",
    "Unlabeled_Pool_Stratified",
    "Val",
    "Test"  # ou "val" / "test" selon la casse exacte de vos dossiers
]

# Taille cible apres interpolation cubique (YOLO aime les grands multiples de 32)
TARGET_SIZE = 1024

def process_image(img_path, lbl_path, out_img_dir, out_lbl_dir, grid_size):
    """
    Decoupe l'image en une grille (grid_size x grid_size),
    redimensionne avec interpolation cubique,
    et adapte les bounding boxes.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return
    
    H, W = img.shape[:2]
    step_h = H // grid_size
    step_w = W // grid_size

    # Charger les boites existantes
    boxes = []
    if lbl_path.exists():
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xc, yc, w, h = map(float, parts)
                    # Convertir en coordonnees absolues [x1, y1, x2, y2]
                    x1 = (xc - w/2) * W
                    y1 = (yc - h/2) * H
                    x2 = (xc + w/2) * W
                    y2 = (yc + h/2) * H
                    boxes.append([int(c), x1, y1, x2, y2])

    # Decoupage de la grille
    for i in range(grid_size):
        for j in range(grid_size):
            # Limites du crop
            crop_y1 = i * step_h
            crop_y2 = (i + 1) * step_h if i < grid_size - 1 else H
            crop_x1 = j * step_w
            crop_x2 = (j + 1) * step_w if j < grid_size - 1 else W
            
            crop_H = crop_y2 - crop_y1
            crop_W = crop_x2 - crop_x1

            # 1. Extraction et interpolation cubique
            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            res_img = cv2.resize(crop_img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_CUBIC)
            
            new_img_name = f"{img_path.stem}_{i}_{j}.jpg"
            new_lbl_name = f"{img_path.stem}_{i}_{j}.txt"
            
            # 2. Traitement des boites
            new_boxes = []
            for box in boxes:
                c, x1, y1, x2, y2 = box
                
                # Intersection entre la boite et le crop
                inter_x1 = max(x1, crop_x1)
                inter_y1 = max(y1, crop_y1)
                inter_x2 = min(x2, crop_x2)
                inter_y2 = min(y2, crop_y2)
                
                # Si la boite est dans le crop
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    # Transformer en coordonnees relatives au crop
                    rel_x1 = inter_x1 - crop_x1
                    rel_y1 = inter_y1 - crop_y1
                    rel_x2 = inter_x2 - crop_x1
                    rel_y2 = inter_y2 - crop_y1
                    
                    # Convertir au format YOLO (0 a 1)
                    new_xc = ((rel_x1 + rel_x2) / 2.0) / crop_W
                    new_yc = ((rel_y1 + rel_y2) / 2.0) / crop_H
                    new_w = (rel_x2 - rel_x1) / crop_W
                    new_h = (rel_y2 - rel_y1) / crop_H
                    
                    # Securite
                    new_xc = max(0.0, min(1.0, new_xc))
                    new_yc = max(0.0, min(1.0, new_yc))
                    new_w = max(0.0, min(1.0, new_w))
                    new_h = max(0.0, min(1.0, new_h))
                    
                    if new_w > 0.001 and new_h > 0.001:  # Ignorer les micro-debris
                        new_boxes.append(f"{int(c)} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}")

            # 3. Sauvegarde
            cv2.imwrite(str(out_img_dir / new_img_name), res_img)
            
            with open(out_lbl_dir / new_lbl_name, 'w') as f:
                if new_boxes:
                    f.write("\n".join(new_boxes))

def generate_dataset(dst_dataset, grid_size):
    print(f"\n{'='*50}")
    print(f"GENERATION DU DATASET SPLIT {grid_size}x{grid_size}")
    print(f"{'='*50}")
    
    if dst_dataset.exists():
        shutil.rmtree(dst_dataset)
    
    for subfolder in SUBFOLDERS:
        src_img = SRC_DATASET / "images" / subfolder
        src_lbl = SRC_DATASET / "labels" / subfolder
        
        # Verifie la casse (majuscule/minuscule)
        if not src_img.exists():
            src_img = SRC_DATASET / "images" / subfolder.lower()
        if not src_lbl.exists():
            src_lbl = SRC_DATASET / "labels" / subfolder.lower()
            
        if not src_img.exists():
            print(f"Dossier ignore (introuvable) : {src_img}")
            continue
            
        dst_img = dst_dataset / "images" / subfolder
        dst_lbl = dst_dataset / "labels" / subfolder
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        
        images = list(src_img.glob("*.jpg"))
        print(f"Traitement de {subfolder} ({len(images)} images)...")
        
        for img_path in tqdm(images):
            lbl_path = src_lbl / (img_path.stem + ".txt")
            process_image(img_path, lbl_path, dst_img, dst_lbl, grid_size)

if __name__ == "__main__":
    # 1. Dataset Split 4 (2x2 grid -> 4 images)
    generate_dataset(SPLIT4_DATASET, grid_size=2)
    
    # 2. Dataset Split 16 (4x4 grid -> 16 images)
    generate_dataset(SPLIT16_DATASET, grid_size=4)
    
    print("\n[SUCCES] Les nouveaux datasets ont ete generes avec interpolation cubique !")