# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path("/linux/antoimartin/v2/dataset").resolve()

IMG_DIR = BASE_DIR / "images"
LBL_DIR = BASE_DIR / "labels"

TRAIN_IMG_DIR = IMG_DIR / "train"
TRAIN_LBL_DIR = LBL_DIR / "train"

# Les dossiers crees lors du split precedent
INIT_IMG_DIR = IMG_DIR / "Train_Init"
POOL_IMG_DIR = IMG_DIR / "Unlabeled_Pool"
INIT_LBL_DIR = LBL_DIR / "Train_Init"
POOL_LBL_DIR = LBL_DIR / "Unlabeled_Pool"

def restore_dataset():
    print("--- Restauration du Dataset Complet ---")
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        (INIT_IMG_DIR, INIT_LBL_DIR, "Train_Init"),
        (POOL_IMG_DIR, POOL_LBL_DIR, "Unlabeled_Pool")
    ]

    total_restored = 0

    for img_src, lbl_src, name in sources:
        if not img_src.exists():
            print(f"[INFO] Le dossier {img_src} n'existe pas.")
            continue

        images = list(img_src.glob("*.jpg"))
        print(f"Restauration depuis {name} ({len(images)} images)...")

        for img in tqdm(images, desc=f"Restauration {name}"):
            # On DEPLACE les fichiers pour restaurer l'etat initial
            shutil.move(str(img), str(TRAIN_IMG_DIR / img.name))
            
            lbl_file = lbl_src / (img.stem + ".txt")
            if lbl_file.exists():
                shutil.move(str(lbl_file), str(TRAIN_LBL_DIR / lbl_file.name))
            total_restored += 1

    print(f"\n[SUCCESS] Restauration terminee. {total_restored} images/labels replaces dans 'train'.")

if __name__ == "__main__":
    restore_dataset()