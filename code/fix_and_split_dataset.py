# fix_and_split_dataset.py
import os
import random
import shutil
from pathlib import Path

# ==========================================
# CONFIGURATION DES CHEMINS
# ==========================================
# Point de depart : Le dossier principal Dataset
DATASET_DIR = Path("/linux/antoimartin/v2/dataset").resolve()

# Dossiers d'images
IMG_DIR = DATASET_DIR / "images"
TRAIN_IMG_DIR = IMG_DIR / "train"
INIT_IMG_DIR = IMG_DIR / "Train_Init"
POOL_IMG_DIR = IMG_DIR / "Unlabeled_Pool"

# Dossiers de labels
LBL_DIR = DATASET_DIR / "labels"
TRAIN_LBL_DIR = LBL_DIR / "train"
INIT_LBL_DIR = LBL_DIR / "Train_Init"
POOL_LBL_DIR = LBL_DIR / "Unlabeled_Pool"

def restore_train_folder():
    """
    Etape 1 : Reparation.
    On ramene toutes les images qui etaient dans Train_Init et Unlabeled_Pool
    vers le dossier train original pour repartir sur une base saine.
    """
    print("\n--- Etape 1 : Reparation (Restauration du dossier train) ---")
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    count_restored = 0
    
    # Ramener depuis Train_Init
    if INIT_IMG_DIR.exists():
        for img in INIT_IMG_DIR.glob("*.jpg"):
            shutil.move(str(img), str(TRAIN_IMG_DIR / img.name))
            count_restored += 1
            
    # Ramener depuis Unlabeled_Pool
    if POOL_IMG_DIR.exists():
        for img in POOL_IMG_DIR.glob("*.jpg"):
            shutil.move(str(img), str(TRAIN_IMG_DIR / img.name))
            count_restored += 1
            
    print(f"Restauration terminee : {count_restored} images replacees dans {TRAIN_IMG_DIR.name}.")

def perform_clean_split(init_ratio=0.20):
    """
    Etape 2 : Le vrai split.
    On deplace (Images + Labels) vers Train_Init (20%) et Unlabeled_Pool (80%).
    """
    print("\n--- Etape 2 : Split propre (Images & Labels) ---")
    
    # Creation des dossiers de destination pour les Images
    INIT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    POOL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Creation des dossiers de destination pour les Labels
    INIT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    POOL_LBL_DIR.mkdir(parents=True, exist_ok=True)
    
    # On liste toutes les images du dossier d'origine
    all_images = list(TRAIN_IMG_DIR.glob("*.jpg"))
    total_images = len(all_images)
    print(f"Total images dans Train : {total_images}")
    
    if total_images == 0:
        print("[ERREUR CRITIQUE] Aucune image trouvee. La restauration a echoue ou le dossier est vide.")
        return

    # Melange aleatoire avec une graine fixe (pour reproductibilite)
    random.seed(42)
    random.shuffle(all_images)
    
    # Calcul du nombre d'images pour le set "Initial"
    num_init = int(total_images * init_ratio)
    
    # 1. Deplacement vers Train_Init (Les 20%)
    print(f"\nDeplacement de {num_init} fichiers (Images + Labels) vers Train_Init...")
    for img_path in all_images[:num_init]:
        # Deplacer l'image
        shutil.move(str(img_path), str(INIT_IMG_DIR / img_path.name))
        
        # Trouver et deplacer le label correspondant
        # Ex: t_14154.jpg -> t_14154.txt
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(INIT_LBL_DIR / lbl_path.name))

    # 2. Deplacement vers Unlabeled_Pool (Les 80%)
    print(f"Deplacement de {total_images - num_init} fichiers (Images + Labels) vers Unlabeled_Pool...")
    for img_path in all_images[num_init:]:
        # Deplacer l'image
        shutil.move(str(img_path), str(POOL_IMG_DIR / img_path.name))
        
        # Trouver et deplacer le label correspondant
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.move(str(lbl_path), str(POOL_LBL_DIR / lbl_path.name))

    print("\n--- Bilan ---")
    print("Le dataset est maintenant proprement separe.")
    print(f"-> Baseline Initiale (20%)   : Images dans {INIT_IMG_DIR.name} | Labels dans {INIT_LBL_DIR.name}")
    print(f"-> Pool Active Learning (80%) : Images dans {POOL_IMG_DIR.name} | Labels dans {POOL_LBL_DIR.name}")

if __name__ == "__main__":
    # Etape 1 : On remet tout au meme endroit
    restore_train_folder()
    
    # Etape 2 : On fait le split correctement (images ET labels)
    perform_clean_split()