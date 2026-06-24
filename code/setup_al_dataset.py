import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Chemin racine de votre dataset (qui contient 'images' et 'labels')
DATASET_ROOT = Path("/linux/antoimartin/v2/dataset")

# --- DOSSIERS IMAGES ---
IMG_TRAIN_DIR = DATASET_ROOT / "images" / "train"
IMG_INIT_DIR = DATASET_ROOT / "images" / "Train_Init"
IMG_POOL_DIR = DATASET_ROOT / "images" / "Unlabeled_Pool"

# --- DOSSIERS LABELS ---
LBL_TRAIN_DIR = DATASET_ROOT / "labels" / "train"
LBL_INIT_DIR = DATASET_ROOT / "labels" / "Train_Init"
LBL_POOL_DIR = DATASET_ROOT / "labels" / "Unlabeled_Pool"

def split_for_true_active_learning(init_ratio=0.20):
    print("Creation des dossiers d'Images et de Labels...")
    IMG_INIT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_POOL_DIR.mkdir(parents=True, exist_ok=True)
    LBL_INIT_DIR.mkdir(parents=True, exist_ok=True)
    LBL_POOL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. On liste toutes les images (jpg, png, tif...)
    # On utilise iterdir pour supporter plusieurs extensions si besoin
    images = [f for f in IMG_TRAIN_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.tif', '.jpeg']]
    total_images = len(images)
    print(f"Total images trouvees dans 'images/train' : {total_images}")
    
    if total_images == 0:
        print("[ERREUR] Aucune image trouvee. Verifiez vos chemins.")
        return

    # 2. Melange aleatoire (reproductible)
    random.seed(42)
    random.shuffle(images)
    
    # 3. Calcul de la separation (Split)
    num_init = int(total_images * init_ratio)
    init_images = images[:num_init]
    pool_images = images[num_init:]
    
    # --- FONCTION DE DEPLACEMENT HELPER ---
    def move_file_and_label(img_path, dest_img_dir, dest_lbl_dir):
        # Deplace l'image
        dest_img_path = dest_img_dir / img_path.name
        shutil.move(str(img_path), str(dest_img_path))
        
        # Trouve le label correspondant (remplace ex: .jpg par .txt)
        lbl_path = LBL_TRAIN_DIR / f"{img_path.stem}.txt"
        
        # S'il existe, on le deplace aussi
        if lbl_path.exists():
            dest_lbl_path = dest_lbl_dir / lbl_path.name
            shutil.move(str(lbl_path), str(dest_lbl_path))
        else:
            # Optionnel : Creer un fichier .txt vide si l'image est un "background" assume.
            # Normalement, votre script `prepare_labels.py` creait deja les .txt vides.
            pass

    # 4. Execution du deplacement (INIT)
    print(f"\nDeplacement de {num_init} paires (Image + Label) vers 'Train_Init' (Baseline)...")
    for img in tqdm(init_images):
        move_file_and_label(img, IMG_INIT_DIR, LBL_INIT_DIR)
        
    # 5. Execution du deplacement (POOL)
    print(f"\nDeplacement de {len(pool_images)} paires (Image + Label) vers 'Unlabeled_Pool' (Drone)...")
    for img in tqdm(pool_images):
        move_file_and_label(img, IMG_POOL_DIR, LBL_POOL_DIR)
        
    print("\nTermine ! L'arborescence est prete et propre pour YOLO.")
    
    # Verification finale
    print("\n--- Bilan de la restructuration ---")
    print(f"Images dans Train_Init     : {len(list(IMG_INIT_DIR.glob('*.jpg')))}")
    print(f"Labels dans Train_Init     : {len(list(LBL_INIT_DIR.glob('*.txt')))}")
    print(f"Images dans Unlabeled_Pool : {len(list(IMG_POOL_DIR.glob('*.jpg')))}")
    print(f"Labels dans Unlabeled_Pool : {len(list(LBL_POOL_DIR.glob('*.txt')))}")

if __name__ == "__main__":
    split_for_true_active_learning()