import os
import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
DATASET_DIR = Path("/linux/antoimartin/v2/Dataset")
VAL_IMG_DIR = DATASET_DIR / "Images" / "Val"
TEST_IMG_DIR = DATASET_DIR / "Images" / "Test"

# Creation du dossier Test
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

def split_val_to_test(split_ratio=0.5):
    print("Analyse du dossier Val...")

    val_images = list(VAL_IMG_DIR.glob("*.jpg"))  # ou .tif
    total_val = len(val_images)

    print(f"Images trouvees dans Val : {total_val}")

    if total_val == 0:
        print("Aucune image a deplacer.")
        return

    # Melange aleatoire avec une seed fixe pour la reproductibilite
    random.seed(42)
    random.shuffle(val_images)

    # Calcul du nombre d'images a deplacer
    num_to_move = int(total_val * split_ratio)
    test_images = val_images[:num_to_move]

    print(f"Deplacement de {num_to_move} images vers le dossier Test...")

    for img_path in test_images:
        dest_path = TEST_IMG_DIR / img_path.name
        shutil.move(str(img_path), str(dest_path))

    print("Termine ! Le dataset est maintenant divise en Train / Val / Test.")

if __name__ == "__main__":
    split_val_to_test()