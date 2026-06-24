import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import traceback

# ==============================
# CONFIGURATION
# ==============================

DATA_INIT_YAML = Path("/linux/antoimartin/v2/data_init_stratified.yaml").resolve()

# Creation d'un chemin ABSOLU pour le dossier de sauvegarde
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_initial_model(model_type, imgsz, batch, device):
    """
    Entraine le modele AL de base (Baseline Initial) sur le subset Train_Init.
    L'entrainement s'arrete automatiquement grace a 'patience=20' 
    si le modele n'apprend plus sur le Val Set.
    """
    print(f"\n[TRAIN] Lancement de {model_type} a {imgsz}px sur les GPUs {device}")
    print(f"[DATA] Utilisation du fichier YAML : {DATA_INIT_YAML}")
    print(f"[SAVE] Sauvegarde prevue dans : {MODELS_DIR.absolute()}")

    model = YOLO(model_type)

    try:
        results = model.train(
            data=str(DATA_INIT_YAML),
            epochs=200,            # Laisse de la marge pour bien converger sur le petit subset
            imgsz=imgsz,           # Garde la resolution native (1024 pour HR)
            batch=batch,           # Attention au OOM en 1024px, meme avec 4 GPUs
            project=str(MODELS_DIR.absolute()),
            # Nom specifique pour ne pas confondre avec l'ancien "golden"
            name=f"baseline_stratified_20pct_{Path(model_type).stem}_{imgsz}", 
            exist_ok=True,
            patience=20,           # Early Stopping crucial pour ne pas overfitter le subset
            device=device,         # Ex: "0,1,2,3"
            workers=8,             # Optimise le chargement Dataloader
            amp=True               # Mixed Precision (Accelerateur)
        )

        best_pt = MODELS_DIR / f"baseline_stratified_20pct_{Path(model_type).stem}_{imgsz}" / "weights" / "best.pt"
        print(f"\n[SUCCES] Modele de depart (AL Baseline) sauvegarde ici : {best_pt}")

    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] L'entrainement a echoue : {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Script pour entrainer le modele initial (20%) avant Active Learning.")
    parser.add_argument("--mode", required=True, choices=["hr", "lr"], help="hr=Large/1024px, lr=Small/640px")
    # Valeur par defaut utilise vos 4 GPUs
    parser.add_argument("--device", default="0,1,2,3", help="ex: 0,1,2,3 pour utiliser les 4 GPUs")
    args = parser.parse_args()

    # Formatage propre de l'argument device pour DDP (Distributed Data Parallel de PyTorch)
    device_arg = args.device

    if args.mode == "hr":
        print("\n=== Entrainement Modele INITIAL HR (YOLOv8-Large sur 20%) ===")
        # Batch=8 sur 4 GPUs (soit 2 images par GPU) est souvent le maximum safe pour 1024px/Large sur 11GB VRAM
        # Si vous avez un OutOfMemory (OOM), baissez batch a 4.
        train_initial_model(model_type="yolov8l.pt", imgsz=1024, batch=8, device=device_arg)

    elif args.mode == "lr":
        print("\n=== Entrainement Modele INITIAL LR (YOLOv8-Small sur 20%) ===")
        # Modele petit (Small) + Image plus petite (640px) = on peut monter le batch size
        train_initial_model(model_type="yolov8s.pt", imgsz=640, batch=64, device=device_arg)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ATTENTION: Aucun GPU detecte. L'entrainement va echouer ou etre extremement lent.")
    else:
        print(f"Detecte {torch.cuda.device_count()} GPU(s).")
    main()