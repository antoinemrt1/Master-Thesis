import argparse
import torch
from pathlib import Path
from ultralytics import YOLO

# ==============================
# CONFIGURATION
# ==============================

DATA_YAML = Path("/linux/antoimartin/v2/data.yaml").resolve()

# Creation d'un chemin ABSOLU pour le dossier de sauvegarde
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_golden_model(model_type, imgsz, batch, device):
    """
    Entraine le modele. L'entrainement s'arrete automatiquement
    grace a 'patience=20' si le modele n'apprend plus.
    """
    print(f"\n[TRAIN] Lancement de {model_type} a {imgsz}px sur les GPUs {device}")
    print(f"Sauvegarde prevue dans : {MODELS_DIR.absolute()}")

    model = YOLO(model_type)

    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=200,
            imgsz=imgsz,
            batch=batch,
            # Ici on force le chemin absolu
            project=str(MODELS_DIR.absolute()),
            name=f"golden_{Path(model_type).stem}_{imgsz}",
            exist_ok=True,
            patience=20,
            device=device,
            workers=8,
            amp=True
        )

        best_pt = MODELS_DIR / f"golden_{Path(model_type).stem}_{imgsz}" / "weights" / "best.pt"
        print(f"\n[SUCCES] Meilleur modele sauvegarde ici : {best_pt}")

    except Exception as e:
        print(f"\n[ERREUR] L'entrainement a echoue : {e}")

        # Ajout du traceback pour mieux comprendre si ca plante encore
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["hr", "lr"])
    parser.add_argument("--device", required=True, help="ex: 0,1 ou 2,3")
    args = parser.parse_args()

    # Formatage de l'argument device (gestion du multi-GPU)
    device_arg = args.device
    if ',' in device_arg:
        # Ultralytics accepte soit une liste de int, soit un string "0,1"
        # La documentation preconise le string pour plusieurs GPUs
        device_arg = args.device
    else:
        # Si un seul GPU (ex: "0"), on le garde en string
        device_arg = args.device

    if args.mode == "hr":
        print("\n=== Entrainement Modele HR (YOLOv8-Large) ===")
        # Batch petit pour eviter OOM sur 1024px avec le modele Large
        train_golden_model(model_type="yolov8l.pt", imgsz=1024, batch=4, device=device_arg)

    elif args.mode == "lr":
        print("\n=== Entrainement Modele LR (YOLOv8-Small) ===")
        # Batch plus grand possible car image petite (640px)
        train_golden_model(model_type="yolov8s.pt", imgsz=640, batch=16, device=device_arg)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ATTENTION: Aucun GPU detecte.")
    main()