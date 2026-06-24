import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
DATA_YAML = Path("/linux/antoimartin/v2/data.yaml").resolve()

# Choisir modele a evaluer
MODEL_PATH = Path("/linux/antoimartin/v2/code/trained_models/golden_yolov8s_640/weights/best.pt")
#MODEL_PATH = Path("/linux/antoimartin/v2/code/trained_models/golden_yolov8l_1024/weights/best.pt")

RESULTS_DIR = Path("evaluation_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_on_test_set(model_path, imgsz, device):
    """
    Evalue un modele sur le Test Set (donnees jamais vues par le modele).
    """
    print(f"\n[EVALUATION] Lancement de l'evaluation sur le TEST SET")
    print(f"Modele : {model_path}")
    print(f"Dataset : {DATA_YAML}")

    # 1. Chargement du modele
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le modele : {e}")
        return

    # 2. Validation forcee sur le split "test"
    try:
        # Le parametre split="test" force Ultralytics a utiliser le dossier 'test' du .yaml
        metrics = model.val(
            data=str(DATA_YAML),
            split="test",       # On evalue sur le Test Set
            imgsz=imgsz,
            device=device,
            batch=16,           # Batch plus grand = evaluation plus rapide
            plots=True,         # Genere des matrices de confusion, PR curves, etc.
            project=str(RESULTS_DIR.resolve()),
            name="eval_test_set",
            exist_ok=True
        )

        # 3. Recuperation des scores
        map50 = metrics.box.map50
        map5095 = metrics.box.map

        print("\n" + "="*50)
        print("RESULTATS SUR LE TEST SET (Images jamais vues)")
        print("="*50)
        print(f"mAP@50    : {map50:.4f}")
        print(f"mAP@50-95 : {map5095:.4f}")
        print(f"Vitesse moy.: {metrics.speed['inference']:.2f} ms / image")
        print("="*50)
        print(f"Des graphiques d'analyse ont ete generes dans : {RESULTS_DIR}/eval_test_set/")

        return map50, map5095

    except Exception as e:
        print("[ERREUR] L'evaluation a echoue. Avez-vous bien defini 'test:' dans votre data.yaml ?")
        print(f"Detail : {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640, help="Resolution de l'image")
    parser.add_argument("--device", default="0", help="GPU id (ex: 0)")
    args = parser.parse_args()

    evaluate_on_test_set(MODEL_PATH, args.imgsz, args.device)

if __name__ == "__main__":
    main()