# -*- coding: utf-8 -*-

import argparse
import torch
from pathlib import Path
from ultralytics import YOLO
import traceback

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "trained_models_splits"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_split_baseline(split_version, model_type, imgsz, batch, device):
    """
    Entraine la Baseline (Train_Init 20%) sur un dataset 'Super-Resolution'.
    """
    print(f"\n" + "="*50)
    print(f"ENTRAINEMENT BASELINE : {split_version.upper()}")
    print("="*50)
    
    # Choix du YAML en fonction de l'argument
    if split_version == "split4":
        yaml_path = Path("/linux/antoimartin/v2/data_split4.yaml").resolve()
    elif split_version == "split16":
        yaml_path = Path("/linux/antoimartin/v2/data_split16.yaml").resolve()
    else:
        print("Erreur: split_version doit etre 'split4' ou 'split16'")
        return

    print(f"[TRAIN] Modele : {model_type} @ {imgsz}px sur les GPUs {device}")
    print(f"[DATA] YAML : {yaml_path}")

    model = YOLO(model_type)
    run_name = f"baseline_init_{split_version}_{Path(model_type).stem}_{imgsz}"

    try:
        results = model.train(
            data=str(yaml_path),
            epochs=200,            
            imgsz=imgsz,           
            batch=batch,           
            project=str(MODELS_DIR.absolute()),
            name=run_name, 
            exist_ok=True,
            patience=20,           
            device=device,         
            workers=8,             
            amp=True               
        )

        best_pt = MODELS_DIR / run_name / "weights" / "best.pt"
        print(f"\n[SUCCES] Modele de depart sauvegarde ici : {best_pt}")

    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] L'entrainement a echoue : {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["split4", "split16", "all"], help="Quel dataset entrainer")
    parser.add_argument("--model", default="yolov8s.pt", help="Le modele de base (ex: yolov8s.pt)")
    parser.add_argument("--imgsz", type=int, default=640, help="Resolution YOLO")
    parser.add_argument("--batch", type=int, default=32, help="Taille de lot")
    parser.add_argument("--device", required=True, help="ex: 0,1 ou 0,1,2,3")
    args = parser.parse_args()

    device_arg = args.device

    if args.split in ["split4", "all"]:
        train_split_baseline("split4", args.model, args.imgsz, args.batch, device_arg)

    if args.split in ["split16", "all"]:
        train_split_baseline("split16", args.model, args.imgsz, args.batch, device_arg)

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ATTENTION: Aucun GPU detecte.")
    main()