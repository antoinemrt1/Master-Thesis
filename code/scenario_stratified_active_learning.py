import argparse
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os

import matplotlib
matplotlib.use('Agg')

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

DATA_YAML = BASE_DIR / "data_init_stratified.yaml"

BASE_MODEL_PATH = BASE_DIR / "code/trained_models/baseline_stratified_20pct_yolov8l_1024/weights/best.pt"

# NOUVEAUX CHEMINS STRATIFIES
MISSION_POOL_DIR = BASE_DIR / "dataset/images/Unlabeled_Pool_Stratified"
VAL_DIR = BASE_DIR / "dataset/images/val"
TEST_DIR = BASE_DIR / "dataset/images/test"

RESULTS_DIR = BASE_DIR / "code/AL_Experiments_V6"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Empeche YOLO d'utiliser /runs a la racine
os.environ["YOLO_SAVE_DIR"] = str(RESULTS_DIR)

# ==============================
# SELECTION
# ==============================
def compute_tile_score(results, class_counts, conf_threshold=0.15):
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0

    total_score = 0.0
    valid_boxes = 0

    for box in results[0].boxes:
        conf = float(box.conf.item())

        if conf < conf_threshold:
            continue

        cls_id = int(box.cls.item())

        p_best = conf
        p_second = conf * 0.8
        U = 1.0 - (p_best - p_second)

        Nc = class_counts.get(cls_id, 0)
        Wc = 1.0 / np.log(np.e + Nc)

        total_score += U * Wc
        valid_boxes += 1

    return total_score if valid_boxes > 0 else 0.0

# ==============================
# ACTIVE LEARNING (Single Run)
# ==============================
def execute_single_run(strategy, percent, missions, imgsz, device, run_id, all_images):
    print(f"\n" + "-"*40)
    print(f"--- DEMARRAGE DU RUN {run_id} ({strategy.upper()} - {percent}%) ---")
    print("-"*40)

    exp_dir = RESULTS_DIR / f"AL_{strategy}_{percent}pct" / f"run_{run_id}"

    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    (exp_dir / "images").mkdir(parents=True, exist_ok=True)
    (exp_dir / "labels").mkdir(parents=True, exist_ok=True)

    al_yaml = exp_dir / "al_dataset.yaml"

    CLASS_NAMES = [
        "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
        "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
        "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
        "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
        "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
        "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
        "Tennis Court", "Roundabout", "Intersection", "Bridge"
    ]

    with open(al_yaml, "w") as f:
        f.write(f"path: {exp_dir}\n")
        f.write(f"train: images\n")
        f.write(f"val: {VAL_DIR}\n")
        f.write(f"test: {TEST_DIR}\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    current_model_path = BASE_MODEL_PATH
    class_counts = {}

    pool_size = max(1, len(all_images) // missions)
    budget_k = max(1, int(pool_size * percent / 100))

    history = []

    # ==========================
    # INIT EVAL
    # ==========================
    print(f"[RUN {run_id}] Evaluation Initiale (Baseline)...")
    model = YOLO(current_model_path)
    res = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=imgsz,
        device=device,
        project=str(exp_dir),
        name="eval_init",
        exist_ok=True,
        verbose=False
    )
    history.append({"Run": run_id, "Mission": 0, "mAP50": res.box.map50})

    # ==========================
    # MISSIONS
    # ==========================
    for m in range(missions):
        print(f"\n[RUN {run_id}] --- MISSION {m+1}/{missions} ---")
        mission_imgs = all_images[m*pool_size:(m+1)*pool_size]

        model = YOLO(current_model_path)
        scored = []

        print(f"[INFERENCE] Analyse de {len(mission_imgs)} images...")
        for img in mission_imgs:
            res = model(str(img), imgsz=imgsz, device=device, verbose=False)
            score = compute_tile_score(res, class_counts)
            scored.append((img, score))

        if strategy == "top":
            scored.sort(key=lambda x: x[1], reverse=True)
        elif strategy == "bottom":
            scored.sort(key=lambda x: x[1])
        else:
            random.shuffle(scored)

        selected = scored[:budget_k]
        print(f"[SELECTION] {len(selected)} images retenues.")

        for img, _ in selected:
            shutil.copy(img, exp_dir / "images")
            
            lbl = BASE_DIR / "dataset/labels/Unlabeled_Pool_Stratified" / (img.stem + ".txt")

            if lbl.exists():
                shutil.copy(lbl, exp_dir / "labels")
                with open(lbl) as f:
                    for line in f:
                        c = int(line.split()[0])
                        class_counts[c] = class_counts.get(c, 0) + 1

        print(f"[DATASET CUMULATIF] Taille actuelle : {len(list((exp_dir/'images').glob('*.jpg')))} images.")

        # ==========================
        # TRAIN
        # ==========================
        print("[TRAIN] Fine-tuning en cours...")
        train_model = YOLO(current_model_path)
        train_model.train(
            data=str(al_yaml),
            epochs=10,
            patience=0,
            freeze=10,
            lr0=0.001,
            imgsz=imgsz,
            batch=4,
            device=device,
            workers=0,
            project=str(exp_dir),
            name=f"train_m{m+1}",
            exist_ok=True,
            val=False
        )

        current_model_path = exp_dir / f"train_m{m+1}/weights/last.pt"

        # ==========================
        # EVAL
        # ==========================
        print(f"[EVAL] Test du modele M{m+1}...")
        eval_model = YOLO(current_model_path)
        res = eval_model.val(
            data=str(DATA_YAML),
            split="test",
            imgsz=imgsz,
            device=device,
            project=str(exp_dir),
            name=f"eval_m{m+1}",
            exist_ok=True,
            verbose=False
        )

        print(f"mAP50 = {res.box.map50:.4f}")
        history.append({"Run": run_id, "Mission": m+1, "mAP50": res.box.map50})

    return history

# ==============================
# MULTI RUN
# ==============================
def run_multiple_experiments(strategy, percent, missions, num_runs, imgsz, device):
    print("="*60)
    print(f"CAMPAGNE EXPERIMENTALE : {strategy.upper()} a {percent}%")
    print(f"Nombre de Runs (Repetitions) : {num_runs}")
    print("="*60)

    all_images_pool = list(MISSION_POOL_DIR.glob("*.jpg"))
    if len(all_images_pool) == 0:
        print("[ERREUR] Aucun fichier image trouve dans le pool de mission.")
        return

    all_runs_history = []

    for run_id in range(1, num_runs + 1):
        random.seed(42 + run_id)
        current_run_images = all_images_pool.copy()
        random.shuffle(current_run_images)

        run_history = execute_single_run(
            strategy=strategy, 
            percent=percent, 
            missions=missions, 
            imgsz=imgsz, 
            device=device, 
            run_id=run_id, 
            all_images=current_run_images
        )
        all_runs_history.extend(run_history)

    df_all = pd.DataFrame(all_runs_history)
    
    summary = df_all.groupby('Mission')['mAP50'].agg(['mean', 'std']).reset_index()
    summary = summary.rename(columns={'mean': 'mAP50_mean', 'std': 'mAP50_std'})

    final_dir = RESULTS_DIR / f"AL_{strategy}_{percent}pct"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    df_all.to_csv(final_dir / "all_runs_raw_metrics.csv", index=False)
    summary.to_csv(final_dir / "summary_metrics.csv", index=False)

    print("\n" + "="*60)
    print(f"RESULTATS FINAUX AGREGES ({num_runs} RUNS) - {strategy.upper()} {percent}%")
    print("="*60)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["top", "bottom", "random"], required=True)
    parser.add_argument("--percent", type=int, default=10)
    parser.add_argument("--missions", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="0")
    
    args = parser.parse_args()

    run_multiple_experiments(
        strategy=args.strategy,
        percent=args.percent,
        missions=args.missions,
        num_runs=args.runs, 
        imgsz=1024,
        device=args.device
    )