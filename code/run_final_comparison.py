import argparse
import shutil
import random
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os
import time

import matplotlib
matplotlib.use('Agg')

# ==============================
# CONFIGURATION FULL DATASET
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()
DATA_YAML = BASE_DIR / "data.yaml"

# Modele initial (20% Stratifie sur 37 classes)
BASE_MODEL_PATH = BASE_DIR / "code/trained_models/baseline_stratified_20pct_yolov8l_1024/weights/best.pt"

MISSION_POOL_DIR = BASE_DIR / "dataset/images/Unlabeled_Pool_Stratified"
LABELS_POOL_DIR = BASE_DIR / "dataset/labels/Unlabeled_Pool_Stratified"
VAL_DIR = BASE_DIR / "dataset/images/val"
TEST_DIR = BASE_DIR / "dataset/images/test"

RESULTS_DIR = BASE_DIR / "code/AL_Final_Comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_SAVE_DIR"] = str(RESULTS_DIR)

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

TIME_PER_IMAGE_SEC = 2.0 
TIME_PER_BBOX_SEC = 3.5  

# ==============================
# SCORING 
# ==============================
def compute_tile_score_and_cost(results, class_counts, conf_threshold=0.15):
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0, 0.0

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

    if valid_boxes == 0:
        return 0.0, 0.0
        
    annotation_cost = TIME_PER_IMAGE_SEC + (valid_boxes * TIME_PER_BBOX_SEC)
    return total_score, annotation_cost

# ==============================
# ACTIVE LEARNING LOOP
# ==============================
def execute_run(strategy, budget_value, missions, imgsz, device, run_id, all_images):
    """
    strategy: "lightweight" (budget_value = nb_images) ou "ceal" (budget_value = minutes)
    """
    print(f"\n--- RUN {run_id} FULL DATASET ({strategy.upper()} - Budget: {budget_value}) ---")
    exp_dir = RESULTS_DIR / f"{strategy}_{budget_value}" / f"run_{run_id}"
    if exp_dir.exists(): 
        shutil.rmtree(exp_dir)
    (exp_dir / "images").mkdir(parents=True, exist_ok=True)
    (exp_dir / "labels").mkdir(parents=True, exist_ok=True)

    al_yaml = exp_dir / "al_dataset.yaml"
    with open(al_yaml, "w") as f:
        f.write(f"path: {exp_dir}\ntrain: images\nval: {VAL_DIR}\ntest: {TEST_DIR}\nnc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")

    current_model_path = BASE_MODEL_PATH
    class_counts = {}
    pool_size = max(1, len(all_images) // missions)
    history = []

    print(f"[RUN {run_id}] Evaluation Initiale (Baseline)...")
    model = YOLO(current_model_path)
    res = model.val(data=str(DATA_YAML), split="test", imgsz=imgsz, device=device, project=str(exp_dir), name="eval_init", exist_ok=True, verbose=False)
    history.append({
        "Run": run_id, "Mission": 0, "mAP50": res.box.map50, 
        "Images_Transmises": 0, "Temps_Annotation_s": 0.0, "Temps_Entrainement_s": 0.0
    })

    for m in range(missions):
        print(f"\n[MISSION {m+1}]")
        mission_imgs = all_images[m*pool_size:(m+1)*pool_size]
        model = YOLO(current_model_path)
        scored = []

        for img in mission_imgs:
            res = model(str(img), imgsz=imgsz, device=device, verbose=False)
            score, cost = compute_tile_score_and_cost(res, class_counts)
            if score > 0 and cost > 0:
                efficiency = score / cost 
                scored.append({'img': img, 'score': score, 'cost': cost, 'efficiency': efficiency})

        selected = []
        time_spent_annotation = 0.0

        if strategy == "lightweight":
            # Budget en NOMBRE D'IMAGES
            scored.sort(key=lambda x: x['score'], reverse=True)
            selected = scored[:budget_value]
            time_spent_annotation = sum(item['cost'] for item in selected)
            
        elif strategy == "ceal":
            # Budget en TEMPS
            budget_seconds = budget_value * 60.0
            scored.sort(key=lambda x: x['efficiency'], reverse=True)
            for item in scored:
                if time_spent_annotation + item['cost'] <= budget_seconds:
                    selected.append(item)
                    time_spent_annotation += item['cost']

        print(f"[SELECTION] {len(selected)} images retenues (Cout estime: {time_spent_annotation/60:.1f} min).")

        for item in selected:
            img = item['img']
            shutil.copy(img, exp_dir / "images")
            lbl = LABELS_POOL_DIR / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, exp_dir / "labels")
                with open(lbl) as f:
                    for line in f:
                        c = int(line.split()[0])
                        class_counts[c] = class_counts.get(c, 0) + 1

        print("[TRAIN] Fine-tuning...")
        train_start = time.time()
        train_model = YOLO(current_model_path)
        train_model.train(
            data=str(al_yaml), epochs=10, patience=0, freeze=10, lr0=0.001,
            imgsz=imgsz, batch=4, device=device, workers=0, project=str(exp_dir), name=f"train_m{m+1}", exist_ok=True, val=False
        )
        train_time = time.time() - train_start

        current_model_path = exp_dir / f"train_m{m+1}/weights/last.pt"
        eval_model = YOLO(current_model_path)
        res = eval_model.val(data=str(DATA_YAML), split="test", imgsz=imgsz, device=device, project=str(exp_dir), name=f"eval_m{m+1}", exist_ok=True, verbose=False)
        
        print(f"mAP50 = {res.box.map50:.4f}")
        history.append({
            "Run": run_id, 
            "Mission": m+1, 
            "mAP50": res.box.map50, 
            "Images_Transmises": len(selected),
            "Temps_Annotation_s": time_spent_annotation,
            "Temps_Entrainement_s": train_time
        })

    return history

def run_multiple_experiments(strategy, budget_value, missions, num_runs, imgsz, device):
    print("="*60)
    print(f"CAMPAGNE FINALE : {strategy.upper()} - Budget: {budget_value}")
    print("="*60)

    all_images_pool = list(MISSION_POOL_DIR.glob("*.jpg"))
    if len(all_images_pool) == 0: 
        return

    all_runs_history = []
    for run_id in range(1, num_runs + 1):
        random.seed(42 + run_id)
        current_run_images = all_images_pool.copy()
        random.shuffle(current_run_images)

        run_history = execute_run(strategy, budget_value, missions, imgsz, device, run_id, current_run_images)
        all_runs_history.extend(run_history)

    df_all = pd.DataFrame(all_runs_history)
    
    summary = df_all.groupby('Mission').agg(
        mAP50_mean=('mAP50', 'mean'), 
        mAP50_std=('mAP50', 'std'), 
        Images_Transmises_mean=('Images_Transmises', 'mean'),
        Temps_Annotation_min=('Temps_Annotation_s', lambda x: x.mean() / 60.0),
        Temps_Entrainement_min=('Temps_Entrainement_s', lambda x: x.mean() / 60.0)
    ).reset_index()

    final_dir = RESULTS_DIR / f"{strategy}_{budget_value}"
    final_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(final_dir / "all_runs_raw_metrics.csv", index=False)
    summary.to_csv(final_dir / "summary_metrics.csv", index=False)
    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["lightweight", "ceal"], required=True)
    parser.add_argument("--budget", type=int, required=True, help="Nb d'images pour 'lightweight', Minutes pour 'ceal'")
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    run_multiple_experiments(args.strategy, args.budget, missions=4, num_runs=3, imgsz=1024, device=args.device)