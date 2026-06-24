# scenario_ceal_airplanes.py
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
# CONFIGURATION (AVIONS UNIQUEMENT)
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

# YAML pointant vers le dataset specifique avions
DATA_YAML = BASE_DIR / "data_airplanes.yaml"

# Modele de base pre-entraine sur Train_Init_Stratified (Avions)
BASE_MODEL_PATH = BASE_DIR / "code/trained_models/baseline_airplanes_20pct_yolov8l_1024/weights/best.pt"

# Nouveaux dossiers filtres (contenant uniquement les avions)
MISSION_POOL_DIR = BASE_DIR / "dataset_airplanes/images/unlabeled_pool"
LABELS_POOL_DIR = BASE_DIR / "dataset_airplanes/labels/unlabeled_pool"
VAL_DIR = BASE_DIR / "dataset_airplanes/images/val"
TEST_DIR = BASE_DIR / "dataset_airplanes/images/test"

RESULTS_DIR = BASE_DIR / "code/AL_Experiments_CEAL"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_SAVE_DIR"] = str(RESULTS_DIR)

CLASS_NAMES = ["Boeing747", "Boeing777", "Boeing787", "A321", "A330", "A350"]

# ==============================
# PARAMETRES CEAL (Cost-Effective AL)
# ==============================
# Temps fixe d'ouverture d'une image par l'humain (secondes)
TIME_PER_IMAGE_SEC = 2.0 
# Temps necessaire pour tracer manuellement une boite englobante (secondes)
TIME_PER_BBOX_SEC = 3.5  

# ==============================
# SELECTION & COST COMPUTATION
# ==============================
def compute_tile_score_and_cost(results, class_counts, conf_threshold=0.15):
    """
    Retourne le Score Informatif (Incertitude*Rarete) 
    ET le cout en secondes estime pour annoter cette image.
    """
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

        # Rarete sur les 6 classes avions
        Nc = class_counts.get(cls_id, 0)
        Wc = 1.0 / np.log(np.e + Nc)

        total_score += U * Wc
        valid_boxes += 1

    if valid_boxes == 0:
        return 0.0, 0.0
        
    # Cout = Ouverture + (Nb_boites * temps_par_boite)
    annotation_cost = TIME_PER_IMAGE_SEC + (valid_boxes * TIME_PER_BBOX_SEC)

    return total_score, annotation_cost

# ==============================
# ACTIVE LEARNING (Single Run)
# ==============================
def execute_single_run(strategy, budget_minutes, missions, imgsz, device, run_id, all_images):
    print(f"\n" + "-"*50)
    print(f"--- DEMARRAGE DU RUN {run_id} ({strategy.upper()} - Budget: {budget_minutes}min/mission) ---")
    print("-"*50)

    exp_dir = RESULTS_DIR / f"CEAL_{strategy}_{budget_minutes}min" / f"run_{run_id}"

    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    (exp_dir / "images").mkdir(parents=True, exist_ok=True)
    (exp_dir / "labels").mkdir(parents=True, exist_ok=True)

    al_yaml = exp_dir / "al_dataset.yaml"

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
    budget_seconds = budget_minutes * 60.0

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
    history.append({"Run": run_id, "Mission": 0, "mAP50": res.box.map50, "Time_Spent_s": 0.0})

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
            score, cost = compute_tile_score_and_cost(res, class_counts)
            
            if score > 0 and cost > 0:
                # Calcul de l'Efficacite = Info par seconde
                efficiency = score / cost
                scored.append({
                    'img': img, 
                    'score': score, 
                    'cost': cost, 
                    'efficiency': efficiency
                })

        # ==========================
        # TRI ET SELECTION (LE KNAPSACK)
        # ==========================
        if strategy == "top_efficiency":
            scored.sort(key=lambda x: x['efficiency'], reverse=True)
        elif strategy == "top_score_only":
            scored.sort(key=lambda x: x['score'], reverse=True)
        elif strategy == "random":
            random.shuffle(scored)

        selected = []
        time_spent = 0.0
        
        for item in scored:
            if time_spent + item['cost'] <= budget_seconds:
                selected.append(item)
                time_spent += item['cost']

        print(f"[SELECTION] {len(selected)} images retenues.")
        print(f"[BUDGET] Temps consomme : {time_spent/60:.2f} min / {budget_minutes} min")

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

        print(f"[DATASET] Taille cumulative : {len(list((exp_dir/'images').glob('*.jpg')))} images.")

        # ==========================
        # TRAIN
        # ==========================
        print("[TRAIN] Fine-tuning en cours...")
        train_model = YOLO(current_model_path)
        train_model.train(
            data=str(al_yaml),
            epochs=3,
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
        history.append({
            "Run": run_id, 
            "Mission": m+1, 
            "mAP50": res.box.map50,
            "Time_Spent_s": time_spent
        })

    return history


# ==============================
# MULTI RUN
# ==============================
def run_multiple_experiments(strategy, budget, missions, num_runs, imgsz, device):
    print("="*60)
    print(f"CAMPAGNE CEAL : {strategy.upper()} avec budget de {budget} min")
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
            budget_minutes=budget, 
            missions=missions, 
            imgsz=imgsz, 
            device=device, 
            run_id=run_id, 
            all_images=current_run_images
        )
        all_runs_history.extend(run_history)

    df_all = pd.DataFrame(all_runs_history)
    
    summary = df_all.groupby('Mission').agg(
        mAP50_mean=('mAP50', 'mean'),
        mAP50_std=('mAP50', 'std'),
        Time_Spent_mean=('Time_Spent_s', 'mean')
    ).reset_index()

    final_dir = RESULTS_DIR / f"CEAL_{strategy}_{budget}min"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    df_all.to_csv(final_dir / "all_runs_raw_metrics.csv", index=False)
    summary.to_csv(final_dir / "summary_metrics.csv", index=False)

    print("\n" + "="*60)
    print(f"RESULTATS FINAUX AGREGES ({num_runs} RUNS) - {strategy.upper()} {budget}min")
    print("="*60)
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["top_efficiency", "top_score_only", "random"], required=True)
    parser.add_argument("--budget", type=int, default=10, help="Budget temps en minutes par mission")
    parser.add_argument("--missions", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", default="0")
    
    args = parser.parse_args()

    run_multiple_experiments(
        strategy=args.strategy,
        budget=args.budget,
        missions=args.missions,
        num_runs=args.runs, 
        imgsz=1024,
        device=args.device
    )