# scenario_2_al_loop.py
import matplotlib
matplotlib.use('Agg')
import time
import shutil
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import config
from smart_selection import SmartTileSelector

def setup_al_environment():
    """Prepare les dossiers pour le dataset cumulatif d'Active Learning."""
    if config.AL_DATASET_DIR.exists():
        shutil.rmtree(config.AL_DATASET_DIR)
    
    (config.AL_DATASET_DIR / "images" / "train").mkdir(parents=True)
    (config.AL_DATASET_DIR / "labels" / "train").mkdir(parents=True)
    
    # Creation du data.yaml pour l'AL (qui utilise le test set global pour la val)
    al_yaml_path = config.EXPERIMENT_DIR / "al_data.yaml"
    with open(al_yaml_path, "w") as f:
        f.write(f"path: {config.AL_DATASET_DIR.absolute()}\n")
        f.write("train: images/train\n")
        f.write(f"val: {config.TEST_IMG_DIR.absolute()}\n")  # On valide directement sur le Test Set
        f.write(f"nc: {len(config.CLASS_NAMES)}\n")
        f.write(f"names: {config.CLASS_NAMES}\n")
    return al_yaml_path

def simulate_human_annotation(selected_image_paths, selector):
    """
    Simule l'Humain: Copie l'image et son VRAI label (Ground Truth) dans le dataset AL.
    Met a jour l'historique des classes pour la prochaine mission.
    """
    new_annotations = 0
    for img_path in selected_image_paths:
        # Copie Image
        dest_img = config.AL_DATASET_DIR / "images" / "train" / img_path.name
        shutil.copy(img_path, dest_img)
        
        # Copie Label (Ground Truth)
        lbl_path = config.TRAIN_LBL_DIR / f"{img_path.stem}.txt"
        dest_lbl = config.AL_DATASET_DIR / "labels" / "train" / lbl_path.name
        
        if lbl_path.exists():
            shutil.copy(lbl_path, dest_lbl)
            
            # Met a jour le compteur Nc de CALD-UAV
            with open(lbl_path, 'r') as f:
                classes_in_file = [int(line.split()[0]) for line in f]
                selector.update_history(classes_in_file)
                new_annotations += len(classes_in_file)
        else:
            # Creer un fichier vide si pas de label
            open(dest_lbl, 'a').close()
            
    return new_annotations

def run_scenario_2(device="0"):
    print("="*60)
    print("SCENARIO 2 : ACTIVE LEARNING INTER-MISSIONS")
    print("="*60)

    al_yaml_path = setup_al_environment()
    selector = SmartTileSelector()
    
    # On definit le pool "Unlabeled" comme l'ensemble des images d'entrainement disponibles
    unlabeled_pool = list(config.TRAIN_IMG_DIR.glob("*.jpg"))
    
    # Melange aleatoire pour simuler differents vols
    import random
    random.seed(42)
    random.shuffle(unlabeled_pool)

    # Initialisation avec le modele V0 (Golden LR)
    current_model_path = config.MODEL_LR_START
    results_log = []

    for mission_id in range(1, config.MISSIONS + 1):
        print(f"\n=== MISSION {mission_id}/{config.MISSIONS} ===")
        
        model_uav = YOLO(current_model_path)
        
        # 1. VOL & INFERENCE (Simulation UAV)
        batch_size = len(unlabeled_pool) // config.MISSIONS
        mission_images = unlabeled_pool[(mission_id-1)*batch_size : mission_id*batch_size]
        
        print(f"[UAV] Exploration de {len(mission_images)} tuiles...")
        scored_tiles = []
        
        for img_path in mission_images:
            res = model_uav(str(img_path), imgsz=config.IMG_RES_EMBEDDED, device=device, verbose=False)[0]
            score = selector.score_tile(res)
            scored_tiles.append((img_path, score))

        # 2. SELECTION TOP-K
        scored_tiles.sort(key=lambda x: x[1], reverse=True)
        selected_tiles = [x[0] for x in scored_tiles[:config.BUDGET_K_PER_MISSION]]
        print(f"[TRANSMISSION] {len(selected_tiles)} tuiles transmises au Sol.")

        # 3. ANNOTATION HUMAINE (Sol)
        new_annots = simulate_human_annotation(selected_tiles, selector)
        total_images_al = len(list((config.AL_DATASET_DIR / "images" / "train").glob("*.jpg")))
        
        print(f"[SOL] Annotateur : +{new_annots} objets. Taille du dataset AL cumulatif : {total_images_al} images.")

        # 4. FINE-TUNING (Sol - GTX 1080 Ti)
        print(f"[SOL] Fine-tuning du modele M{mission_id} sur le dataset cumulatif...")
        model_sol = YOLO(current_model_path)
        
        train_res = model_sol.train(
            data=str(al_yaml_path),
            epochs=config.FINE_TUNE_EPOCHS,
            imgsz=config.IMG_RES_EMBEDDED,
            batch=config.BATCH_SIZE_TRAIN,
            project=str(config.MODELS_DIR),
            name=f"mission_{mission_id}",
            exist_ok=True,
            device=1,
            workers=4,
            val=True
        )
        
        # Sauvegarde des performances du modele M_k
        current_model_path = config.MODELS_DIR / f"mission_{mission_id}" / "weights" / "best.pt"
        
        map50 = train_res.box.map50
        map5095 = train_res.box.map
        
        print(f"[FIN MISSION {mission_id}] mAP@50 evalue sur Test Set : {map50:.4f}")
        
        results_log.append({
            "Mission": mission_id,
            "Images_Transmises": total_images_al,
            "mAP50": map50,
            "mAP50-95": map5095
        })

    # === BILAN FINAL ===
    df = pd.DataFrame(results_log)
    df.to_csv(config.EXPERIMENT_DIR / "active_learning_results.csv", index=False)
    
    print("\n" + "="*60)
    print("BILAN DE LA CAMPAGNE D'ACTIVE LEARNING")
    print(df.to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    run_scenario_2()