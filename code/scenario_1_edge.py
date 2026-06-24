# scenario_1_edge.py
import time
import numpy as np
from ultralytics import YOLO
import config
from smart_selection import SmartTileSelector

def run_scenario_1(device="0"):
    print("="*60)
    print("SCENARIO 1 : CONTRAINTES REALISTES EMBARQUEES")
    print("="*60)

    # 1. Chargement du modele embarque
    model = YOLO(config.MODEL_LR_START)
    selector = SmartTileSelector()  # Historique vide car pas d'AL
    
    # On simule un vol de drone avec 1000 images (tuiles)
    all_images = list(config.TRAIN_IMG_DIR.glob("*.jpg"))[:1000]
    
    scored_tiles = []
    latencies = []
    
    print(f"[UAV] Survol de {len(all_images)} tuiles en cours...")
    
    # 2. Boucle d'inference (Single-Pass)
    for img_path in all_images:
        start_t = time.time()
        
        # Inference YOLO (imgsz=640, device=0)
        result = model(str(img_path), imgsz=config.IMG_RES_EMBEDDED, device=device, verbose=False)[0]
        
        end_t = time.time()
        latencies.append((end_t - start_t) * 1000)  # En millisecondes
        
        # Scoring de la tuile
        score = selector.score_tile(result)
        scored_tiles.append((img_path, score))

    # 3. Selection (Top-K)
    # Tri decroissant selon la valeur informative
    scored_tiles.sort(key=lambda x: x[1], reverse=True)
    
    # Transmission du budget alloue
    top_k_tiles = scored_tiles[:config.BUDGET_K_PER_MISSION]
    
    # 4. Metriques et Resultats
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    data_reduction = 100 - ((len(top_k_tiles) / len(all_images)) * 100)
    
    print("\n--- RESULTATS SCENARIO 1 ---")
    print(f"Latence moyenne d'inference : {avg_latency:.2f} ms / image")
    print(f"Respect de la contrainte (<= 100ms) : {'OUI' if avg_latency <= 100 else 'NON'}")
    print(f"Tuiles transmises : {len(top_k_tiles)} sur {len(all_images)}")
    print(f"Economie de Bande Passante : {data_reduction:.1f} %")
    print("="*60)

if __name__ == "__main__":
    run_scenario_1()