# -*- coding: utf-8 -*-

import os
import shutil
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path("/linux/antoimartin/v2/dataset").resolve()

# Dossiers sources (Doivent contenir TOUTES les donnees d'entrainement)
TRAIN_IMG_DIR = BASE_DIR / "images" / "train"
TRAIN_LBL_DIR = BASE_DIR / "labels" / "train"

# Dossiers cibles pour le Split Stratifie
INIT_IMG_DIR = BASE_DIR / "images" / "Train_Init"
INIT_LBL_DIR = BASE_DIR / "labels" / "Train_Init"
POOL_IMG_DIR = BASE_DIR / "images" / "Unlabeled_Pool"
POOL_LBL_DIR = BASE_DIR / "labels" / "Unlabeled_Pool"

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

def analyze_and_plot_distribution():
    print("--- 1. ANALYSE DU DATASET ---")
    class_counts = defaultdict(int)
    image_to_classes = defaultdict(list)
    
    all_labels = list(TRAIN_LBL_DIR.glob("*.txt"))
    
    for lbl_path in tqdm(all_labels, desc="Lecture des labels"):
        with open(lbl_path, 'r') as f:
            for line in f:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1
                image_to_classes[lbl_path.stem].append(cls_id)
                
    df_counts = pd.DataFrame([
        {"Classe": CLASS_NAMES[k], "Instances": v, "ID": k} 
        for k, v in class_counts.items()
    ])
    df_counts = df_counts.sort_values(by="Instances", ascending=False)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_counts, x="Classe", y="Instances", palette="viridis")
    plt.xticks(rotation=90)
    plt.yscale('log')
    plt.title("Distribution des Instances par Classe dans FAIR1M (Train Set)\nEchelle Logarithmique", fontweight='bold')
    plt.ylabel("Nombre d'instances (Log)")
    plt.xlabel("Classes")
    plt.tight_layout()
    plt.savefig("class_distribution_fair1m.png", dpi=300)
    print("\n[SUCCESS] Graphique sauvegarde : class_distribution_fair1m.png")
    
    return df_counts, image_to_classes

def perform_stratified_split(df_counts, image_to_classes, init_ratio=0.20):
    print(f"\n--- 2. SPLIT STRATIFIE ({init_ratio*100}%) ---")
    
    INIT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    INIT_LBL_DIR.mkdir(parents=True, exist_ok=True)
    POOL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    POOL_LBL_DIR.mkdir(parents=True, exist_ok=True)
    
    target_counts = {row['ID']: int(row['Instances'] * init_ratio) for _, row in df_counts.iterrows()}
    
    for k in target_counts:
        if target_counts[k] == 0 and df_counts.loc[df_counts['ID'] == k, 'Instances'].values[0] > 0:
            target_counts[k] = 1

    current_counts = defaultdict(int)
    selected_for_init = set()
    
    all_images = list(image_to_classes.keys())
    random.seed(42)
    random.shuffle(all_images)
    
    rare_to_frequent = df_counts.sort_values(by="Instances", ascending=True)['ID'].tolist()
    
    print("Selection intelligente des images pour garantir les classes rares...")
    for cls_id in rare_to_frequent:
        for img in all_images:
            if current_counts[cls_id] >= target_counts[cls_id]:
                break
                
            if img not in selected_for_init and cls_id in image_to_classes[img]:
                selected_for_init.add(img)
                for c in image_to_classes[img]:
                    current_counts[c] += 1
                    
    target_image_count = int(len(all_images) * init_ratio)
    print(f"Images selectionnees : {len(selected_for_init)}")
    print(f"Cible totale : {target_image_count}")
    
    for img in all_images:
        if len(selected_for_init) >= target_image_count:
            break
        if img not in selected_for_init:
            selected_for_init.add(img)
            
    print("\nDeplacement des fichiers...")
    for img_stem in tqdm(all_images, desc="Deplacement Images & Labels"):
        img_path = TRAIN_IMG_DIR / f"{img_stem}.jpg"
        lbl_path = TRAIN_LBL_DIR / f"{img_stem}.txt"
        
        if img_stem in selected_for_init:
            if img_path.exists(): shutil.move(str(img_path), str(INIT_IMG_DIR / img_path.name))
            if lbl_path.exists(): shutil.move(str(lbl_path), str(INIT_LBL_DIR / lbl_path.name))
        else:
            if img_path.exists(): shutil.move(str(img_path), str(POOL_IMG_DIR / img_path.name))
            if lbl_path.exists(): shutil.move(str(lbl_path), str(POOL_LBL_DIR / lbl_path.name))
            
    print("\n[SUCCESS] Split stratifie termine.")

if __name__ == "__main__":
    df, img_dict = analyze_and_plot_distribution()
    perform_stratified_split(df, img_dict, init_ratio=0.20)