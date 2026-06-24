# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random

# ==============================
# CONFIGURATION
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

# Chemins vers les sous-dossiers
ORIG_IMG_DIR = BASE_DIR / "dataset/images/Train_Init_Stratified"
ORIG_LBL_DIR = BASE_DIR / "dataset/labels/Train_Init_Stratified"

SPLIT4_IMG_DIR = BASE_DIR / "dataset_split4/images/Train_Init_Stratified"
SPLIT4_LBL_DIR = BASE_DIR / "dataset_split4/labels/Train_Init_Stratified"

SPLIT16_IMG_DIR = BASE_DIR / "dataset_split16/images/Train_Init_Stratified"
SPLIT16_LBL_DIR = BASE_DIR / "dataset_split16/labels/Train_Init_Stratified"

def draw_boxes_on_image(ax, img_path, lbl_path, title):
    """Dessine une image et ses boites YOLO sur un axe matplotlib donne."""
    if not img_path.exists():
        print(f"Image introuvable: {img_path}")
        return
        
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    h, w, _ = img.shape

    if lbl_path.exists():
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c, xc, yc, bw, bh = map(float, parts)
                    
                    box_w = bw * w
                    box_h = bh * h
                    x_min = (xc * w) - (box_w / 2)
                    y_min = (yc * h) - (box_h / 2)

                    rect = patches.Rectangle(
                        (x_min, y_min),
                        box_w,
                        box_h,
                        linewidth=2,
                        edgecolor='#2ecc71',
                        facecolor='none'
                    )
                    ax.add_patch(rect)

def generate_split_visualization():
    print("Recherche d'une image avec des objets...")

    labeled_images = []
    for lbl in ORIG_LBL_DIR.glob("*.txt"):
        if os.path.getsize(lbl) > 0:
            labeled_images.append(lbl.stem)

    if not labeled_images:
        print("[ERREUR] Aucune image annotee trouvee.")
        return

    random.seed(20)
    target_img_name = random.choice(labeled_images)
    print(f"Image selectionnee : {target_img_name}")

    orig_img = ORIG_IMG_DIR / f"{target_img_name}.jpg"
    orig_lbl = ORIG_LBL_DIR / f"{target_img_name}.txt"

    split4_img = SPLIT4_IMG_DIR / f"{target_img_name}_0_0.jpg"
    split4_lbl = SPLIT4_LBL_DIR / f"{target_img_name}_0_0.txt"

    split16_img = SPLIT16_IMG_DIR / f"{target_img_name}_0_0.jpg"
    split16_lbl = SPLIT16_LBL_DIR / f"{target_img_name}_0_0.txt"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))


    draw_boxes_on_image(
        axes[0],
        orig_img,
        orig_lbl,
        "Image originale (1000x1000)\nDetection de petits objets"
    )

    if split4_img.exists():
        draw_boxes_on_image(
            axes[1],
            split4_img,
            split4_lbl,
            "Split 4 (Grid 2x2)\nPatch 500px -> 1024px"
        )
    else:
        axes[1].text(0.5, 0.5, "Patch Split4 introuvable", ha='center', va='center')

    if split16_img.exists():
        draw_boxes_on_image(
            axes[2],
            split16_img,
            split16_lbl,
            "Split 16 (Grid 4x4)\nPatch 250px -> 1024px"
        )
    else:
        axes[2].text(0.5, 0.5, "Patch Split16 introuvable", ha='center', va='center')

    plt.tight_layout()
    output_file = "visual_proof_splits.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Image generee avec succes : {output_file}")

if __name__ == "__main__":
    generate_split_visualization()