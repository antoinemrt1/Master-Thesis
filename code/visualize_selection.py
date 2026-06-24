# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

# ==============================
# CONFIGURATION
# ==============================
# Chemins vers les dossiers de la Mission 1
TOP_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_V6/AL_top_10pct/run_1")
BOT_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_V6/AL_bottom_10pct/run_1")

# On utilise les noms de classes pour l'affichage
CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

# Couleurs pour differencier les boites
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#34495e']

def draw_boxes(ax, img_path, label_path):
    """Draw image and YOLO bounding boxes."""
    img = mpimg.imread(str(img_path))
    ax.imshow(img)
    ax.axis('off')

    h, w, _ = img.shape

    if label_path.exists():
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        show_text = len(lines) < 50

        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])

            box_w = bw * w
            box_h = bh * h
            x_min = (xc * w) - (box_w / 2)
            y_min = (yc * h) - (box_h / 2)

            color = COLORS[cls_id % len(COLORS)]

            rect = patches.Rectangle(
                (x_min, y_min),
                box_w,
                box_h,
                linewidth=1.5,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)

            if show_text:
                ax.text(
                    x_min,
                    y_min - 2,
                    CLASS_NAMES[cls_id],
                    color='white',
                    fontsize=8,
                    weight='bold',
                    bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=1)
                )

def generate_comparison_plot():
    print("Generating comparison image...")

    top_images = list((TOP_DIR / "images").glob("*.jpg"))
    bot_images = list((BOT_DIR / "images").glob("*.jpg"))

    if not top_images or not bot_images:
        print("[ERROR] Cannot find images in TOP/BOT folders.")
        return

    random.seed(42)
    top_sample = random.sample(top_images, 3)
    bot_sample = random.sample(bot_images, 3)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    fig.suptitle(
        "Visualization of Selection Strategies (Active Learning)",
        fontsize=20,
        fontweight='bold',
        y=0.98
    )

    # TOP row
    for i, img_path in enumerate(top_sample):
        ax = axes[0, i]
        lbl_path = TOP_DIR / "labels" / (img_path.stem + ".txt")
        draw_boxes(ax, img_path, lbl_path)

        num_objs = len(open(lbl_path).readlines()) if lbl_path.exists() else 0

        if i == 1:
            ax.set_title(
                f"TOP Strategy (Density & Uncertainty)\nExample: {num_objs} complex objects",
                fontsize=14,
                fontweight='bold',
                color='#27ae60',
                pad=10
            )
        else:
            ax.set_title(f"{num_objs} objects", fontsize=12)

    # BOTTOM row
    for i, img_path in enumerate(bot_sample):
        ax = axes[1, i]
        lbl_path = BOT_DIR / "labels" / (img_path.stem + ".txt")
        draw_boxes(ax, img_path, lbl_path)

        num_objs = len(open(lbl_path).readlines()) if lbl_path.exists() else 0

        if i == 1:
            ax.set_title(
                f"BOTTOM Strategy (High confidence)\nExample: {num_objs} simple object",
                fontsize=14,
                fontweight='bold',
                color='#e74c3c',
                pad=10
            )
        else:
            ax.set_title(f"{num_objs} object(s)", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.05)

    output_filename = "visual_comparison_top_vs_bottom.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print(f"[SUCCESS] Image generated: {output_filename}")

if __name__ == "__main__":
    generate_comparison_plot()