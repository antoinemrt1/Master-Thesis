import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from pathlib import Path
import random

# --- CONFIGURATION (A adapter selon vos dossiers de resultats) ---
# On va regarder les images choisies lors de la MISSION 1 du RUN 1
TOP_EFF_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_CEAL/CEAL_top_efficiency_30min/run_1")
RANDOM_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_CEAL/CEAL_random_30min/run_1")

CLASS_NAMES = ["Boeing747", "Boeing777", "Boeing787", "A321", "A330", "A350"]
COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22']

def draw_boxes_with_cost(ax, img_path, label_path):
    """Affiche l'image, ses boites, et calcule le cout humain theorique."""
    img = mpimg.imread(str(img_path))
    ax.imshow(img)
    ax.axis('off')
    h, w, _ = img.shape
    
    num_boxes = 0
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = f.readlines()
            num_boxes = len(lines)
            
            # Ne dessine les noms de classes que s'il y a peu d'avions (pour la lisibilite)
            show_text = num_boxes < 15
            
            for line in lines:
                parts = line.split()
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                
                box_w, box_h = bw * w, bh * h
                x_min, y_min = (xc * w) - (box_w / 2), (yc * h) - (box_h / 2)
                
                color = COLORS[cls_id % len(COLORS)]
                rect = patches.Rectangle((x_min, y_min), box_w, box_h, 
                                         linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
                
                if show_text:
                    ax.text(x_min, y_min - 5, CLASS_NAMES[cls_id], 
                            color='white', fontsize=9, weight='bold',
                            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2))
                            
    # Calcul du cout (Rappel: 2s ouverture + 3.5s/boite)
    cost_sec = 2.0 + (num_boxes * 3.5)
    
    # Affichage du bandeau de cout en bas de l'image
    ax.text(0.5, 0.05, f"{num_boxes} avions | Cout estime : {cost_sec:.1f}s", 
            horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes,
            color='black', fontsize=12, weight='bold',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', pad=5, boxstyle='round,pad=0.5'))

def generate_visual_proof():
    print("Generation de la preuve visuelle CEAL...")
    top_images = list((TOP_EFF_DIR / "images").glob("*.jpg"))
    rnd_images = list((RANDOM_DIR / "images").glob("*.jpg"))

    if not top_images or not rnd_images:
        print("[ERREUR] Images introuvables. Verifiez les chemins TOP_EFF_DIR et RANDOM_DIR.")
        return

    random.seed(42)  # Pour reproductibilite des images choisies
    # On prend 3 images au hasard parmi celles selectionnees
    top_sample = random.sample(top_images, min(3, len(top_images)))
    rnd_sample = random.sample(rnd_images, min(3, len(rnd_images)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Preuve par l'image : Allocation du Budget d'Annotation (30 minutes)", 
                 fontsize=22, fontweight='bold', y=0.96)

    # Ligne 1 : TOP EFFICIENCY
    for i, img_path in enumerate(top_sample):
        ax = axes[0, i]
        lbl_path = TOP_EFF_DIR / "labels" / (img_path.stem + ".txt")
        draw_boxes_with_cost(ax, img_path, lbl_path)
        if i == 1:
            ax.set_title("Strategie TOP EFFICIENCY\n(Recherche du meilleur ratio Info/Temps)", 
                         fontsize=16, fontweight='bold', color='#27ae60', pad=15)

    # Ligne 2 : RANDOM
    for i, img_path in enumerate(rnd_sample):
        ax = axes[1, i]
        lbl_path = RANDOM_DIR / "labels" / (img_path.stem + ".txt")
        draw_boxes_with_cost(ax, img_path, lbl_path)
        if i == 1:
            ax.set_title("Strategie RANDOM (Temoin)\n(Selection aveugle au cout humain)", 
                         fontsize=16, fontweight='bold', color='#e74c3c', pad=15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.1)
    
    out_file = "ceal_visual_proof.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Image generee : {out_file}")

if __name__ == "__main__":
    generate_visual_proof()