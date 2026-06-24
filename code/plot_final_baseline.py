# plot_final_baseline.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Config style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# --- CHEMIN DU FICHIER ---
CSV_PATH = Path("/linux/antoimartin/v2/code/trained_models/baseline_init_20pct_yolov8l_1024/results.csv")
OUTPUT_DIR = CSV_PATH.parent

def main():
    if not CSV_PATH.exists():
        print(f"[ERREUR] Le fichier {CSV_PATH} n'existe pas.")
        return

    print(f"Chargement des donnees depuis {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Nettoyer les noms de colonnes
    df.columns = [c.strip() for c in df.columns]

    # ?? FIX CRUCIAL (pandas -> numpy)
    epochs = df['epoch'].values
    map50 = df['metrics/mAP50(B)'].values
    map5095 = df['metrics/mAP50-95(B)'].values
    train_loss = df['train/box_loss'].values
    val_loss = df['val/box_loss'].values

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Graphique 1 : mAP ---
    ax1.plot(epochs, map50, label='mAP@50', color='#3498db', linewidth=2.5)
    ax1.plot(epochs, map5095, label='mAP@50-95', color='#2ecc71', linewidth=2.5)
    
    ax1.set_title("Evolution de la Precision (mAP) - Baseline Init 20%", fontweight='bold')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Score")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 2 : Loss ---
    ax2.plot(epochs, train_loss, label='Train Box Loss', color='#e74c3c', linewidth=2)
    ax2.plot(epochs, val_loss, label='Val Box Loss', color='#e67e22', linewidth=2, linestyle='--')
    
    ax2.set_title("Convergence des Pertes (Box Loss)", fontweight='bold')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    output_img = OUTPUT_DIR / "baseline_init_metrics.png"
    plt.savefig(output_img, dpi=300)
    print(f"[SUCCESS] Graphique sauvegarde ici : {output_img}")
    plt.show()

if __name__ == "__main__":
    main()