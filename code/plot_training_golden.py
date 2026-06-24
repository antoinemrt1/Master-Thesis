import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

CSV_HR_PATH = Path("/linux/antoimartin/v2/code/trained_models/golden_yolov8l_1024/results.csv")
CSV_LR_PATH = Path("/linux/antoimartin/v2/code/trained_models/golden_yolov8s_640/results.csv")

def clean_dataframe(csv_path):
    """Charge et nettoie les noms de colonnes du CSV genere par YOLO."""
    df = pd.read_csv(csv_path)
    
    # Nettoyage noms colonnes
    df.columns = [c.strip() for c in df.columns]
    
    # Conversion en numerique
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Supprimer lignes NaN (important pour matplotlib)
    df = df.dropna()
    
    return df

def generate_training_plots():
    print("Chargement des donnees d'entrainement...")
    
    df_hr = clean_dataframe(CSV_HR_PATH)
    df_lr = clean_dataframe(CSV_LR_PATH)

    # Conversion en numpy (FIX PRINCIPAL)
    epoch_hr = df_hr['epoch'].to_numpy()
    map50_hr = df_hr['metrics/mAP50(B)'].to_numpy()
    
    epoch_lr = df_lr['epoch'].to_numpy()
    map50_lr = df_lr['metrics/mAP50(B)'].to_numpy()

    # ---------------------------------------------------------
    # GRAPHIQUE 1 : Evolution du mAP@50
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    
    plt.plot(epoch_hr, map50_hr, 
             label='Baseline (YOLOv8-Large @ 1024px)', color='#e74c3c', linewidth=2.5)
    
    plt.plot(epoch_lr, map50_lr, 
             label='Embedded (YOLOv8-Small @ 640px)', color='#2980b9', linewidth=2.5)

    plt.title("Convergence de la Precision durant l'Entrainement", fontweight='bold', pad=15)
    plt.xlabel("Epoques")
    plt.ylabel("mAP@50")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    
    # Marquer l'arret
    plt.scatter(epoch_hr[-1], map50_hr[-1], color='#e74c3c', s=100, zorder=5)
    plt.scatter(epoch_lr[-1], map50_lr[-1], color='#2980b9', s=100, zorder=5)

    plt.tight_layout()
    plt.savefig("training_map_convergence.png", dpi=300)
    print("Graphique 1 genere : training_map_convergence.png")

    # ---------------------------------------------------------
    # GRAPHIQUE 2 : Evolution de la perte
    # ---------------------------------------------------------
    loss_hr = df_hr['val/box_loss'].to_numpy()
    loss_lr = df_lr['val/box_loss'].to_numpy()

    plt.figure(figsize=(10, 6))
    
    plt.plot(epoch_hr, loss_hr, 
             label='Baseline (YOLOv8-Large @ 1024px)', color='#e74c3c', linewidth=2)
    
    plt.plot(epoch_lr, loss_lr, 
             label='Embedded (YOLOv8-Small @ 640px)', color='#2980b9', linewidth=2)

    plt.title("Diminution de l'Erreur de Localisation (Validation Box Loss)", fontweight='bold', pad=15)
    plt.xlabel("Epoques")
    plt.ylabel("Box Loss (Validation)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("training_loss_convergence.png", dpi=300)
    print("Graphique 2 genere : training_loss_convergence.png")

if __name__ == "__main__":
    if CSV_HR_PATH.exists() and CSV_LR_PATH.exists():
        generate_training_plots()
    else:
        print("[ERREUR] Les chemins vers les fichiers CSV sont introuvables. Veuillez les corriger dans le script.")