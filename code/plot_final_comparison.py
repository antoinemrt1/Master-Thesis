import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# --- ADAPTEZ CES CHEMINS APRES VOS RUNS ---
RESULTS_DIR = Path("AL_Final_Comparison")
CSV_LIGHTWEIGHT = RESULTS_DIR / "lightweight_200" / "summary_metrics.csv"
CSV_CEAL = RESULTS_DIR / "ceal_60" / "summary_metrics.csv"

def plot_final_comparison():
    if not CSV_LIGHTWEIGHT.exists() or not CSV_CEAL.exists():
        print("Fichiers CSV introuvables.")
        return

    df_lw = pd.read_csv(CSV_LIGHTWEIGHT)
    df_ceal = pd.read_csv(CSV_CEAL)

    baseline = df_lw['mAP50_mean'].iloc[0]
    missions = df_lw['Mission']

    # ==========================================
    # GRAPHIQUE 1 : Performances mAP
    # ==========================================
    plt.figure(figsize=(10, 6))
    plt.axhline(y=baseline, color='black', linestyle=':', alpha=0.6, label="Baseline Initiale")
    
    plt.errorbar(
        missions,
        df_lw['mAP50_mean'],
        yerr=df_lw['mAP50_std'],
        label="Strategie 1: Lightweight-Density (Budget: 200 images)",
        color="#e74c3c",
        marker="o",
        linewidth=2.5
    )
    
    plt.errorbar(
        missions,
        df_ceal['mAP50_mean'],
        yerr=df_ceal['mAP50_std'],
        label="Strategie 2: CEAL - Top Efficiency (Budget: 60 min)",
        color="#27ae60",
        marker="s",
        linewidth=2.5
    )

    plt.title("Comparaison des strategies sur le dataset complet (37 classes)", fontweight='bold')
    plt.xlabel("Missions d'Active Learning")
    plt.ylabel("Precision (mAP@50)")
    plt.xticks(missions)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("final_map_comparison.png", dpi=300)
    print("Genere : final_map_comparison.png")

    # ==========================================
    # GRAPHIQUE 2 : Efficacite (Bande passante & entrainement)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Somme sur les missions 1 a 4
    lw_imgs_total = df_lw['Images_Transmises_mean'][1:].sum()
    ceal_imgs_total = df_ceal['Images_Transmises_mean'][1:].sum()
    
    lw_train_total = df_lw['Temps_Entrainement_min'][1:].sum()
    ceal_train_total = df_ceal['Temps_Entrainement_min'][1:].sum()

    # Subplot 1 : Images transmises
    ax1.bar(
        ['Lightweight\n(Budget fixe images)', 'CEAL\n(Budget fixe temps)'],
        [lw_imgs_total, ceal_imgs_total],
        color=['#e74c3c', '#27ae60'],
        edgecolor='black'
    )
    ax1.set_title("Volume de transmission (bande passante)", fontweight='bold')
    ax1.set_ylabel("Nombre total d'images transmises")
    
    # Subplot 2 : Temps d'entrainement
    ax2.bar(
        ['Lightweight\n(Budget fixe images)', 'CEAL\n(Budget fixe temps)'],
        [lw_train_total, ceal_train_total],
        color=['#c0392b', '#2ecc71'],
        edgecolor='black'
    )
    ax2.set_title("Cout computationnel (fine-tuning)", fontweight='bold')
    ax2.set_ylabel("Temps total d'entrainement (minutes)")

    fig.suptitle(
        "Efficacite systeme : la strategie CEAL reduit l'utilisation des ressources",
        fontsize=16,
        fontweight='bold',
        y=1.05
    )
    plt.tight_layout()
    plt.savefig("final_resource_efficiency.png", dpi=300, bbox_inches='tight')
    print("Genere : final_resource_efficiency.png")

if __name__ == "__main__":
    plot_final_comparison()