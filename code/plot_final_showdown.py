import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration esthetique
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 13, 'font.family': 'sans-serif'})

RESULTS_DIR = Path("AL_Final_Showdown")

def load_data(strategy):
    csv_path = RESULTS_DIR / f"{strategy}_120min" / "summary_metrics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print(f"[ERREUR] Fichier introuvable : {csv_path}")
        return None

def plot_showdown():
    df_ceal = load_data("top_efficiency")
    df_rnd = load_data("random")

    if df_ceal is None or df_rnd is None:
        return

    missions = df_ceal['Mission']
    baseline_map = df_ceal['mAP50_mean'].iloc[0]

    plt.figure(figsize=(11, 7))

    # Tracer les courbes (avec zone d'ombre pour l'ecart-type)
    for df, label, color, marker, ls in [
        (df_ceal, "Cost-Effective AL (Top Efficiency)", "#27ae60", "s", "-"),
        (df_rnd, "Selection Aleatoire (Random)", "#e74c3c", "o", "--")
    ]:
        plt.plot(missions, df['mAP50_mean'], label=label, color=color, 
                 linestyle=ls, linewidth=3, marker=marker, markersize=8)
        plt.fill_between(missions, 
                         df['mAP50_mean'] - df['mAP50_std'], 
                         df['mAP50_mean'] + df['mAP50_std'], 
                         color=color, alpha=0.15)

    # Ligne de Baseline
    plt.axhline(y=baseline_map, color='black', linestyle=':', linewidth=2, alpha=0.7, 
                label=f"Baseline Initiale ({baseline_map:.4f})")

    # Annotations pour guider la lecture
    plt.annotate(
        "Victoire du CEAL\nL'algorithme se detache nettement", 
        xy=(4, df_ceal['mAP50_mean'].iloc[4]), 
        xytext=(2.5, df_ceal['mAP50_mean'].iloc[4] + 0.005),
        arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=2, headwidth=8),
        fontsize=11, color='darkgreen', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
    )

    plt.title("Evaluation Finale au Point d'Optimum (Budget : 120 min / Mission)\nDataset Complet (37 Classes) - Moyenne sur 3 Runs", 
              fontweight='bold', pad=15)
    plt.xlabel("Missions d'Active Learning Successives")
    plt.ylabel("Precision (mAP@50) sur le Test Set")
    
    plt.xticks(missions)
    plt.legend(loc="lower right", frameon=True, shadow=True, fancybox=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_filename = "final_showdown_ceal_vs_random.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graphique final genere avec succes : {output_filename}")

if __name__ == "__main__":
    plot_showdown()