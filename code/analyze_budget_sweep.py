# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================
# CONFIGURATION
# ==============================
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Chemin vers les resultats de votre Sweep
SWEEP_DIR = Path("/linux/antoimartin/v2/code/AL_Sweep_Budgets")

# Les budgets que vous avez testes
BUDGETS = [10, 30, 60, 90, 120, 180, 240, 300, 420, 600]
STRATEGIES = ["ceal", "lightweight", "random"]

# Baseline du modele avant Active Learning (Mission 0)
BASELINE_MAP = 0.4310

def load_sweep_data():
    """
    Parcourt les dossiers et extrait le mAP final (Mission 3) 
    pour chaque combinaison de strategie et de budget.
    """
    results = []

    for strat in STRATEGIES:
        for b in BUDGETS:
            folder_path = SWEEP_DIR / f"{strat}_{b}"
            csv_path = folder_path / "summary_metrics.csv"
            
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    
                    if 3 in df['Mission'].values:
                        m3_data = df[df['Mission'] == 3].iloc[0]
                        
                        results.append({
                            "Strategy": strat.upper(),
                            "Budget": b,
                            "mAP50": m3_data['mAP50_mean'],
                            "mAP50_std": m3_data['mAP50_std'],
                            "Temps_Annotation_min": m3_data.get('Temps_Annotation_min', 0.0),
                            "Images_Transmises": m3_data.get('Images_Transmises_mean', 0.0)
                        })
                except Exception as e:
                    print(f"Erreur de lecture pour {strat} a budget {b} : {e}")

    return pd.DataFrame(results)

def print_results_for_latex(df):
    """
    Affiche les resultats dans la console de maniere propre 
    pour que vous puissiez me les copier-coller.
    """
    print("\n" + "="*70)
    print("RESULTATS DE L'EXPERIENCE SWEEP (mAP FINAL A LA MISSION 3)")
    print("="*70)
    
    pivot_map = df.pivot(index="Budget", columns="Strategy", values="mAP50")
    print("\n--- Evolution du mAP@50 en fonction du Budget ---")
    print(pivot_map.to_string())

    print("\n" + "-"*70)
    
    if 'Temps_Annotation_min' in df.columns and 'Images_Transmises' in df.columns:
        pivot_cost = df[df['Strategy'] == 'CEAL'].set_index('Budget')[['Images_Transmises', 'Temps_Annotation_min']]
        print("\n--- Efficacite du CEAL (Top Efficiency) ---")
        print("Budget Alloue (min) | Images Transmises | Temps Reel Utilise (min)")
        for index, row in pivot_cost.iterrows():
            print(f"{index:<19} | {row['Images_Transmises']:<17.1f} | {row['Temps_Annotation_min']:.1f}")
    
    print("="*70 + "\n")

def plot_sweet_spot(df):
    """
    Genere le graphique central du Sweep : l'evolution de la performance
    en fonction de l'augmentation du budget.
    """
    if df.empty:
        print("[Erreur] Aucune donnee trouvee pour le graphique.")
        return

    plt.figure(figsize=(12, 7))
    
    plt.axhline(y=BASELINE_MAP, color='black', linestyle=':', alpha=0.7, label=f"Baseline Initiale ({BASELINE_MAP:.4f})")

    styles = {
        "CEAL": {"color": "#27ae60", "marker": "o", "ls": "-", "label": "CEAL (Temps d'Annotation)"},
        "LIGHTWEIGHT": {"color": "#2980b9", "marker": "s", "ls": "-", "label": "Lightweight (Nombre d'Images)"},
        "RANDOM": {"color": "#e74c3c", "marker": "x", "ls": "--", "label": "Random (Temoin Aleatoire)"}
    }

    for strat in df['Strategy'].unique():
        subset = df[df['Strategy'] == strat].sort_values(by="Budget")
        st = styles.get(strat, {"color": "gray", "marker": ".", "ls": "-", "label": strat})
        
        plt.plot(subset['Budget'].to_numpy(), subset['mAP50'].to_numpy(),
                 label=st["label"], color=st["color"], 
                 marker=st["marker"], linestyle=st["ls"], linewidth=2.5, markersize=8)
        
        x = subset['Budget'].to_numpy()
        y = subset['mAP50'].to_numpy()
        y_std = subset['mAP50_std'].to_numpy()
        
        plt.fill_between(x, y - y_std, y + y_std,
                 color=st["color"], alpha=0.1)

    plt.title("Recherche du 'Sweet Spot' Operationnel\nEvolution de la Performance en fonction du Budget", fontweight='bold', pad=15)
    plt.xlabel("Budget Alloue par Mission\n(Minutes pour CEAL/Random  |  Nb d'Images pour Lightweight)")
    plt.ylabel("Precision Finale (mAP@50 a la Mission 3)")
    
    plt.xscale('log')
    plt.xticks(BUDGETS, labels=[str(b) for b in BUDGETS])
    
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("sweep_sweet_spot_comparison.png", dpi=300)
    print("Graphique sauvegarde : sweep_sweet_spot_comparison.png")

if __name__ == "__main__":
    df_results = load_sweep_data()
    print_results_for_latex(df_results)
    plot_sweet_spot(df_results)