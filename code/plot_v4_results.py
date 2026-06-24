import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

RESULTS_DIR = Path("AL_Experiments_V4")

def load_summary(strategy, percent):
    """Charge le fichier summary_metrics.csv d'une experience V4."""
    filepath = RESULTS_DIR / f"AL_{strategy}_{percent}pct" / "summary_metrics.csv"
    if filepath.exists():
        return pd.read_csv(filepath)
    else:
        print(f"Avertissement : Fichier non trouve -> {filepath}")
        return None

def plot_strategy_comparison():
    """Graphique 1 : Top vs Random vs Bottom (Budget fixe = 10%)"""
    df_top = load_summary("top", 10)
    df_rnd = load_summary("random", 10)
    df_bot = load_summary("bottom", 10)

    if df_top is None or df_rnd is None or df_bot is None:
        print("Graphique 1 annule (donnees manquantes).")
        return

    plt.figure(figsize=(10, 6))

    for df, label, color, marker in [
        (df_top, "Strategie TOP (Incertitude+Rarete)", "#27ae60", "o"),
        (df_rnd, "Strategie RANDOM (Aleatoire)", "#f39c12", "s"),
        (df_bot, "Strategie BOTTOM (Certitudes)", "#e74c3c", "X")
    ]:
        plt.errorbar(
            df['Mission'].to_numpy(),
            df['mAP50_mean'].to_numpy(),
            yerr=df['mAP50_std'].to_numpy(),
            label=label,
            color=color,
            marker=marker,
            linewidth=2.5,
            capsize=5,
            capthick=1.5
        )

    baseline_map = df_top['mAP50_mean'].iloc[0]
    plt.axhline(y=baseline_map, color='black', linestyle='--', alpha=0.5,
                label=f"Baseline Initiale ({baseline_map:.4f})")

    plt.title("Impact de la Strategie de Selection sur l'Apprentissage (Budget Fixe : 10%)", fontweight='bold', pad=15)
    plt.xlabel("Missions d'Active Learning")
    plt.ylabel("Precision (mAP@50) sur le Test Set")
    plt.xticks(df_top['Mission'].to_numpy())
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("strategy_comparison_10pct.png", dpi=300)
    print("Graphique sauvegarde : strategy_comparison_10pct.png")

def plot_budget_efficiency():
    """Graphique 2 : Efficacite de la bande passante (Top 5% vs Random 10%)"""
    df_top5 = load_summary("top", 5)
    df_top10 = load_summary("top", 10)
    df_rnd10 = load_summary("random", 10)

    if df_top5 is None or df_top10 is None or df_rnd10 is None:
        print("Graphique 2 annule (donnees manquantes).")
        return

    plt.figure(figsize=(10, 6))

    for df, label, color, linestyle in [
        (df_top10, "TOP 10% (Notre Algorithme)", "#27ae60", "-"),
        (df_top5,  "TOP 5% (Notre Algorithme)", "#2ecc71", "--"),
        (df_rnd10, "RANDOM 10% (Temoin)", "#f39c12", "-")
    ]:
        plt.plot(
            df['Mission'].to_numpy(),
            df['mAP50_mean'].to_numpy(),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker='o'
        )

    baseline_map = df_top10['mAP50_mean'].iloc[0]
    plt.axhline(y=baseline_map, color='black', linestyle='--', alpha=0.5)

    plt.annotate(
        "Efficacite Bande Passante:\nLe TOP 5% surpasse le RANDOM 10%",
        xy=(3, df_top5['mAP50_mean'].iloc[-1]),
        xytext=(1.5, 0.428),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

    plt.title("Efficacite de Transmission : Gain de mAP par Octet Transmis", fontweight='bold', pad=15)
    plt.xlabel("Missions d'Active Learning")
    plt.ylabel("Precision (mAP@50)")
    plt.xticks(df_top10['Mission'].to_numpy())
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig("budget_efficiency_comparison.png", dpi=300)
    print("Graphique sauvegarde : budget_efficiency_comparison.png")

if __name__ == "__main__":
    plot_strategy_comparison()
    plot_budget_efficiency()
    plt.show()