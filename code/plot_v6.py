# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

# Configuration esthetique
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_final_al_results_v6():
    # ==========================================
    # 1. DONNEES EN DUR (Issues des logs V6)
    # ==========================================
    missions = np.array([0, 1, 2, 3])

    # TOP 10%
    top10_mean = np.array([0.366416, 0.365924, 0.374508, 0.372799])
    top10_std  = np.array([0.000000, 0.004389, 0.000915, 0.001783])

    # RANDOM 10%
    rnd10_mean = np.array([0.366416, 0.367306, 0.367840, 0.368363])
    rnd10_std  = np.array([0.000000, 0.002204, 0.003855, 0.003451])

    # BOTTOM 10%
    bot10_mean = np.array([0.366416, 0.356123, 0.345484, 0.329252])
    bot10_std  = np.array([0.000000, 0.002525, 0.000818, 0.003409])

    # TOP 20%
    top20_mean = np.array([0.366416, 0.369659, 0.376811, 0.377746])
    top20_std  = np.array([0.000000, 0.002005, 0.002248, 0.000692])

    # ==========================================
    # 2. GENERATION DU GRAPHIQUE
    # ==========================================
    plt.figure(figsize=(11, 7))

    # Definition des courbes
    plots = [
        ("TOP 20% (Uncertainty+Rarity)", top20_mean, top20_std, "#27ae60", "-", "s"),
        ("TOP 10% (Uncertainty+Rarity)", top10_mean, top10_std, "#2ecc71", "--", "o"),
        ("RANDOM 10% (Random)",         rnd10_mean, rnd10_std, "#f39c12", "-", "D"),
        ("BOTTOM 10% (High confidence)", bot10_mean, bot10_std, "#e74c3c", "-", "X")
    ]

    for label, mean, std, color, ls, marker in plots:
        plt.errorbar(
            missions,
            mean,
            yerr=std,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=2.5,
            marker=marker,
            markersize=8,
            capsize=5,
            capthick=1.5
        )

    # Baseline
    plt.axhline(
        y=0.366416,
        color='black',
        linestyle=':',
        alpha=0.5,
        label="Initial Baseline (0.3664)"
    )

    # Format
    plt.title(
        "Performance Evolution with Active Learning\n(FAIR1M Dataset - Mean over 3 Runs)",
        fontweight='bold',
        pad=15
    )
    plt.xlabel("Missions and Fine-Tuning Steps")
    plt.ylabel("Precision (mAP@50) on Test Set")
    plt.xticks(missions)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=True, shadow=True)
    plt.grid(True)

    plt.tight_layout()

    output_filename = "al_strategies_comparison_v6.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_final_al_results_v6()