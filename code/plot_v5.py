import matplotlib.pyplot as plt
import numpy as np

# Configuration esthetique (compatible toutes versions)
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_final_al_results():
    # ==========================================
    # 1. DONNEES EN DUR (Issues des logs V5)
    # ==========================================
    missions = np.array([0, 1, 2, 3])

    # TOP 10%
    top10_mean = np.array([0.349247, 0.352756, 0.359876, 0.359729])
    top10_std  = np.array([0.000000, 0.007343, 0.004419, 0.002018])

    # RANDOM 10%
    rnd10_mean = np.array([0.349247, 0.349971, 0.352975, 0.353499])
    rnd10_std  = np.array([0.000000, 0.004547, 0.003125, 0.004341])

    # BOTTOM 10%
    bot10_mean = np.array([0.349247, 0.327228, 0.321303, 0.318412])
    bot10_std  = np.array([0.000000, 0.004047, 0.004690, 0.001389])

    # TOP 20%
    top20_mean = np.array([0.349247, 0.357167, 0.363680, 0.364808])
    top20_std  = np.array([0.000000, 0.002153, 0.004665, 0.002377])

    # ==========================================
    # 2. GENERATION DU GRAPHIQUE
    # ==========================================
    plt.figure(figsize=(11, 7))

    plots = [
        ("TOP 20% (Incertitude+Rarete)", top20_mean, top20_std, "#27ae60", "-", "s"),
        ("TOP 10% (Incertitude+Rarete)", top10_mean, top10_std, "#2ecc71", "--", "o"),
        ("RANDOM 10% (Aleatoire)",       rnd10_mean, rnd10_std, "#f39c12", "-", "D"),
        ("BOTTOM 10% (Certitudes)",      bot10_mean, bot10_std, "#e74c3c", "-", "X")
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

    # Ligne de Baseline
    plt.axhline(
        y=0.349247,
        color='black',
        linestyle=':',
        alpha=0.5,
        label="Baseline Initiale (0.3492)"
    )

    # Formatage
    plt.title(
        "Evolution des Performances via Active Learning Inter-Missions\n(Moyenne sur 3 Runs Statistiques)",
        fontweight='bold',
        pad=15
    )
    plt.xlabel("Missions de Survol et Fine-Tuning")
    plt.ylabel("Precision (mAP@50) sur le Test Set")
    plt.xticks(missions)

    plt.legend(loc="upper left", frameon=True)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("al_strategies_comparison_v5.png", dpi=300)
    print("Graphique sauvegarde sous : al_strategies_comparison_v5.png")

if __name__ == "__main__":
    plot_final_al_results()