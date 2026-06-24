import matplotlib.pyplot as plt
import numpy as np

# Configuration esthetique
plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_ceal_efficiency():
    # ==========================================
    # 1. DONNEES EN DUR (Vos derniers resultats)
    # ==========================================
    
    # Axe X : Temps cumulatif d'annotation en minutes (Mission 0, 1, 2, 3)
    time_30m = np.array([0, 30, 60, 90])
    time_60m = np.array([0, 60, 120, 180])

    baseline = 0.431013

    # TOP EFFICIENCY
    top_30m_mean = np.array([baseline, 0.398164, 0.421878, 0.424593])
    top_30m_std  = np.array([0.000000, 0.010659, 0.007895, 0.012078])
    
    top_60m_mean = np.array([baseline, 0.426749, 0.431399, 0.422690])
    top_60m_std  = np.array([0.000000, 0.005095, 0.004114, 0.011767])

    # RANDOM
    rnd_30m_mean = np.array([baseline, 0.408761, 0.414756, 0.406211])
    rnd_30m_std  = np.array([0.000000, 0.003790, 0.009422, 0.016616])
    
    rnd_60m_mean = np.array([baseline, 0.420385, 0.416707, 0.424321])
    rnd_60m_std  = np.array([0.000000, 0.009392, 0.005353, 0.004680])

    # ==========================================
    # 2. GENERATION DU GRAPHIQUE
    # ==========================================
    plt.figure(figsize=(12, 7))

    # Ligne de Baseline
    plt.axhline(y=baseline, color='black', linestyle='--', alpha=0.7, label=f"Baseline Initiale ({baseline:.4f})")

    # Traces TOP EFFICIENCY
    plt.errorbar(time_30m, top_30m_mean, yerr=top_30m_std, label="TOP Efficiency (Budget 30m)", 
                 color="#27ae60", linestyle="-", linewidth=2.5, marker="o", markersize=8, capsize=5)
    plt.errorbar(time_60m, top_60m_mean, yerr=top_60m_std, label="TOP Efficiency (Budget 60m)", 
                 color="#2ecc71", linestyle="-", linewidth=2.5, marker="s", markersize=8, capsize=5)

    # Traces RANDOM
    plt.errorbar(time_30m, rnd_30m_mean, yerr=rnd_30m_std, label="RANDOM (Budget 30m)", 
                 color="#c0392b", linestyle="--", linewidth=2, marker="x", markersize=8, capsize=5)
    plt.errorbar(time_60m, rnd_60m_mean, yerr=rnd_60m_std, label="RANDOM (Budget 60m)", 
                 color="#e74c3c", linestyle="--", linewidth=2, marker="^", markersize=8, capsize=5)

    # Annotation du Depassement
    plt.annotate(
        "Depassement de la Baseline\n(Victoire AL !)", 
        xy=(120, top_60m_mean[2]), xytext=(100, 0.445),
        arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=2, headwidth=8),
        fontsize=11, color='darkgreen', weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
    )

    # Formatage
    plt.title("Cost-Effective Active Learning (CEAL)\nEvolution des performances en fonction du temps d'annotation humain cumule", fontweight='bold', pad=15)
    plt.xlabel("Temps cumulatif d'annotation humaine (Minutes)")
    plt.ylabel("Precision (mAP@50) sur le Test Set")
    
    plt.xticks(np.arange(0, 181, 30))
    plt.legend(loc="lower right", frameon=True, shadow=True)
    plt.grid(True, linestyle=":", alpha=0.7)

    plt.tight_layout()
    plt.savefig("ceal_time_efficiency_plot.png", dpi=300)
    print("Graphique sauvegarde sous : ceal_time_efficiency_plot.png")
    plt.show()

if __name__ == "__main__":
    plot_ceal_efficiency()