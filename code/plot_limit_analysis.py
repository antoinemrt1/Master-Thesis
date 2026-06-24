import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-whitegrid')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_long_term_airplanes():
    missions = np.arange(11)
    baseline = 0.431013
    
    # Donnees Top Efficiency (10 missions)
    top_mean = np.array([baseline, 0.4108, 0.4230, 0.4206, 0.4207, 0.4234, 0.4080, 0.4128, 0.4081, 0.3996, 0.3913])
    top_std = np.array([0, 0.0031, 0.0143, 0.0113, 0.0063, 0.0026, 0.0060, 0.0066, 0.0075, 0.0083, 0.0118])

    # Donnees Random (10 missions)
    rnd_mean = np.array([baseline, 0.4082, 0.4176, 0.4135, 0.4159, 0.4238, 0.4175, 0.4188, 0.4116, 0.4148, 0.4107])
    rnd_std = np.array([0, 0.0078, 0.0069, 0.0094, 0.0070, 0.0119, 0.0111, 0.0017, 0.0061, 0.0131, 0.0097])

    plt.figure(figsize=(10, 6))
    plt.axhline(y=baseline, color='black', linestyle=':', alpha=0.6, label="Baseline Initiale")
    
    plt.errorbar(missions, top_mean, yerr=top_std, label="TOP_EFFICIENCY (Derive Semantique)", color="#27ae60", marker="o", linewidth=2.5)
    plt.errorbar(missions, rnd_mean, yerr=rnd_std, label="RANDOM (Temoin)", color="#f39c12", marker="s", linewidth=2, linestyle="--")

    plt.annotate("Oubli Catastrophique continu\n(Le modele ne voit plus d'avions 'normaux')", 
                 xy=(9, 0.399), xytext=(5, 0.390),
                 arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=1.5),
                 bbox=dict(boxstyle="round", fc="white", ec="gray"))

    plt.title("Test d'Endurance : Les limites de l'Active Learning sur le long terme\n(Budget : 30 min/mission - 6 Classes Avions)", fontweight='bold')
    plt.xlabel("Missions d'Active Learning")
    plt.ylabel("Precision (mAP@50)")
    plt.xticks(missions)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("long_term_drift_plot.png", dpi=300)
    print("Genere : long_term_drift_plot.png")

def plot_scale_up_full():
    missions = np.arange(6)
    baseline = 0.366416
    
    top_mean = np.array([baseline, 0.3619, 0.3637, 0.3655, 0.3641, 0.3633])
    rnd_mean = np.array([baseline, 0.3596, 0.3643, 0.3628, 0.3621, 0.3592])

    plt.figure(figsize=(10, 6))
    plt.axhline(y=baseline, color='black', linestyle=':', alpha=0.6, label="Baseline Initiale")
    
    plt.plot(missions, top_mean, label="TOP_EFFICIENCY (120 min)", color="#27ae60", marker="o", linewidth=2.5)
    plt.plot(missions, rnd_mean, label="RANDOM (120 min)", color="#f39c12", marker="s", linewidth=2, linestyle="--")

    plt.title("Passage a l'Echelle (Scale-Up) sur 37 Classes\nL'effet 'Goutte d'eau' face a la complexite", fontweight='bold')
    plt.xlabel("Missions d'Active Learning")
    plt.ylabel("Precision (mAP@50)")
    plt.xticks(missions)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("scale_up_full_dataset.png", dpi=300)
    print("Genere : scale_up_full_dataset.png")

if __name__ == "__main__":
    plot_long_term_airplanes()
    plot_scale_up_full()