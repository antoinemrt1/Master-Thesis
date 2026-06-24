import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# CONFIGURATION GRAPHIQUE
# ==========================================
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def generate_pareto_chart():
    # --- DONNEES FINALES DU TEST SET ---
    data = {
        'Modele': ['Baseline (Serveur)', 'Embedded (Drone)'],
        'Configuration': ['YOLOv8-Large @ 1024px', 'YOLOv8-Small @ 640px'],
        'mAP50': [0.3987, 0.3438],
        'Latence_ms': [26.60, 5.32],
        'Type': ['Serveur (HR)', 'Embarque (LR)']
    }
    
    df = pd.DataFrame(data)

    # --- CREATION DE LA FIGURE ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Points du graphique (Scatter Plot)
    scatter = sns.scatterplot(
        data=df, 
        x='Latence_ms', 
        y='mAP50', 
        hue='Type', 
        style='Type',
        s=500,
        palette=['#e74c3c', '#2980b9'],
        ax=ax,
        zorder=3
    )

    # Annotation des points
    for i in range(df.shape[0]):
        ax.text(
            df['Latence_ms'][i], 
            df['mAP50'][i] + 0.005,
            df['Configuration'][i], 
            horizontalalignment='center', 
            size='medium', 
            color='black', 
            weight='bold',
            zorder=4
        )

    # --- ELEMENTS DE CONTEXTE VISUEL ---
    # Limite de la contrainte temps reel (100 ms)
    ax.axvline(x=100, color='gray', linestyle='--', linewidth=1.5, zorder=1)
    ax.text(102, 0.37, 'Contrainte Temps Reel\n(Limite 100 ms)', color='gray', fontstyle='italic')

    # Ligne reliant les deux points pour illustrer le "Front de Pareto"
    ax.plot(df['Latence_ms'].values, df['mAP50'].values, color='gray', linestyle=':', alpha=0.5, zorder=2)

    # Zone cible pour l'Active Learning (Objectif du TFE)
    ax.annotate(
        "Objectif du TFE (Active Learning) :\nRemonter la precision sans sacrifier la vitesse", 
        xy=(df['Latence_ms'][1], df['mAP50'][0]),
        xytext=(30, 0.38),
        arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.6, width=2, headwidth=8),
        fontsize=10,
        color='darkgreen',
        weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8)
    )
    
    # Ligne pointillee horizontale pour la target mAP
    ax.axhline(y=df['mAP50'][0], xmin=0, xmax=0.3, color='green', linestyle=':', alpha=0.5, zorder=1)
    # Ligne pointillee verticale depuis l'Embedded
    ax.plot([df['Latence_ms'][1], df['Latence_ms'][1]], [df['mAP50'][1], df['mAP50'][0]], 
            color='green', linestyle=':', alpha=0.5, zorder=1)

    # --- FORMATAGE DES AXES ---
    ax.set_title("Bilan des Modeles de Reference sur le Test Set (FAIR1M V2)", fontweight='bold', pad=20)
    ax.set_xlabel("Latence d'inference (ms / image) - Echelle Logarithmique")
    ax.set_ylabel("Precision de detection (mAP@50)")
    
    # Echelle logarithmique pour l'axe X
    ax.set_xscale('log')
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Limites pour aerer le graphique
    ax.set_ylim(0.33, 0.41)
    ax.set_xlim(3, 150)

    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(title="Environnement", loc='lower right')

    plt.tight_layout()
    output_filename = "baselines_pareto_testset.png"
    plt.savefig(output_filename, dpi=300)
    print(f"\nGraphique sauvegarde sous : {output_filename}")
    plt.show()

if __name__ == "__main__":
    generate_pareto_chart()