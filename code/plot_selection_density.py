import matplotlib.pyplot as plt
import numpy as np

# Configuration esthetique
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def plot_density_comparison():
    # ==========================================
    # 1. DONNEES EN DUR (Issues de votre script d'analyse)
    # ==========================================
    # Donnees pour la Strategie TOP 10%
    top_images = 1317
    top_objects = 159065
    top_density = 120.8
    
    # Top 5 classes du TOP
    top_classes = ['Small Car', 'Van', 'Dump Truck', 'Cargo Truck', 'Motorboat']
    top_counts = [68483, 62536, 10109, 4377, 3626]

    # Donnees pour la Strategie BOTTOM 10%
    bot_images = 1317
    bot_objects = 1442
    bot_density = 1.1
    
    # Top 5 classes du BOTTOM
    bot_classes = ['Dry Cargo Ship', 'Liquid Cargo Ship', 'other-airplane', 'Boeing737', 'A220']
    bot_counts = [407, 239, 203, 111, 104]

    # ==========================================
    # 2. GENERATION DU GRAPHIQUE (2 Sous-Graphes)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Graphique 1 : Comparaison de la Densite Globale ---
    strategies = ['Strategie BOTTOM (Certitudes)', 'Strategie TOP (Incertitude + Densite)']
    densities = [bot_density, top_density]
    colors = ['#e74c3c', '#27ae60']

    bars = ax1.bar(strategies, densities, color=colors, width=0.5, edgecolor='black')
    ax1.set_title("Densite d'Information par Tuile Transmise (Budget Fixe : 1317 tuiles)", fontweight='bold', pad=15)
    ax1.set_ylabel("Nombre moyen d'objets par tuile")
    
    # Ajouter la valeur sur les barres
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=14)

    # --- Graphique 2 : Composition des Classes (Top 5) ---
    x_top = np.arange(len(top_classes))
    ax2.bar(x_top, top_counts, color='#27ae60', alpha=0.7, edgecolor='black', label='TOP 10%')
    
    x_bot = np.arange(len(bot_classes))
    ax2.bar(x_bot, bot_counts, color='#e74c3c', alpha=0.9, edgecolor='black', hatch='//', label='BOTTOM 10%')

    ax2.set_yscale('log')
    ax2.set_title("Volume d'Instances Capturees (Top 5 Classes) - Echelle Logarithmique", fontweight='bold', pad=15)
    ax2.set_ylabel("Nombre total d'instances")
    ax2.set_xticks([])
    ax2.legend()
    
    # Textes explicatifs
    ax2.text(0, 100000, "Majorite : Petits vehicules (Scenes tres denses)", color='darkgreen', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    ax2.text(2, 500, "Majorite : Navires/avions isoles (Scenes triviales)", color='darkred', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()
    output_filename = "selection_density_comparison.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegarde sous : {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_density_comparison()