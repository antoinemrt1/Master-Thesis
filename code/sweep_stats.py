import pandas as pd
from pathlib import Path

# --- CHEMIN A ADAPTER ---
SWEEP_DIR = Path("/linux/antoimartin/v2/code/AL_Sweep_Budgets")
BUDGETS = [10, 30, 60, 90, 120, 180, 240, 300, 420, 600]

def extract_experiment_stats():
    print("=== STATISTIQUES GLOBAL DE L'EXPERIENCE SWEEP ===\n")
    print("Structure de la simulation :")
    print(f"- Budget evalues : {len(BUDGETS)} paliers (de 10 a 600)")
    print(f"- Strategies testees : 3 (CEAL, Lightweight, Random)")
    print(f"- Missions successives par experience : 4 (Mission 0 a 3)")
    print(f"- Repetitions statistiques (Runs) : 2 (ou 3 selon vos logs)\n")

    # Analyse d'un fichier au hasard pour avoir le volume global
    sample_file = SWEEP_DIR / "ceal_600" / "summary_metrics.csv"
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        nb_missions = len(df['Mission'].unique()) - 1
        print(f"-> Soit un total de {len(BUDGETS) * 3 * nb_missions * 2} sessions d'entrainement (Fine-Tuning) realisees pour cette analyse de Sweep.\n")

if __name__ == "__main__":
    extract_experiment_stats()