import pandas as pd
from pathlib import Path

# Chemins
EXP_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_CEAL")

def generate_stats_report():
    report_path = "experiment_stats_report.txt"
    with open(report_path, "w") as f:
        f.write("=== RAPPORT STATISTIQUE DES EXPERIENCES CEAL ===\n\n")
        
        # Analysons par exemple le Top Efficiency 30min vs Random 30min (Run 1)
        for strat in ["CEAL_top_efficiency_30min", "CEAL_random_30min"]:
            run_dir = EXP_DIR / strat / "run_1"
            if not run_dir.exists(): 
                continue
            
            images_dir = run_dir / "images"
            labels_dir = run_dir / "labels"
            
            num_images = len(list(images_dir.glob("*.jpg")))
            total_boxes = 0
            
            for lbl in labels_dir.glob("*.txt"):
                with open(lbl, 'r') as txt:
                    total_boxes += len(txt.readlines())
                    
            cost_sec = (num_images * 2.0) + (total_boxes * 3.5)
            
            f.write(f"--- Strategie : {strat} (Cumul a la derniere mission) ---\n")
            f.write(f"Images annotees au total : {num_images}\n")
            f.write(f"Boites tracees au total  : {total_boxes}\n")
            f.write(f"Densite moyenne          : {total_boxes/num_images:.2f} avions/image\n")
            f.write(f"Cout humain total simule : {cost_sec:.1f} secondes ({cost_sec/60:.1f} minutes)\n\n")
            
    print(f"Rapport genere : {report_path}")

if __name__ == "__main__":
    generate_stats_report()