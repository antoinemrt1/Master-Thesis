import pandas as pd
from pathlib import Path
from collections import Counter

# Chemins vers les dossiers de labels du RUN 1 pour la MISSION 1
# On regarde ce que le systeme a selectionne lors du premier passage.
TOP_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_V6/AL_top_10pct/run_1/labels")
BOT_DIR = Path("/linux/antoimartin/v2/code/AL_Experiments_V6/AL_bottom_10pct/run_1/labels")

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

def analyze_selection(directory, name):
    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    txt_files = list(directory.glob("*.txt"))
    total_images = len(txt_files)
    total_boxes = 0
    class_counts = Counter()

    for txt in txt_files:
        with open(txt, 'r') as f:
            lines = f.readlines()
            total_boxes += len(lines)
            for line in lines:
                cls_id = int(line.split()[0])
                class_counts[cls_id] += 1

    avg_boxes = total_boxes / total_images if total_images > 0 else 0

    print(f"\n{'='*40}")
    print(f"SELECTION ANALYSIS: {name.upper()}")
    print(f"{'='*40}")
    print(f"Images analyzed : {total_images}")
    print(f"Total objects   : {total_boxes}")
    print(f"Average density : {avg_boxes:.1f} objects / image")
    
    print("\nTop 5 selected classes:")
    for cls_id, count in class_counts.most_common(5):
        print(f" - {CLASS_NAMES[cls_id]:<20} : {count} instances")

if __name__ == "__main__":
    analyze_selection(TOP_DIR, "Top strategy 10%")
    analyze_selection(BOT_DIR, "Bottom strategy 10%")