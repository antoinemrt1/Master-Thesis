# config.py
from pathlib import Path

# ==========================================
# CHEMINS DES DOSSIERS
# ==========================================

BASE_DIR = Path("/linux/antoimartin/v2")
DATASET_DIR = BASE_DIR / "dataset"

# Dossiers d'images et labels (Generes precedemment)
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
TRAIN_LBL_DIR = DATASET_DIR / "labels" / "train"

TEST_IMG_DIR = DATASET_DIR / "images" / "test"
TEST_LBL_DIR = DATASET_DIR / "labels" / "test"

# Dossiers pour l'experimentation Active Learning
EXPERIMENT_DIR = BASE_DIR / "AL_Experiments"
AL_DATASET_DIR = EXPERIMENT_DIR / "AL_Dataset"
MODELS_DIR = EXPERIMENT_DIR / "models"

# Fichier YAML pour l'evaluation sur le Test Set
TEST_YAML = BASE_DIR / "data.yaml"

# ==========================================
# PARAMETRES DES SCENARIOS (Cf. Tableau LaTeX)
# ==========================================
# Modeles initiaux (Golden Models entraines precedemment)
MODEL_LR_START = BASE_DIR / "code" / "trained_models" / "golden_yolov8s_640" / "weights" / "best.pt"

IMG_RES_EMBEDDED = 640     # Resolution UAV
INFERENCE_BUDGET_MS = 100  # Latence max toleree

# Parametres Active Learning
MISSIONS = 5               # Nombre de vols successifs
BUDGET_K_PER_MISSION = 200 # Nombre de tuiles (images) transmises par mission au sol
FINE_TUNE_EPOCHS = 20      # Epoques de re-entrainement entre les missions
BATCH_SIZE_TRAIN = 16      # Station Sol (GTX 1080 Ti)

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]