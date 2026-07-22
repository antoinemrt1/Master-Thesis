# multires_common.py
# -*- coding: utf-8 -*-
"""
Module partage pour les experimentations MULTI-RESOLUTION (Scenario 1).

Contenu :
  1. Chemins des 3 datasets (R0 / R1=Split4 / R2=Split16) et mapping deterministe
  2. Score CEAL par tuile (copie exacte du pipeline champion run_night_full_dataset.py)
  3. Comptabilite bande passante : octets JPEG reellement transmis
     (meme qualite JPEG pour toutes les resolutions -> comparaison honnete)
  4. Cout d'annotation humain base sur la Ground Truth des unites transmises
  5. Geometrie des patches : reprojection des boites R1/R2 vers le referentiel R0
  6. NMS classe-agnostique (fusion variante 1b)
  7. Evaluation mAP@50 custom avec decoupage par taille d'objet (small/medium/large)
  8. Equilibrage du sampler par sur-echantillonnage (variante 1a)
"""
import json
import os
import shutil

import cv2
import numpy as np
from pathlib import Path

# ==============================
# 1. CHEMINS & CONSTANTES
# ==============================
BASE_DIR = Path("/linux/antoimartin/v2").resolve()

R0_DATASET = BASE_DIR / "dataset"          # images 1000x1000 (tuiles unitaires)
R1_DATASET = BASE_DIR / "dataset_split4"   # 4 patches 500x500 -> 1024x1024 (zoom x2)
R2_DATASET = BASE_DIR / "dataset_split16"  # 16 patches 250x250 -> 1024x1024 (zoom x4)

DATASET_BY_RES = {"r0": R0_DATASET, "r1": R1_DATASET, "r2": R2_DATASET}
GRID_BY_RES = {"r1": 2, "r2": 4}
PATCH_SIZE = 1024  # taille cible des patches apres interpolation cubique

MAPPING_JSON = BASE_DIR / "multires_mappings.json"
MAPPING_SUBFOLDERS = ["Train_Init_Stratified", "Unlabeled_Pool_Stratified", "Val", "Test"]

DATA_YAML = BASE_DIR / "data.yaml"  # evaluation officielle sur le Test Set R0

CLASS_NAMES = [
    "Passenger Ship", "Motorboat", "Fishing Boat", "Tugboat", "other-ship",
    "Engineering Ship", "Liquid Cargo Ship", "Dry Cargo Ship", "Warship",
    "Small Car", "Bus", "Cargo Truck", "Dump Truck", "other-vehicle", "Van",
    "Trailer", "Tractor", "Excavator", "Truck Tractor", "Boeing737", "Boeing747",
    "Boeing777", "Boeing787", "ARJ21", "C919", "A220", "A321", "A330", "A350",
    "other-airplane", "Baseball Field", "Basketball Court", "Football Field",
    "Tennis Court", "Roundabout", "Intersection", "Bridge"
]

# Modele de cout humain (identique au CEAL existant)
TIME_PER_IMAGE_SEC = 2.0
TIME_PER_BBOX_SEC = 3.5

JPEG_QUALITY_DEFAULT = 85

# Bornes COCO de taille d'objet (en pixels^2, referentiel R0)
SIZE_BINS = {
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, 1e12),
}


def resolve_subdir(dataset_root, kind, subfolder):
    """
    Resout un sous-dossier images/labels en gerant les differences de casse
    (le dataset R0 utilise 'val'/'test', les splits utilisent 'Val'/'Test').
    """
    for name in (subfolder, subfolder.lower(), subfolder.capitalize(), subfolder.upper()):
        p = dataset_root / kind / name
        if p.exists():
            return p
    return dataset_root / kind / subfolder


# ==============================
# MAPPING R0 -> R1 / R2
# ==============================
def expected_patch_stems(stem, grid):
    """Nommage deterministe issu de create_super_res_dataset.py : {stem}_{i}_{j}."""
    return [f"{stem}_{i}_{j}" for i in range(grid) for j in range(grid)]


def load_mapping(subfolder):
    """
    Charge le mapping r0_stem -> {"r1": [...], "r2": [...]} pour un subfolder.
    Le JSON doit avoir ete genere par build_multires_mapping.py ; a defaut,
    il est reconstruit a la volee (et non sauvegarde, pour eviter les courses
    entre plusieurs runs paralleles).
    """
    if MAPPING_JSON.exists():
        with open(MAPPING_JSON) as f:
            full = json.load(f)
        if subfolder in full:
            return full[subfolder]
        for key in full:
            if key.lower() == subfolder.lower():
                return full[key]

    print(f"[WARN] Mapping absent pour '{subfolder}' -> reconstruction a la volee. "
          f"Lancez build_multires_mapping.py pour le materialiser.")
    r0_img_dir = resolve_subdir(R0_DATASET, "images", subfolder)
    r1_present = {p.stem for p in resolve_subdir(R1_DATASET, "images", subfolder).glob("*.jpg")}
    r2_present = {p.stem for p in resolve_subdir(R2_DATASET, "images", subfolder).glob("*.jpg")}

    mapping = {}
    for p in r0_img_dir.glob("*.jpg"):
        mapping[p.stem] = {
            "r1": [s for s in expected_patch_stems(p.stem, 2) if s in r1_present],
            "r2": [s for s in expected_patch_stems(p.stem, 4) if s in r2_present],
        }
    return mapping


def grid_cells(W, H, grid):
    """
    Coordonnees absolues des cellules de la grille, IDENTIQUES a la geometrie
    de create_super_res_dataset.py (division entiere + remainder sur la derniere
    ligne/colonne). Retourne {(i, j): (x1, y1, x2, y2)}.
    """
    step_h, step_w = H // grid, W // grid
    cells = {}
    for i in range(grid):
        for j in range(grid):
            y1 = i * step_h
            y2 = (i + 1) * step_h if i < grid - 1 else H
            x1 = j * step_w
            x2 = (j + 1) * step_w if j < grid - 1 else W
            cells[(i, j)] = (x1, y1, x2, y2)
    return cells


def parse_patch_indices(patch_stem):
    """'{stem}_{i}_{j}' -> (i, j). On parse depuis la droite car stem peut contenir des '_'."""
    parts = patch_stem.rsplit("_", 2)
    return int(parts[1]), int(parts[2])


# ==============================
# 2. SCORE CEAL (identique a l'existant)
# ==============================
def compute_tile_score_and_cost(results, class_counts, conf_threshold=0.15):
    """Copie exacte de run_night_full_dataset.py (le champion CEAL actuel)."""
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return 0.0, 0.0

    total_score = 0.0
    valid_boxes = 0

    for box in results[0].boxes:
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue

        cls_id = int(box.cls.item())

        p_best = conf
        p_second = conf * 0.8
        U = 1.0 - (p_best - p_second)

        Nc = class_counts.get(cls_id, 0)
        Wc = 1.0 / np.log(np.e + Nc)

        total_score += U * Wc
        valid_boxes += 1

    if valid_boxes == 0:
        return 0.0, 0.0

    annotation_cost = TIME_PER_IMAGE_SEC + (valid_boxes * TIME_PER_BBOX_SEC)
    return total_score, annotation_cost


# ==============================
# 3. COMPTABILITE BANDE PASSANTE
# ==============================
def encoded_jpeg_bytes(img_path, quality=JPEG_QUALITY_DEFAULT):
    """
    Taille en octets de l'image re-encodee en JPEG a qualite fixe.
    On re-encode TOUTES les resolutions a la meme qualite pour que la
    comparaison R0 vs R1 vs R2 soit honnete (les fichiers sur disque ont pu
    etre ecrits avec des parametres differents).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return int(len(buf)) if ok else 0


class BandwidthLedger:
    """Compteur cumulatif d'octets transmis, ventile par resolution."""

    def __init__(self):
        self.total_bytes = 0
        self.by_res = {"r0": 0, "r1": 0, "r2": 0}

    def add(self, res, n_bytes):
        self.total_bytes += int(n_bytes)
        self.by_res[res] = self.by_res.get(res, 0) + int(n_bytes)

    @property
    def total_mb(self):
        return self.total_bytes / 1e6

    def mb(self, res):
        return self.by_res.get(res, 0) / 1e6


# ==============================
# 4. GROUND TRUTH & COUT HUMAIN
# ==============================
def read_gt_boxes(lbl_path):
    """Lit un label YOLO -> liste [(cls, xc, yc, w, h)] normalises. [] si vide/absent."""
    boxes = []
    lbl_path = Path(lbl_path)
    if not lbl_path.exists():
        return boxes
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                c = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
                boxes.append((c, xc, yc, w, h))
    return boxes


def gt_annotation_cost_sec(lbl_path):
    """Cout humain reel d'une unite transmise : ouverture + detourage des boites GT."""
    n = len(read_gt_boxes(lbl_path))
    return TIME_PER_IMAGE_SEC + n * TIME_PER_BBOX_SEC


def update_class_counts_from_label(lbl_path, class_counts):
    """Met a jour l'historique {Nc} avec les annotations recues au sol."""
    for (c, *_rest) in read_gt_boxes(lbl_path):
        class_counts[c] = class_counts.get(c, 0) + 1


# ==============================
# 5. REPROJECTION R1/R2 -> R0
# ==============================
def reproject_patch_dets(dets, cell, patch_size=PATCH_SIZE):
    """
    Reprojette des detections faites sur un patch (coordonnees pixel du patch
    patch_size x patch_size) vers le referentiel R0.
    C'est la logique INVERSE du decoupage de create_super_res_dataset.py :
    le patch est un crop (x1,y1,x2,y2) de l'image R0, upscale a patch_size.

    dets : np.ndarray [N, 6] = (x1, y1, x2, y2, conf, cls)
    cell : (x1, y1, x2, y2) coordonnees du crop dans R0
    """
    if dets.size == 0:
        return dets.reshape(0, 6)
    cx1, cy1, cx2, cy2 = cell
    sx = (cx2 - cx1) / float(patch_size)
    sy = (cy2 - cy1) / float(patch_size)
    out = dets.copy().astype(np.float64)
    out[:, [0, 2]] = out[:, [0, 2]] * sx + cx1
    out[:, [1, 3]] = out[:, [1, 3]] * sy + cy1
    return out


# ==============================
# 6. NMS CLASSE-AGNOSTIQUE
# ==============================
def _iou_matrix(box, boxes):
    """IoU d'une boite [4] contre un lot [N,4] (format xyxy)."""
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def nms_class_agnostic(dets, iou_thr=0.5):
    """
    NMS classe-agnostique pour fusionner les detections du modele global (R0)
    et du specialiste (R1/R2 reprojetees). dets : np.ndarray [N,6].
    """
    if len(dets) == 0:
        return np.zeros((0, 6))
    dets = np.asarray(dets, dtype=np.float64)
    order = np.argsort(-dets[:, 4])
    dets = dets[order]
    keep = []
    suppressed = np.zeros(len(dets), dtype=bool)
    for idx in range(len(dets)):
        if suppressed[idx]:
            continue
        keep.append(idx)
        rest = np.arange(idx + 1, len(dets))
        rest = rest[~suppressed[rest]]
        if rest.size == 0:
            continue
        ious = _iou_matrix(dets[idx, :4], dets[rest, :4])
        suppressed[rest[ious >= iou_thr]] = True
    return dets[keep]


# ==============================
# 7. EVALUATION mAP@50 CUSTOM (+ tailles S/M/L)
# ==============================
def _ap_from_scores(scores, tps, n_gt):
    """AP par interpolation 'all-points' (enveloppe de precision)."""
    if n_gt == 0:
        return None
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    fp = 1.0 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for k in range(len(mpre) - 2, -1, -1):
        mpre[k] = max(mpre[k], mpre[k + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _map50_for_bin(dets_by_img, gts_by_img, num_classes, iou_thr=0.5, area_rng=None):
    """
    mAP@50 sur un sous-ensemble de tailles d'objets (area_rng en px^2 dans R0).
    Convention type COCO :
      - les GT hors du bin sont 'ignorees' : une detection qui les matche
        n'est ni TP ni FP,
      - une detection non matchee dont l'aire est hors du bin est ignoree,
      - les classes sans GT active sont exclues de la moyenne.
    area_rng=None -> evaluation globale standard.
    """
    aps = []
    for c in range(num_classes):
        scores, tps = [], []
        n_gt_active = 0

        for img_id, gts in gts_by_img.items():
            gt_c = gts[gts[:, 4] == c] if len(gts) else np.zeros((0, 5))
            areas = (gt_c[:, 2] - gt_c[:, 0]) * (gt_c[:, 3] - gt_c[:, 1]) if len(gt_c) else np.zeros(0)
            if area_rng is None:
                active = np.ones(len(gt_c), dtype=bool)
            else:
                active = (areas >= area_rng[0]) & (areas < area_rng[1])
            n_gt_active += int(active.sum())

            dets = dets_by_img.get(img_id, np.zeros((0, 6)))
            det_c = dets[dets[:, 5] == c] if len(dets) else np.zeros((0, 6))
            if len(det_c) == 0:
                continue
            det_c = det_c[np.argsort(-det_c[:, 4])]

            used = np.zeros(len(gt_c), dtype=bool)
            for d in det_c:
                if len(gt_c) == 0:
                    best_iou, best_k = 0.0, -1
                else:
                    ious = _iou_matrix(d[:4], gt_c[:, :4])
                    # 1) meilleur GT actif non utilise
                    cand = np.where(active & ~used)[0]
                    best_iou, best_k = 0.0, -1
                    if cand.size:
                        k = cand[np.argmax(ious[cand])]
                        if ious[k] >= iou_thr:
                            best_iou, best_k = ious[k], k
                    if best_k == -1:
                        # 2) sinon, matche-t-elle une GT ignoree ? -> detection ignoree
                        cand_ign = np.where(~active)[0]
                        if cand_ign.size and ious[cand_ign].max() >= iou_thr:
                            best_k = -2  # marqueur 'ignoree'

                if best_k >= 0:
                    used[best_k] = True
                    scores.append(d[4]); tps.append(1.0)
                elif best_k == -2:
                    continue  # detection ignoree (matche une GT hors bin)
                else:
                    if area_rng is not None:
                        d_area = (d[2] - d[0]) * (d[3] - d[1])
                        if not (area_rng[0] <= d_area < area_rng[1]):
                            continue  # FP hors du bin -> ignoree
                    scores.append(d[4]); tps.append(0.0)

        ap = _ap_from_scores(scores, tps, n_gt_active)
        if ap is not None:
            aps.append(ap)

    return float(np.mean(aps)) if aps else 0.0


def evaluate_detections(dets_by_img, gts_by_img, num_classes=len(CLASS_NAMES), iou_thr=0.5):
    """
    Evaluation complete : mAP@50 global + par taille d'objet.
    dets_by_img : {img_id: np.ndarray [N,6] (x1,y1,x2,y2,conf,cls)} referentiel R0
    gts_by_img  : {img_id: np.ndarray [M,5] (x1,y1,x2,y2,cls)}      referentiel R0
    """
    out = {"mAP50": _map50_for_bin(dets_by_img, gts_by_img, num_classes, iou_thr, None)}
    for name, rng in SIZE_BINS.items():
        out[f"mAP50_{name}"] = _map50_for_bin(dets_by_img, gts_by_img, num_classes, iou_thr, rng)
    return out


def load_gt_for_images(img_paths, lbl_dir):
    """
    Charge les GT (labels YOLO normalises) en boites absolues xyxy du
    referentiel R0. Necessite de lire la taille de chaque image.
    Retourne ({img_id: [M,5]}, {img_id: (W, H)}).
    """
    gts, dims = {}, {}
    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        dims[img_path.stem] = (W, H)
        rows = []
        for (c, xc, yc, w, h) in read_gt_boxes(Path(lbl_dir) / (img_path.stem + ".txt")):
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            rows.append([x1, y1, x2, y2, c])
        gts[img_path.stem] = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 5))
    return gts, dims


def results_to_dets(results):
    """Convertit un objet Results ultralytics en np.ndarray [N,6] (coords image originale)."""
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 6))
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float64)
    conf = boxes.conf.cpu().numpy().astype(np.float64).reshape(-1, 1)
    cls = boxes.cls.cpu().numpy().astype(np.float64).reshape(-1, 1)
    return np.hstack([xyxy, conf, cls])


# ==============================
# 8. EQUILIBRAGE DU SAMPLER (variante 1a)
# ==============================
def _link_or_copy(src, dst):
    """Symlink (rapide, zero disque) avec fallback copie si le FS refuse."""
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy(src, dst)


def build_train_view(exp_dir, view_name, master_images_dir, master_labels_dir,
                     balance_mode="none", balance_cap=10, restrict_res=None):
    """
    Construit une 'vue' d'entrainement (images + labels lies symboliquement)
    a partir du dataset cumulatif maitre. Les fichiers maitres sont prefixes
    'r0_' / 'r1_' / 'r2_' selon la resolution transmise.

    balance_mode :
      - "none"       : vue brute (le melange R0/R1/R2 tel quel)
      - "oversample" : sur-echantillonnage des resolutions minoritaires par
                       duplication (liens symboliques *_dupN), equivalent en
                       esperance a un weighted sampling inversement
                       proportionnel au nombre d'images par resolution.
                       Le facteur de duplication est plafonne a balance_cap.
    restrict_res : liste de prefixes a inclure (ex: ["r0"] ou ["r1", "r2"])
                   pour la variante 1b. None = toutes.

    Retourne (view_images_dir, counts_par_resolution).
    """
    view_dir = Path(exp_dir) / view_name
    if view_dir.exists():
        shutil.rmtree(view_dir)
    (view_dir / "images").mkdir(parents=True)
    (view_dir / "labels").mkdir(parents=True)

    files_by_res = {"r0": [], "r1": [], "r2": []}
    for img in sorted(Path(master_images_dir).glob("*.jpg")):
        res = img.name.split("_", 1)[0]
        if res not in files_by_res:
            continue
        if restrict_res is not None and res not in restrict_res:
            continue
        lbl = Path(master_labels_dir) / (img.stem + ".txt")
        files_by_res[res].append((img, lbl))

    counts = {res: len(v) for res, v in files_by_res.items()}

    # 1. Liens de base
    for res, files in files_by_res.items():
        for img, lbl in files:
            _link_or_copy(img, view_dir / "images" / img.name)
            if lbl.exists():
                _link_or_copy(lbl, view_dir / "labels" / lbl.name)

    # 2. Sur-echantillonnage des resolutions minoritaires
    if balance_mode == "oversample":
        present = {res: files for res, files in files_by_res.items() if files}
        if len(present) >= 2:
            target = max(len(f) for f in present.values())
            for res, files in present.items():
                n = len(files)
                n_total_wanted = min(target, n * balance_cap)
                extra = n_total_wanted - n
                k = 0
                while k < extra:
                    img, lbl = files[k % n]
                    dup_tag = f"_dup{k // n + 1}"
                    _link_or_copy(img, view_dir / "images" / f"{img.stem}{dup_tag}.jpg")
                    if lbl.exists():
                        _link_or_copy(lbl, view_dir / "labels" / f"{lbl.stem}{dup_tag}.txt")
                    k += 1

    return view_dir / "images", counts


def write_dataset_yaml(yaml_path, train_images_dir, val_dir, test_dir):
    """Ecrit un data.yaml pointant sur la vue d'entrainement (meme style que l'existant)."""
    with open(yaml_path, "w") as f:
        f.write(f"train: {train_images_dir}\n")
        f.write(f"val: {val_dir}\n")
        f.write(f"test: {test_dir}\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")
