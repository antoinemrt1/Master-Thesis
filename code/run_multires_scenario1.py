# run_multires_scenario1.py
# -*- coding: utf-8 -*-
"""
SCENARIO 1 -- Allocation de resolution par tuile, en un seul passage.

Le drone survole le pool R0 (Unlabeled_Pool_Stratified), score chaque tuile
avec l'heuristique CEAL existante (Incertitude x Rarete x Densite, normalisee
par le cout d'annotation estime) et decide, tuile par tuile et SANS
aller-retour avec la base, a quelle resolution transmettre :

    efficacite < tau1          -> non transmis (poubelle)
    tau1 <= efficacite < tau2  -> transmis en R0 (1 image 1000x1000)
    efficacite >= tau2         -> transmis en R1 (les 4 patches Split4)
    --extreme_score            -> le top (100 - extreme_percentile)% passe
                                  en R2 (les 16 patches Split16)

Budget principal : VOLUME TRANSMIS EN Mo (JPEG a qualite fixe pour toutes les
resolutions). Le cout humain d'annotation reste logge en parallele.

Fine-tuning cote base :
    --variant 1a : un seul YOLO multi-echelle sur le melange R0+R1(+R2),
                   avec equilibrage optionnel du sampler (--balance_mode).
    --variant 1b : un modele 'global' (R0 seul) + un 'specialiste' (R1+R2) ;
                   evaluation par fusion NMS classe-agnostique apres
                   reprojection des boites dans le referentiel R0.

Baselines OBLIGATOIRES lancees dans le meme run (desactivables --no_baselines) :
    - ceal_r0   : CEAL classique (tout R0, tri par efficacite) au MEME budget Mo
    - random_r0 : selection aleatoire au MEME budget Mo

Exemples (machine externe multi-GPU) :
    python run_multires_scenario1.py --device 0 --variant 1a --auto_thresholds --q1 40 --q2 90
    python run_multires_scenario1.py --device 1 --variant 1b --auto_thresholds --q1 40 --q2 90 --extreme_score
    python run_multires_scenario1.py --device 2 --variant 1a --tau1 0.02 --tau2 0.08
"""
import argparse
import gc
import random
import shutil
import os
from datetime import datetime

# Limite la fragmentation memoire CUDA (doit etre fixe AVANT l'init de torch)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from ultralytics import YOLO

import matplotlib
matplotlib.use('Agg')

from multires_common import (
    BASE_DIR, R0_DATASET, R1_DATASET, R2_DATASET, DATASET_BY_RES,
    DATA_YAML, CLASS_NAMES, JPEG_QUALITY_DEFAULT,
    TIME_PER_IMAGE_SEC,
    resolve_subdir, load_mapping, grid_cells, parse_patch_indices,
    compute_tile_score_and_cost, encoded_jpeg_bytes, BandwidthLedger,
    gt_annotation_cost_sec, update_class_counts_from_label,
    reproject_patch_dets, nms_class_agnostic,
    evaluate_detections, load_gt_for_images, results_to_dets,
    build_train_view, write_dataset_yaml, _link_or_copy,
)

# ==============================
# CONFIGURATION
# ==============================
# Meme baseline que le champion CEAL actuel (run_night_full_dataset.py)
DEFAULT_BASE_MODEL = BASE_DIR / "code/trained_models/baseline_stratified_20pct_yolov8l_1024/weights/best.pt"

POOL_SUBFOLDER = "Unlabeled_Pool_Stratified"
R0_POOL_IMG_DIR = R0_DATASET / "images" / POOL_SUBFOLDER
R0_POOL_LBL_DIR = R0_DATASET / "labels" / POOL_SUBFOLDER
VAL_DIR = R0_DATASET / "images" / "val"
TEST_IMG_DIR = R0_DATASET / "images" / "test"
TEST_LBL_DIR = R0_DATASET / "labels" / "test"

RESULTS_ROOT_DEFAULT = BASE_DIR / "code" / "AL_Multires_S1"


# ==============================
# JOURNAL DE CAMPAGNE
# ==============================
# Toutes les etapes cles sont horodatees dans <campagne>/campaign.log ET
# affichees sur stdout (flush force, compatible nohup). Permet de suivre
# l'avancement d'un run long et de diagnostiquer un arret premature.
_LOG_PATH = {"path": None}


def init_logger(campaign_dir):
    _LOG_PATH["path"] = Path(campaign_dir) / "campaign.log"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if _LOG_PATH["path"] is not None:
        with open(_LOG_PATH["path"], "a") as f:
            f.write(line + "\n")


def disk_free_gb():
    return shutil.disk_usage(str(BASE_DIR)).free / 1e9


# ==============================
# BUDGETS REALISTES (grille bande passante x temps d'annotation)
# ==============================
# Le lien radio a un debit de --bw_kbps pendant la fenetre de la mission,
# assimilee au temps d'annotation --annot_min (grille des encadrants :
# {22, 88, 176} kbps x {3, 10, 30} min). Deux contraintes SIMULTANEES :
#   volume   <= bw_kbps * 1000/8 octets/s * annot_min * 60 s
#   annotation estimee <= annot_min
MIN_TX_BYTES = 30_000  # en dessous, plus aucune tuile scoree ne rentre -> arret du scan


def mission_budgets(args):
    volume_bytes = args.bw_kbps * 1000.0 / 8.0 * args.annot_min * 60.0
    annot_sec = args.annot_min * 60.0
    return volume_bytes, annot_sec


def estimated_cost_for_res(item, res):
    """
    Cout d'annotation ESTIME (depuis les predictions, comme le CEAL existant)
    d'une tuile transmise a la resolution res : memes boites, mais une
    ouverture d'image par patch (1, 4 ou 16).
    """
    n_units = {"r0": 1, "r1": 4, "r2": 16}[res]
    return item["cost"] + (n_units - 1) * TIME_PER_IMAGE_SEC


def measure_tx_bytes(item, res, pool_mapping, args):
    """
    Octets JPEG (qualite fixe) que couterait la transmission de la tuile a la
    resolution res. Resultat cache dans item pour etre reutilise par la
    transmission effective. Retourne None si les patches manquent.
    """
    cache = item.setdefault("tx_bytes", {})
    if res in cache:
        return cache[res]

    if res == "r0":
        total = encoded_jpeg_bytes(item["img"], args.jpeg_quality)
    else:
        patch_stems = pool_mapping.get(item["img"].stem, {}).get(res, [])
        expected = 4 if res == "r1" else 16
        if len(patch_stems) != expected:
            cache[res] = None
            return None
        src_img_dir = resolve_subdir(DATASET_BY_RES[res], "images", POOL_SUBFOLDER)
        total = sum(encoded_jpeg_bytes(src_img_dir / (ps + ".jpg"), args.jpeg_quality)
                    for ps in patch_stems)
    cache[res] = total
    return total


def select_budgeted(scored, pool_mapping, args):
    """
    Selection embarquee en mode budget realiste :
      1. les seuils (quantiles du batch ou valeurs absolues) fixent la
         resolution DESIREE de chaque tuile (logique du scenario 1) ;
      2. les tuiles sont parcourues par efficacite decroissante et transmises
         sous DOUBLE contrainte : octets restants ET minutes d'annotation
         estimees restantes ;
      3. si le groupe R1/R2 ne rentre plus dans les octets restants, la tuile
         est retrogradee (R2->R1->R0) pour continuer a transmettre de
         l'information (desactivable via --no_downgrade).
    Retourne (plan, tau1, tau2, tau_ext, stats) avec plan = [(item, res), ...].
    """
    volume_budget, annot_budget = mission_budgets(args)
    alloc, tau1, tau2, tau_ext = allocate_tiles(scored, args)

    desired = {}
    for res in ("r0", "r1", "r2"):
        for item in alloc[res]:
            desired[id(item)] = res

    plan = []
    vol_left = volume_budget
    ann_left = annot_budget
    n_downgraded = 0
    est_spent_sec = 0.0

    for item in sorted(scored, key=lambda x: x["eff"], reverse=True):
        want = desired.get(id(item))
        if want is None:  # zone poubelle (eff < tau1)
            continue
        if vol_left < MIN_TX_BYTES or ann_left < TIME_PER_IMAGE_SEC + 3.5:
            break  # plus rien ne rentre, inutile de scanner la queue

        chain = {"r2": ["r2", "r1", "r0"], "r1": ["r1", "r0"], "r0": ["r0"]}[want]
        if args.no_downgrade:
            chain = chain[:1]

        for res in chain:
            n_bytes = measure_tx_bytes(item, res, pool_mapping, args)
            if n_bytes is None:
                continue  # patches manquants -> resolution suivante
            cost = estimated_cost_for_res(item, res)
            if n_bytes <= vol_left and cost <= ann_left:
                plan.append((item, res))
                vol_left -= n_bytes
                ann_left -= cost
                est_spent_sec += cost
                if res != want:
                    n_downgraded += 1
                break

    stats = {
        "n_downgraded": n_downgraded,
        "est_annot_min": est_spent_sec / 60.0,
        "volume_budget_mb": volume_budget / 1e6,
        "vol_fill_pct": 100.0 * (volume_budget - vol_left) / max(volume_budget, 1e-9),
        "ann_fill_pct": 100.0 * (annot_budget - ann_left) / max(annot_budget, 1e-9),
    }
    return plan, tau1, tau2, tau_ext, stats


# ==============================
# SELECTION EMBARQUEE
# ==============================
def score_mission_tiles(model, mission_imgs, class_counts, args):
    """Inference R0 + score CEAL par tuile (identique au pipeline existant)."""
    scored = []
    for img in mission_imgs:
        res = model(str(img), imgsz=args.imgsz, device=args.device, verbose=False)
        score, cost = compute_tile_score_and_cost(res, class_counts,
                                                  conf_threshold=args.conf_threshold)
        if score > 0 and cost > 0:
            scored.append({"img": img, "score": score, "cost": cost,
                           "eff": score / cost})
    return scored


def allocate_tiles(scored, args):
    """
    Allocation a 3 seuils. Les percentiles (mode --auto_thresholds et
    --extreme_score) sont calcules sur le batch courant, uniquement sur les
    tuiles avec score > 0 (les tuiles sans detection sont toujours jetees,
    comme dans le CEAL existant).
    """
    effs = np.array([s["eff"] for s in scored])

    if args.auto_thresholds:
        tau1 = float(np.percentile(effs, args.q1))
        tau2 = float(np.percentile(effs, args.q2))
    else:
        tau1, tau2 = args.tau1, args.tau2

    tau_ext = float(np.percentile(effs, args.extreme_percentile)) if args.extreme_score else None

    alloc = {"trash": [], "r0": [], "r1": [], "r2": []}
    for s in scored:
        if s["eff"] < tau1:
            alloc["trash"].append(s)
        elif s["eff"] < tau2:
            alloc["r0"].append(s)
        elif tau_ext is not None and s["eff"] >= tau_ext:
            alloc["r2"].append(s)
        else:
            alloc["r1"].append(s)
    return alloc, tau1, tau2, tau_ext


def transmit_tile(item, res, pool_mapping, master_img_dir, master_lbl_dir,
                  ledger, class_counts, args):
    """
    'Transmet' une tuile a la resolution demandee :
      - reference image(s) via LIEN SYMBOLIQUE dans le dataset cumulatif
        maitre (prefixe r0_/r1_/r2_) -- zero duplication disque,
      - copie les labels GT (fichiers texte negligeables),
      - comptabilise les octets JPEG reellement transmis (reutilise la mesure
        deja faite par la selection budgetee si disponible),
      - comptabilise le cout humain GT et met a jour l'historique {Nc}.
    Fallback en R0 si les patches manquent dans le mapping.
    Retourne (res_effective, n_unites, octets, cout_humain_sec).
    """
    stem = item["img"].stem
    precomputed = item.get("tx_bytes", {})

    if res in ("r1", "r2"):
        patch_stems = pool_mapping.get(stem, {}).get(res, [])
        expected = 4 if res == "r1" else 16
        if len(patch_stems) != expected:
            log(f"[WARN] Patches {res} incomplets pour {stem} -> fallback R0")
            res = "r0"

    if res == "r0":
        src_img = item["img"]
        src_lbl = R0_POOL_LBL_DIR / (stem + ".txt")
        n_bytes = precomputed.get("r0")
        if n_bytes is None:
            n_bytes = encoded_jpeg_bytes(src_img, args.jpeg_quality)
        ledger.add("r0", n_bytes)
        _link_or_copy(src_img, master_img_dir / f"r0_{stem}.jpg")
        cost = gt_annotation_cost_sec(src_lbl)
        if src_lbl.exists():
            shutil.copy(src_lbl, master_lbl_dir / f"r0_{stem}.txt")
            update_class_counts_from_label(src_lbl, class_counts)
        return "r0", 1, n_bytes, cost

    src_img_dir = resolve_subdir(DATASET_BY_RES[res], "images", POOL_SUBFOLDER)
    src_lbl_dir = resolve_subdir(DATASET_BY_RES[res], "labels", POOL_SUBFOLDER)

    total_bytes = precomputed.get(res)
    measured = total_bytes is not None
    if not measured:
        total_bytes = 0
    total_cost, n_units = 0.0, 0
    for ps in patch_stems:
        src_img = src_img_dir / (ps + ".jpg")
        src_lbl = src_lbl_dir / (ps + ".txt")
        if not measured:
            total_bytes += encoded_jpeg_bytes(src_img, args.jpeg_quality)
        _link_or_copy(src_img, master_img_dir / f"{res}_{ps}.jpg")
        total_cost += gt_annotation_cost_sec(src_lbl)
        if src_lbl.exists():
            shutil.copy(src_lbl, master_lbl_dir / f"{res}_{ps}.txt")
            update_class_counts_from_label(src_lbl, class_counts)
        n_units += 1
    ledger.add(res, total_bytes)
    return res, n_units, total_bytes, total_cost


# ==============================
# REPLAY BUFFER (option anti oubli catastrophique)
# ==============================
def pick_replay_set(args, run_id):
    """
    Tire une fois par run (seed dediee) l'ensemble Replay : des images de
    Train_Init_Stratified DEJA presentes a la station sol -> cout de bande
    passante et d'annotation NUL. Contrecarre l'oubli catastrophique observe
    quand le fine-tuning ne voit que quelques tuiles difficiles par mission.
    """
    if args.replay_n <= 0:
        return []
    src_img_dir = resolve_subdir(R0_DATASET, "images", "Train_Init_Stratified")
    pool = sorted(src_img_dir.glob("*.jpg"))
    if not pool:
        log(f"[WARN] Replay demande mais {src_img_dir} est vide")
        return []
    rng = random.Random(4242 + run_id)
    return rng.sample(pool, min(args.replay_n, len(pool)))


def apply_replay(images_dir, labels_dir, replay_picks):
    """Lie les images Replay (prefixe replay_) dans une vue d'entrainement. Idempotent."""
    if not replay_picks:
        return 0
    src_lbl_dir = resolve_subdir(R0_DATASET, "labels", "Train_Init_Stratified")
    n = 0
    for p in replay_picks:
        dst = Path(images_dir) / f"replay_{p.name}"
        if dst.exists():
            continue
        _link_or_copy(p, dst)
        lbl = src_lbl_dir / (p.stem + ".txt")
        if lbl.exists():
            _link_or_copy(lbl, Path(labels_dir) / f"replay_{p.stem}.txt")
        n += 1
    return n


# ==============================
# ENTRAINEMENT & EVALUATION
# ==============================
def fine_tune(model_path, data_yaml, exp_dir, name, args):
    """
    Fine-tuning avec les memes hyperparametres que le champion CEAL.
    En cas de CUDA OOM (GPU partage avec un autre process), retente avec un
    batch divise par deux (4 -> 2 -> 1) apres avoir vide le cache.
    """
    batch = args.batch
    last_err = None
    for _attempt in range(3):
        try:
            model = YOLO(str(model_path))
            model.train(
                data=str(data_yaml), epochs=args.epochs, patience=0, freeze=args.freeze,
                lr0=args.lr0, imgsz=args.imgsz, batch=batch, device=args.device,
                workers=0, project=str(exp_dir), name=name, exist_ok=True, val=False
            )
            last_err = None
            break
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            last_err = e
            batch = max(1, batch // 2)
            log(f"[OOM] CUDA out of memory pendant '{name}' -> retry avec batch={batch}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
    if last_err is not None:
        raise last_err

    # best.pt est un doublon de last.pt (val=False) : ~90 Mo economises par mission
    if not args.keep_weights:
        best = exp_dir / name / "weights" / "best.pt"
        if best.exists():
            best.unlink()
    return exp_dir / name / "weights" / "last.pt"


def cleanup_train_dir(model_path, exp_dir, args):
    """
    Supprime un dossier d'entrainement devenu inutile (la chaine de
    fine-tuning est passee au suivant). Ne touche jamais au modele de base :
    seulement les poids situes DANS le dossier du run courant.
    """
    if args.keep_weights or model_path is None:
        return
    p = Path(model_path)
    if str(exp_dir) not in str(p):
        return  # modele de base (hors du run) -> intouchable
    if p.name == "last.pt" and p.parent.name == "weights":
        shutil.rmtree(p.parent.parent, ignore_errors=True)


def official_val(model_path, exp_dir, name, args):
    """Evaluation officielle model.val() sur le Test Set (comparable aux resultats CEAL)."""
    model = YOLO(str(model_path))
    # batch explicite : le defaut de model.val (16) peut provoquer un OOM
    # sur un GPU partage ; args.batch (4) suffit largement.
    res = model.val(data=str(DATA_YAML), split="test", imgsz=args.imgsz,
                    device=args.device, project=str(exp_dir), name=name,
                    exist_ok=True, verbose=False, plots=False, batch=args.batch)
    return float(res.box.map50), float(res.box.map)


def infer_dets_r0(model_path, test_imgs, args):
    """Detections d'un modele sur les images R0 du test set (referentiel R0)."""
    model = YOLO(str(model_path))
    dets = {}
    for p in test_imgs:
        r = model(str(p), imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300,
                  device=args.device, verbose=False)
        dets[p.stem] = results_to_dets(r)
    return dets


def infer_dets_specialist(model_path, test_imgs, test_mapping, dims, args):
    """
    Detections du specialiste sur les 4 patches R1 de chaque image test,
    reprojetees dans le referentiel R0 (logique inverse du decoupage R0->R1).
    """
    model = YOLO(str(model_path))
    patch_img_dir = resolve_subdir(R1_DATASET, "images", "Test")
    dets = {}
    for p in test_imgs:
        if p.stem not in dims:
            continue  # image test illisible (deja ignoree au chargement des GT)
        W, H = dims[p.stem]
        cells = grid_cells(W, H, grid=2)
        rows = []
        for ps in test_mapping.get(p.stem, {}).get("r1", []):
            patch_path = patch_img_dir / (ps + ".jpg")
            if not patch_path.exists():
                continue
            r = model(str(patch_path), imgsz=args.imgsz, conf=0.001, iou=0.7,
                      max_det=300, device=args.device, verbose=False)
            i, j = parse_patch_indices(ps)
            rows.append(reproject_patch_dets(results_to_dets(r), cells[(i, j)]))
        dets[p.stem] = np.vstack(rows) if rows else np.zeros((0, 6))
    return dets


def custom_eval_single(model_path, eval_ctx, args):
    """mAP@50 custom (global + par taille S/M/L) pour un modele unique sur R0."""
    dets = infer_dets_r0(model_path, eval_ctx["test_imgs"], args)
    return evaluate_detections(dets, eval_ctx["gts"])


def custom_eval_fused(global_path, spec_path, eval_ctx, args):
    """
    Variante 1b : fusion des detections du modele global (R0) et du
    specialiste (patches R1 reprojetes) par NMS classe-agnostique.
    """
    dets_g = infer_dets_r0(global_path, eval_ctx["test_imgs"], args)
    dets_s = infer_dets_specialist(spec_path, eval_ctx["test_imgs"],
                                   eval_ctx["test_mapping"], eval_ctx["dims"], args)
    fused = {}
    for stem in dets_g:
        stacked = np.vstack([dets_g[stem], dets_s.get(stem, np.zeros((0, 6)))])
        fused[stem] = nms_class_agnostic(stacked, iou_thr=args.fusion_iou)
    return evaluate_detections(fused, eval_ctx["gts"])


# ==============================
# RUN MULTI-RESOLUTION (strategie principale)
# ==============================
def execute_multires_run(args, run_id, all_images, campaign_dir, eval_ctx, pool_mapping, record):
    tag = "multires"
    log(f"--- RUN {run_id} [{tag}] variante {args.variant} ---")
    exp_dir = campaign_dir / tag / f"run_{run_id}"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)

    master_img = exp_dir / "images"
    master_lbl = exp_dir / "labels"
    master_img.mkdir(parents=True)
    master_lbl.mkdir(parents=True)

    ledger = BandwidthLedger()
    class_counts = {}
    mb_per_mission = []
    human_cost_cum = 0.0
    pool_size = max(1, len(all_images) // args.missions)
    replay_picks = pick_replay_set(args, run_id)
    if replay_picks:
        log(f"[REPLAY] {len(replay_picks)} images Train_Init injectees a chaque fine-tuning (cout BP nul)")

    # Chaines de modeles
    drone_model_path = Path(args.base_model)      # modele embarque (scoring)
    model_1a_path = Path(args.base_model)         # variante 1a
    global_path = Path(args.base_model)           # variante 1b - global (R0)
    spec_path = Path(args.specialist_base_model)  # variante 1b - specialiste (R1/R2)

    # --- Mission 0 : evaluation baseline ---
    log(f"[RUN {run_id}] Evaluation initiale (Mission 0)...")
    map50, map5095 = official_val(args.base_model, exp_dir, "eval_init", args)
    log(f"[RUN {run_id}][M0] mAP50={map50:.4f} | eval custom en cours...")
    custom = {}
    if args.variant == "1b":
        custom = custom_eval_fused(global_path, spec_path, eval_ctx, args)
    elif not args.skip_size_eval and not args.size_eval_final_only:
        custom = custom_eval_single(args.base_model, eval_ctx, args)

    record(_make_row(args, tag, run_id, 0, map50, map5095, custom, ledger,
                     0.0, 0.0, human_cost_cum,
                     n_trash=0, n_r0=0, n_r1=0, n_r2=0, n_units=0,
                     tau1=np.nan, tau2=np.nan, tau_ext=np.nan))

    for m in range(args.missions):
        log(f"[RUN {run_id}][MISSION {m+1}] Scoring embarque de {pool_size} tuiles... "
            f"(disque libre : {disk_free_gb():.1f} Go)")
        mission_imgs = all_images[m * pool_size:(m + 1) * pool_size]
        drone = YOLO(str(drone_model_path))
        scored = score_mission_tiles(drone, mission_imgs, class_counts, args)

        if not scored:
            log("[WARN] Aucune tuile scoree sur cette mission.")
            mb_per_mission.append(0)
            continue

        # --- Selection : mode budget realiste OU mode seuils purs ---
        sel_stats = {"n_downgraded": 0, "est_annot_min": np.nan}
        if args.bw_kbps > 0:
            plan, tau1, tau2, tau_ext, sel_stats = select_budgeted(
                scored, pool_mapping, args)
            log(f"[SELECTION BUDGET] {len(plan)}/{len(scored)} tuiles | "
                f"volume rempli a {sel_stats['vol_fill_pct']:.0f}% | "
                f"annotation remplie a {sel_stats['ann_fill_pct']:.0f}% | "
                f"downgrades={sel_stats['n_downgraded']}")
        else:
            alloc, tau1, tau2, tau_ext = allocate_tiles(scored, args)
            plan = [(item, res) for res in ("r0", "r1", "r2") for item in alloc[res]]
        n_trash = len(scored) - len(plan)
        log(f"[ALLOCATION] tau1={tau1:.5f} tau2={tau2:.5f} "
            f"tau_ext={tau_ext if tau_ext is None else round(tau_ext, 5)} | "
            f"poubelle={n_trash} | plan={len(plan)} tuiles")

        # --- Transmission + comptabilite ---
        mission_bytes = 0
        mission_cost = 0.0
        n_units_tx = 0
        eff_counts = {"r0": 0, "r1": 0, "r2": 0}
        for item, res in plan:
            eff_res, n_units, n_bytes, cost = transmit_tile(
                item, res, pool_mapping, master_img, master_lbl,
                ledger, class_counts, args)
            eff_counts[eff_res] += 1
            mission_bytes += n_bytes
            mission_cost += cost
            n_units_tx += n_units

        mb_per_mission.append(mission_bytes)
        human_cost_cum += mission_cost
        log(f"[BANDE PASSANTE] Mission : {mission_bytes/1e6:.2f} Mo | "
            f"Cumul : {ledger.total_mb:.2f} Mo "
            f"(R0 {ledger.mb('r0'):.1f} / R1 {ledger.mb('r1'):.1f} / R2 {ledger.mb('r2'):.1f})")

        # --- Fine-tuning cote base ---
        custom = {}
        do_size_eval = (not args.skip_size_eval) and \
                       (not args.size_eval_final_only or m == args.missions - 1)
        n_master = sum(1 for _ in master_img.glob("*.jpg"))

        if args.variant == "1a":
            if n_master == 0:
                log("[WARN] Dataset cumulatif vide -> fine-tuning saute")
            else:
                view_imgs, counts = build_train_view(
                    exp_dir, "train_view", master_img, master_lbl,
                    balance_mode=args.balance_mode, balance_cap=args.balance_cap)
                n_rep = apply_replay(view_imgs, view_imgs.parent / "labels", replay_picks)
                log(f"[TRAIN 1a] Vue equilibree ({args.balance_mode}) : {counts} "
                    f"+ {n_rep} replay -> fine-tuning...")
                al_yaml = exp_dir / "al_dataset.yaml"
                write_dataset_yaml(al_yaml, view_imgs, VAL_DIR, TEST_IMG_DIR)
                prev_path = model_1a_path
                model_1a_path = fine_tune(model_1a_path, al_yaml, exp_dir, f"train_m{m+1}", args)
                cleanup_train_dir(prev_path, exp_dir, args)
                drone_model_path = model_1a_path

            map50, map5095 = official_val(model_1a_path, exp_dir, f"eval_m{m+1}", args)
            if do_size_eval:
                custom = custom_eval_single(model_1a_path, eval_ctx, args)

        else:  # variante 1b
            view_r0, counts_r0 = build_train_view(
                exp_dir, "train_view_r0", master_img, master_lbl,
                balance_mode="none", restrict_res=["r0"])
            view_hr, counts_hr = build_train_view(
                exp_dir, "train_view_hr", master_img, master_lbl,
                balance_mode=args.balance_mode, balance_cap=args.balance_cap,
                restrict_res=["r1", "r2"])
            log(f"[TRAIN 1b] Global R0 : {counts_r0} | Specialiste HR : {counts_hr} -> fine-tuning...")

            apply_replay(view_r0, view_r0.parent / "labels", replay_picks)
            if counts_r0["r0"] > 0 or replay_picks:
                yaml_r0 = exp_dir / "al_dataset_r0.yaml"
                write_dataset_yaml(yaml_r0, view_r0, VAL_DIR, TEST_IMG_DIR)
                prev_path = global_path
                global_path = fine_tune(global_path, yaml_r0, exp_dir, f"train_glob_m{m+1}", args)
                cleanup_train_dir(prev_path, exp_dir, args)
            if counts_hr["r1"] + counts_hr["r2"] > 0:
                yaml_hr = exp_dir / "al_dataset_hr.yaml"
                write_dataset_yaml(yaml_hr, view_hr, VAL_DIR, TEST_IMG_DIR)
                prev_path = spec_path
                spec_path = fine_tune(spec_path, yaml_hr, exp_dir, f"train_spec_m{m+1}", args)
                cleanup_train_dir(prev_path, exp_dir, args)

            drone_model_path = global_path
            # mAP officielle du global seul (reference) + fusion custom (headline 1b)
            map50, map5095 = official_val(global_path, exp_dir, f"eval_glob_m{m+1}", args)
            custom = custom_eval_fused(global_path, spec_path, eval_ctx, args)

        record(_make_row(args, tag, run_id, m + 1, map50, map5095, custom, ledger,
                         mission_bytes / 1e6, mission_cost / 60.0, human_cost_cum,
                         n_trash=n_trash, n_r0=eff_counts["r0"],
                         n_r1=eff_counts["r1"], n_r2=eff_counts["r2"],
                         n_units=n_units_tx, tau1=tau1, tau2=tau2,
                         tau_ext=np.nan if tau_ext is None else tau_ext,
                         est_annot_min=sel_stats.get("est_annot_min", np.nan),
                         n_downgraded=sel_stats.get("n_downgraded", 0)))
        if custom:
            log(f"[EVAL M{m+1}] mAP50={map50:.4f} | mAP50_custom={custom['mAP50']:.4f}")
        else:
            log(f"[EVAL M{m+1}] mAP50={map50:.4f}")

    # Nettoyage des derniers poids de la chaine (CSV = seule sortie utile)
    for p in {str(model_1a_path), str(global_path), str(spec_path)}:
        cleanup_train_dir(p, exp_dir, args)

    return mb_per_mission


# ==============================
# BASELINES R0 A BUDGET Mo APPARIE
# ==============================
def execute_r0_baseline_run(args, strategy, run_id, all_images, budgets_bytes,
                            campaign_dir, eval_ctx, record):
    """
    Baselines au MEME budget bande passante (par mission) que le run multires :
      - ceal_r0   : tri par efficacite decroissante (le champion actuel)
      - random_r0 : ordre aleatoire
    Remplissage glouton du budget en octets (une image trop grosse est sautee,
    on continue avec les suivantes -- meme logique que le budget temps du CEAL).
    """
    tag = f"baseline_{strategy}"
    log(f"--- RUN {run_id} [{tag}] budgets={[round(b/1e6, 1) for b in budgets_bytes]} Mo ---")
    exp_dir = campaign_dir / tag / f"run_{run_id}"
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    master_img = exp_dir / "images"
    master_lbl = exp_dir / "labels"
    master_img.mkdir(parents=True)
    master_lbl.mkdir(parents=True)

    al_yaml = exp_dir / "al_dataset.yaml"
    write_dataset_yaml(al_yaml, master_img, VAL_DIR, TEST_IMG_DIR)

    ledger = BandwidthLedger()
    class_counts = {}
    human_cost_cum = 0.0
    current_model_path = Path(args.base_model)
    pool_size = max(1, len(all_images) // args.missions)
    replay_picks = pick_replay_set(args, run_id)

    map50, map5095 = official_val(args.base_model, exp_dir, "eval_init", args)
    custom = {}
    if not args.skip_size_eval and not args.size_eval_final_only:
        custom = custom_eval_single(args.base_model, eval_ctx, args)
    record(_make_row(args, tag, run_id, 0, map50, map5095, custom, ledger,
                     0.0, 0.0, human_cost_cum,
                     n_trash=0, n_r0=0, n_r1=0, n_r2=0, n_units=0,
                     tau1=np.nan, tau2=np.nan, tau_ext=np.nan))

    for m in range(args.missions):
        budget = budgets_bytes[m] if m < len(budgets_bytes) else 0
        log(f"[RUN {run_id}][{tag} M{m+1}] Scoring de {pool_size} tuiles...")
        mission_imgs = all_images[m * pool_size:(m + 1) * pool_size]
        model = YOLO(str(current_model_path))
        scored = score_mission_tiles(model, mission_imgs, class_counts, args)

        if strategy == "ceal_r0":
            scored.sort(key=lambda x: x["eff"], reverse=True)
        else:
            random.shuffle(scored)

        # Remplissage glouton : budget octets + (mode realiste) budget annotation
        ann_budget = args.annot_min * 60.0 if args.bw_kbps > 0 else float("inf")
        mission_bytes = 0
        mission_cost = 0.0
        ann_spent_est = 0.0
        n_selected = 0
        for item in scored:
            if budget - mission_bytes < MIN_TX_BYTES or \
               ann_budget - ann_spent_est < TIME_PER_IMAGE_SEC + 3.5:
                break
            if ann_spent_est + item["cost"] > ann_budget:
                continue
            n_bytes = encoded_jpeg_bytes(item["img"], args.jpeg_quality)
            if mission_bytes + n_bytes > budget:
                continue
            stem = item["img"].stem
            _link_or_copy(item["img"], master_img / f"r0_{stem}.jpg")
            src_lbl = R0_POOL_LBL_DIR / (stem + ".txt")
            mission_cost += gt_annotation_cost_sec(src_lbl)
            if src_lbl.exists():
                shutil.copy(src_lbl, master_lbl / f"r0_{stem}.txt")
                update_class_counts_from_label(src_lbl, class_counts)
            ledger.add("r0", n_bytes)
            mission_bytes += n_bytes
            ann_spent_est += item["cost"]
            n_selected += 1

        human_cost_cum += mission_cost
        log(f"[{tag} M{m+1}] {n_selected} images R0 | {mission_bytes/1e6:.2f} Mo "
            f"(budget {budget/1e6:.2f} Mo) -> fine-tuning...")

        apply_replay(master_img, master_lbl, replay_picks)
        if sum(1 for _ in master_img.glob("*.jpg")) > 0:
            prev_path = current_model_path
            current_model_path = fine_tune(current_model_path, al_yaml, exp_dir,
                                           f"train_m{m+1}", args)
            cleanup_train_dir(prev_path, exp_dir, args)
        else:
            log("[WARN] Dataset cumulatif vide -> fine-tuning saute")

        do_size_eval = (not args.skip_size_eval) and \
                       (not args.size_eval_final_only or m == args.missions - 1)
        map50, map5095 = official_val(current_model_path, exp_dir, f"eval_m{m+1}", args)
        custom = custom_eval_single(current_model_path, eval_ctx, args) if do_size_eval else {}
        record(_make_row(args, tag, run_id, m + 1, map50, map5095, custom, ledger,
                         mission_bytes / 1e6, mission_cost / 60.0, human_cost_cum,
                         n_trash=len(scored) - n_selected, n_r0=n_selected,
                         n_r1=0, n_r2=0, n_units=n_selected,
                         tau1=np.nan, tau2=np.nan, tau_ext=np.nan,
                         est_annot_min=ann_spent_est / 60.0 if args.bw_kbps > 0 else np.nan))
        log(f"[{tag} EVAL M{m+1}] mAP50={map50:.4f}")

    cleanup_train_dir(current_model_path, exp_dir, args)


# ==============================
# LOGGING
# ==============================
def _make_row(args, strategy, run_id, mission, map50, map5095, custom, ledger,
              mb_mission, cost_mission_min, cost_cum_sec,
              n_trash, n_r0, n_r1, n_r2, n_units, tau1, tau2, tau_ext,
              est_annot_min=np.nan, n_downgraded=0):
    budget_mode = args.bw_kbps > 0
    return {
        "Strategy": strategy,
        "Variant": args.variant,
        "Run": run_id,
        "Mission": mission,
        "BW_kbps": args.bw_kbps if budget_mode else np.nan,
        "Budget_annot_min": args.annot_min if budget_mode else np.nan,
        "Budget_MB_mission": (mission_budgets(args)[0] / 1e6) if budget_mode else np.nan,
        "Est_annot_min_mission": est_annot_min,
        "N_downgraded": n_downgraded,
        "N_replay": args.replay_n,
        "mAP50": map50,
        "mAP50_95": map5095,
        "mAP50_custom": custom.get("mAP50", np.nan),
        "mAP50_small": custom.get("mAP50_small", np.nan),
        "mAP50_medium": custom.get("mAP50_medium", np.nan),
        "mAP50_large": custom.get("mAP50_large", np.nan),
        "MB_mission": mb_mission,
        "MB_cum": ledger.total_mb,
        "MB_r0_cum": ledger.mb("r0"),
        "MB_r1_cum": ledger.mb("r1"),
        "MB_r2_cum": ledger.mb("r2"),
        "N_trash": n_trash,
        "N_tiles_r0": n_r0,
        "N_tiles_r1": n_r1,
        "N_tiles_r2": n_r2,
        "N_units_tx": n_units,
        "Tau1": tau1,
        "Tau2": tau2,
        "Tau_ext": tau_ext,
        "Human_min_mission": cost_mission_min,
        "Human_min_cum": cost_cum_sec / 60.0,
    }


# ==============================
# CAMPAGNE
# ==============================
def campaign_name(args):
    if args.auto_thresholds:
        thr = f"autoQ{args.q1:g}-{args.q2:g}"
    else:
        thr = f"tau{args.tau1:g}-{args.tau2:g}"
    name = f"S1_{args.variant}"
    if args.bw_kbps > 0:
        name += f"_BW{args.bw_kbps:g}k_T{args.annot_min:g}m"
    name += f"_{thr}"
    if args.replay_n > 0:
        name += f"_rep{args.replay_n}"
    if args.extreme_score:
        name += f"_ext{args.extreme_percentile:g}"
    if args.variant == "1a":
        name += f"_bal-{args.balance_mode}"
    if args.campaign_suffix:
        name += f"_{args.campaign_suffix}"
    return name


def run_campaign(args):
    results_root = Path(args.results_root)
    campaign_dir = results_root / campaign_name(args)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_SAVE_DIR"] = str(campaign_dir)
    init_logger(campaign_dir)

    log("=" * 60)
    log(f"CAMPAGNE SCENARIO 1 MULTI-RESOLUTION : {campaign_dir.name}")
    log("=" * 60)

    all_images_pool = sorted(R0_POOL_IMG_DIR.glob("*.jpg"))
    if not all_images_pool:
        log(f"[ERREUR] Pool vide : {R0_POOL_IMG_DIR}")
        return
    log(f"Pool R0 : {len(all_images_pool)} images | Missions : {args.missions} | "
        f"Runs : {args.runs} | pool_cap : {args.pool_cap}")

    # Mapping pool (genere par build_multires_mapping.py)
    pool_mapping = load_mapping(POOL_SUBFOLDER)

    # Contexte d'evaluation custom (GT chargees une seule fois)
    test_imgs = sorted(TEST_IMG_DIR.glob("*.jpg"))
    if args.test_subset > 0:
        test_imgs = test_imgs[:args.test_subset]
    need_custom = (args.variant == "1b") or (not args.skip_size_eval)
    eval_ctx = {"test_imgs": test_imgs, "gts": {}, "dims": {}, "test_mapping": {}}
    if need_custom:
        log(f"[EVAL] Chargement des GT du test set ({len(test_imgs)} images)...")
        gts, dims = load_gt_for_images(test_imgs, TEST_LBL_DIR)
        eval_ctx["gts"] = gts
        eval_ctx["dims"] = dims
        if args.variant == "1b":
            eval_ctx["test_mapping"] = load_mapping("Test")

    # Sauvegarde incrementale : le CSV est reecrit apres CHAQUE mission de
    # chaque strategie, pour pouvoir suivre/analyser une campagne en cours.
    all_rows = []

    def record(row):
        all_rows.append(row)
        _save_metrics(campaign_dir, all_rows)

    if args.bw_kbps > 0:
        vol_budget, _ = mission_budgets(args)
        log(f"[BUDGET REALISTE] {args.bw_kbps:g} kbps x {args.annot_min:g} min "
            f"-> {vol_budget/1e6:.2f} Mo/mission + {args.annot_min:g} min d'annotation/mission")

    for run_id in range(1, args.runs + 1):
        random.seed(42 + run_id)
        run_images = all_images_pool.copy()
        random.shuffle(run_images)
        if args.pool_cap > 0:
            run_images = run_images[:args.pool_cap]

        mb_per_mission = execute_multires_run(
            args, run_id, run_images, campaign_dir, eval_ctx, pool_mapping, record)

        if not args.no_baselines:
            # Mode realiste : budget exogene (grille) identique pour tous ;
            # mode legacy : budget apparie sur la conso reelle du run multires.
            if args.bw_kbps > 0:
                budgets = [mission_budgets(args)[0]] * args.missions
            else:
                budgets = mb_per_mission
            for strat in ("ceal_r0", "random_r0"):
                execute_r0_baseline_run(
                    args, strat, run_id, run_images, budgets,
                    campaign_dir, eval_ctx, record)

    _save_metrics(campaign_dir, all_rows, final=True)
    log("[SUCCES] Campagne terminee.")


def _save_metrics(campaign_dir, all_rows, final=False):
    df = pd.DataFrame(all_rows)
    df.to_csv(campaign_dir / "all_runs_raw_metrics.csv", index=False)
    summary = df.groupby(["Strategy", "Mission"]).agg(
        mAP50_mean=("mAP50", "mean"),
        mAP50_std=("mAP50", "std"),
        mAP50_custom_mean=("mAP50_custom", "mean"),
        mAP50_custom_std=("mAP50_custom", "std"),
        mAP50_small_mean=("mAP50_small", "mean"),
        mAP50_medium_mean=("mAP50_medium", "mean"),
        mAP50_large_mean=("mAP50_large", "mean"),
        MB_cum_mean=("MB_cum", "mean"),
        MB_cum_std=("MB_cum", "std"),
        N_tiles_r0_mean=("N_tiles_r0", "mean"),
        N_tiles_r1_mean=("N_tiles_r1", "mean"),
        N_tiles_r2_mean=("N_tiles_r2", "mean"),
        N_trash_mean=("N_trash", "mean"),
        Human_min_cum_mean=("Human_min_cum", "mean"),
    ).reset_index()
    summary.to_csv(campaign_dir / "summary_metrics.csv", index=False)
    if final:
        print("\n" + "=" * 60)
        print("RESUME FINAL")
        print("=" * 60)
        print(summary.to_string(index=False))


# ==============================
# MAIN
# ==============================
def build_parser():
    p = argparse.ArgumentParser(description="Scenario 1 : allocation de resolution par tuile (single-pass)")
    p.add_argument("--device", required=True, help="ex: 0 ou 0,1")
    p.add_argument("--variant", choices=["1a", "1b"], default="1a",
                   help="1a = modele unique multi-echelle | 1b = global R0 + specialiste HR")

    # Seuils d'allocation
    p.add_argument("--tau1", type=float, default=None, help="Seuil poubelle/R0 (valeur absolue d'efficacite)")
    p.add_argument("--tau2", type=float, default=None, help="Seuil R0/R1 (valeur absolue d'efficacite)")
    p.add_argument("--auto_thresholds", action="store_true",
                   help="Fixe tau1/tau2 par quantiles empiriques du batch courant")
    p.add_argument("--q1", type=float, default=40.0, help="Percentile pour tau1 (mode auto)")
    p.add_argument("--q2", type=float, default=90.0, help="Percentile pour tau2 (mode auto)")
    p.add_argument("--extreme_score", action="store_true",
                   help="Le top (100-extreme_percentile)%% du batch part en R2 (16 patches)")
    p.add_argument("--extreme_percentile", type=float, default=98.0)

    # Equilibrage du sampler (variante 1a, et sous-ensemble HR de la 1b)
    p.add_argument("--balance_mode", choices=["none", "oversample"], default="oversample")
    p.add_argument("--balance_cap", type=int, default=10,
                   help="Facteur max de duplication en mode oversample")

    # Budget realiste (grille des encadrants) -- active si bw_kbps > 0
    p.add_argument("--bw_kbps", type=float, default=0.0,
                   help="Debit du lien radio en kbit/s (22, 88 ou 176). "
                        "0 = mode legacy sans budget (seuils purs).")
    p.add_argument("--annot_min", type=float, default=0.0,
                   help="Budget d'annotation humaine par mission en minutes (3, 10 ou 30). "
                        "Sert aussi de fenetre de transmission : volume = bw x annot_min.")
    p.add_argument("--no_downgrade", action="store_true",
                   help="Interdit la retrogradation R2->R1->R0 quand le budget octets restant est insuffisant")
    p.add_argument("--replay_n", type=int, default=0,
                   help="Replay Buffer : N images Train_Init (deja au sol, cout BP nul) "
                        "injectees dans chaque fine-tuning contre l'oubli catastrophique. 0 = off.")

    # Protocole
    p.add_argument("--missions", type=int, default=3)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--pool_cap", type=int, default=0,
                   help="Limite le pool par run (0 = tout le pool). Utile pour reduire le cout de calcul.")
    p.add_argument("--no_baselines", action="store_true",
                   help="Desactive les baselines ceal_r0/random_r0 a budget apparie")

    # Bande passante
    p.add_argument("--jpeg_quality", type=int, default=JPEG_QUALITY_DEFAULT,
                   help="Qualite JPEG unique pour TOUTES les resolutions (comparaison honnete)")

    # Entrainement / eval (defauts = champion CEAL)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--freeze", type=int, default=10)
    p.add_argument("--lr0", type=float, default=0.001)
    p.add_argument("--conf_threshold", type=float, default=0.15, help="Seuil de conf du scoring CEAL")
    p.add_argument("--fusion_iou", type=float, default=0.5, help="IoU du NMS classe-agnostique (1b)")
    p.add_argument("--skip_size_eval", action="store_true",
                   help="Desactive l'eval custom par taille d'objet (plus rapide, variante 1a)")
    p.add_argument("--size_eval_final_only", action="store_true",
                   help="Eval custom par taille uniquement a la derniere mission (gain de temps)")
    p.add_argument("--test_subset", type=int, default=0,
                   help="N images test max pour l'eval custom (0 = toutes)")
    p.add_argument("--keep_weights", action="store_true",
                   help="Conserve tous les poids d'entrainement (defaut : suppression "
                        "des dossiers train_* consommes pour economiser le disque)")

    # Modeles & chemins
    p.add_argument("--base_model", default=str(DEFAULT_BASE_MODEL))
    p.add_argument("--specialist_base_model", default=None,
                   help="Poids de depart du specialiste 1b (defaut : base_model)")
    p.add_argument("--results_root", default=str(RESULTS_ROOT_DEFAULT))
    p.add_argument("--campaign_suffix", default="", help="Suffixe du nom de campagne")
    return p


def validate_args(args):
    if not args.auto_thresholds and (args.tau1 is None or args.tau2 is None):
        raise SystemExit("[ERREUR] Fournir --tau1 et --tau2, ou utiliser --auto_thresholds")
    if not args.auto_thresholds and args.tau1 >= args.tau2:
        raise SystemExit("[ERREUR] tau1 doit etre < tau2")
    if args.auto_thresholds and args.q1 >= args.q2:
        raise SystemExit("[ERREUR] q1 doit etre < q2")
    if args.bw_kbps > 0 and args.annot_min <= 0:
        raise SystemExit("[ERREUR] --annot_min requis (> 0) avec --bw_kbps")
    if args.specialist_base_model is None:
        args.specialist_base_model = args.base_model
    return args


if __name__ == "__main__":
    args = validate_args(build_parser().parse_args())
    run_campaign(args)
