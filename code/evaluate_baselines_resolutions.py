# evaluate_baselines_resolutions.py
# -*- coding: utf-8 -*-
"""
EVALUATION DES BASELINES MONO-RESOLUTION (nouvelle direction : un modele au sol
par resolution, fine-tune progressivement dans SA resolution).

Pour chaque baseline disponible (R0 / R1=Split4 / R2=Split16), mesure :

  1. PERF NATIVE  : model.val() sur le test set de SA resolution
     (R0 -> dataset/images/test ; R1 -> dataset_split4/images/Test ; etc.)
     -> la reference d'initialisation de chaque modele, celle qui devra
        s'ameliorer mission apres mission lors du fine-tuning progressif.

  2. PERF COMPARABLE (referentiel R0) : les trois modeles evalues par le MEME
     evaluateur sur les MEMES images test R0 :
       - modele R0 : inference directe ;
       - modeles R1/R2 : inference sur les 4/16 patches de chaque image test,
         reprojection des boites vers R0 (inverse exact du decoupage), fusion
         NMS classe-agnostique.
     -> permet de comparer les resolutions entre elles + mAP par taille (S/M/L).

Usage :
    python evaluate_baselines_resolutions.py --device 0
    python evaluate_baselines_resolutions.py --device 0 --only r1 r2
    python evaluate_baselines_resolutions.py --device 0 --skip_reprojection   # natif seulement

Sortie : code/baseline_resolutions_report.csv + tableau console.
"""
import argparse
import os
from datetime import datetime

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

from multires_common import (
    BASE_DIR, R0_DATASET, DATASET_BY_RES, GRID_BY_RES, CLASS_NAMES,
    resolve_subdir, load_mapping, grid_cells, parse_patch_indices,
    reproject_patch_dets, nms_class_agnostic, results_to_dets,
    evaluate_detections, load_gt_for_images,
)

# Reference historique : baseline R0 = 0.3664 mAP@50 sur le test set R0
R0_REFERENCE_MAP50 = 0.3664

CONFIGS = {
    "r0": {
        "yaml": BASE_DIR / "data.yaml",
        "model": BASE_DIR / "code/trained_models/baseline_stratified_20pct_yolov8l_1024/weights/best.pt",
        "label": "R0 (natif 1000x1000)",
    },
    "r1": {
        "yaml": BASE_DIR / "data_split4.yaml",
        "model": BASE_DIR / "code/trained_models_splits/baseline_init_split4_yolov8l_1024/weights/best.pt",
        "label": "R1 (Split4, zoom x2)",
    },
    "r2": {
        "yaml": BASE_DIR / "data_split16.yaml",
        "model": BASE_DIR / "code/trained_models_splits/baseline_init_split16_yolov8l_1024/weights/best.pt",
        "label": "R2 (Split16, zoom x4)",
    },
}

TEST_IMG_DIR = R0_DATASET / "images" / "test"
TEST_LBL_DIR = R0_DATASET / "labels" / "test"
OUT_DIR = BASE_DIR / "code" / "eval_baselines_resolutions"


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def native_val(res, model_path, args):
    """Perf native : model.val() sur le test set de la resolution du modele."""
    model = YOLO(str(model_path))
    r = model.val(data=str(CONFIGS[res]["yaml"]), split="test", imgsz=args.imgsz,
                  device=args.device, batch=args.batch, plots=False,
                  project=str(OUT_DIR), name=f"native_{res}", exist_ok=True, verbose=False)
    return float(r.box.map50), float(r.box.map)


def infer_r0_frame(res, model_path, test_imgs, mapping, dims, args):
    """Detections dans le referentiel R0 (inference directe pour r0, patches
    reprojetes + NMS classe-agnostique pour r1/r2)."""
    model = YOLO(str(model_path))
    dets = {}

    if res == "r0":
        for p in test_imgs:
            r = model(str(p), imgsz=args.imgsz, conf=0.001, iou=0.7, max_det=300,
                      device=args.device, verbose=False)
            dets[p.stem] = results_to_dets(r)
        return dets

    grid = GRID_BY_RES[res]
    patch_dir = resolve_subdir(DATASET_BY_RES[res], "images", "Test")
    for p in test_imgs:
        if p.stem not in dims:
            continue
        W, H = dims[p.stem]
        cells = grid_cells(W, H, grid)
        rows = []
        for ps in mapping.get(p.stem, {}).get(res, []):
            patch_path = patch_dir / (ps + ".jpg")
            if not patch_path.exists():
                continue
            r = model(str(patch_path), imgsz=args.imgsz, conf=0.001, iou=0.7,
                      max_det=300, device=args.device, verbose=False)
            i, j = parse_patch_indices(ps)
            rows.append(reproject_patch_dets(results_to_dets(r), cells[(i, j)]))
        merged = np.vstack(rows) if rows else np.zeros((0, 6))
        dets[p.stem] = nms_class_agnostic(merged, iou_thr=args.fusion_iou)
    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--only", nargs="+", choices=["r0", "r1", "r2"],
                        default=["r0", "r1", "r2"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--fusion_iou", type=float, default=0.5)
    parser.add_argument("--skip_reprojection", action="store_true",
                        help="Ne fait que la perf native (rapide)")
    parser.add_argument("--test_subset", type=int, default=0,
                        help="N images test max pour l'eval reprojetee (0 = toutes)")
    for res in CONFIGS:
        parser.add_argument(f"--{res}_model", default=None,
                            help=f"Chemin alternatif du modele {res}")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Modeles disponibles ?
    available = {}
    for res in args.only:
        model_path = Path(getattr(args, f"{res}_model") or CONFIGS[res]["model"])
        if model_path.exists():
            available[res] = model_path
            log(f"[OK]      {res} : {model_path}")
        else:
            log(f"[ABSENT]  {res} : {model_path}")
            log(f"          -> a entrainer : python train_baseline_splits.py "
                f"--split split{4 if res == 'r1' else 16} --device 0,1")
    if not available:
        log("[ERREUR] Aucun modele disponible.")
        return

    # Contexte eval reprojetee (GT R0 chargees une fois)
    eval_ctx = None
    if not args.skip_reprojection:
        test_imgs = sorted(TEST_IMG_DIR.glob("*.jpg"))
        if args.test_subset > 0:
            test_imgs = test_imgs[:args.test_subset]
        log(f"[EVAL] Chargement des GT R0 ({len(test_imgs)} images test)...")
        gts, dims = load_gt_for_images(test_imgs, TEST_LBL_DIR)
        mapping = load_mapping("Test")
        eval_ctx = {"test_imgs": test_imgs, "gts": gts, "dims": dims, "mapping": mapping}

    rows = []
    for res, model_path in available.items():
        log(f"===== {CONFIGS[res]['label']} =====")
        log("Perf NATIVE (test set de sa resolution)...")
        map50, map5095 = native_val(res, model_path, args)
        row = {"resolution": res, "label": CONFIGS[res]["label"],
               "model": str(model_path),
               "native_mAP50": map50, "native_mAP50_95": map5095,
               "R0frame_mAP50": np.nan, "R0frame_small": np.nan,
               "R0frame_medium": np.nan, "R0frame_large": np.nan}
        log(f"  -> native mAP50={map50:.4f} | mAP50-95={map5095:.4f}")

        if eval_ctx is not None:
            n_inf = len(eval_ctx["test_imgs"]) * {"r0": 1, "r1": 4, "r2": 16}[res]
            log(f"Perf COMPARABLE referentiel R0 ({n_inf} inferences)...")
            dets = infer_r0_frame(res, model_path, eval_ctx["test_imgs"],
                                  eval_ctx["mapping"], eval_ctx["dims"], args)
            m = evaluate_detections(dets, eval_ctx["gts"])
            row.update({"R0frame_mAP50": m["mAP50"], "R0frame_small": m["mAP50_small"],
                        "R0frame_medium": m["mAP50_medium"], "R0frame_large": m["mAP50_large"]})
            log(f"  -> R0-frame mAP50={m['mAP50']:.4f} | small={m['mAP50_small']:.4f} "
                f"medium={m['mAP50_medium']:.4f} large={m['mAP50_large']:.4f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = BASE_DIR / "code" / "baseline_resolutions_report.csv"
    df.to_csv(out_csv, index=False)
    print("\n" + "=" * 90)
    print("BILAN DES BASELINES PAR RESOLUTION "
          f"(reference historique R0 : mAP50 = {R0_REFERENCE_MAP50})")
    print("=" * 90)
    print(df.drop(columns=["model"]).to_string(index=False))
    print(f"\n[CSV] {out_csv}")
    print("\nRappel : la colonne 'native' est la reference que chaque modele devra "
          "depasser lors du fine-tuning progressif ; la colonne 'R0frame' permet de "
          "comparer les resolutions entre elles (meme evaluateur, memes images).")


if __name__ == "__main__":
    main()
