# build_multires_mapping.py
# -*- coding: utf-8 -*-
"""
Genere UNE FOIS POUR TOUTES le mapping deterministe entre les tuiles R0
(dataset original 1000x1000) et leurs patches :
  - R1 : dataset_split4  -> 4 patches par tuile  (grille 2x2, nommage {stem}_{i}_{j})
  - R2 : dataset_split16 -> 16 patches par tuile (grille 4x4, nommage {stem}_{i}_{j})

Le nommage vient de create_super_res_dataset.py, il est donc deterministe.
Ce script verifie l'existence physique de chaque patch sur le disque et
sauvegarde le mapping dans un JSON unique :

    /linux/antoimartin/v2/multires_mappings.json

Structure du JSON :
{
  "Unlabeled_Pool_Stratified": {
      "<r0_stem>": {"r1": ["<stem>_0_0", ...], "r2": ["<stem>_0_0", ...]},
      ...
  },
  "Test": {...},
  ...
}

Usage (sur la machine externe) :
    python build_multires_mapping.py
"""
import json
from pathlib import Path

from multires_common import (
    BASE_DIR, R0_DATASET, R1_DATASET, R2_DATASET,
    MAPPING_JSON, MAPPING_SUBFOLDERS,
    resolve_subdir, expected_patch_stems,
)


def build_mapping_for_subfolder(subfolder):
    """Construit le mapping r0_stem -> {r1: [...], r2: [...]} pour un subfolder."""
    r0_img_dir = resolve_subdir(R0_DATASET, "images", subfolder)
    r1_img_dir = resolve_subdir(R1_DATASET, "images", subfolder)
    r2_img_dir = resolve_subdir(R2_DATASET, "images", subfolder)

    if not r0_img_dir.exists():
        print(f"[SKIP] {subfolder} : dossier R0 introuvable ({r0_img_dir})")
        return None

    r0_stems = sorted(p.stem for p in r0_img_dir.glob("*.jpg"))
    print(f"[{subfolder}] {len(r0_stems)} tuiles R0 trouvees dans {r0_img_dir}")

    # Sets des fichiers reellement presents (evite 20 appels disque par tuile)
    r1_present = {p.stem for p in r1_img_dir.glob("*.jpg")} if r1_img_dir.exists() else set()
    r2_present = {p.stem for p in r2_img_dir.glob("*.jpg")} if r2_img_dir.exists() else set()

    mapping = {}
    n_r1_missing, n_r2_missing = 0, 0

    for stem in r0_stems:
        r1_ids = [s for s in expected_patch_stems(stem, grid=2) if s in r1_present]
        r2_ids = [s for s in expected_patch_stems(stem, grid=4) if s in r2_present]

        if len(r1_ids) != 4:
            n_r1_missing += 1
        if len(r2_ids) != 16:
            n_r2_missing += 1

        mapping[stem] = {"r1": r1_ids, "r2": r2_ids}

    print(f"  -> R1 : {len(r1_present)} patches presents | tuiles incompletes : {n_r1_missing}")
    print(f"  -> R2 : {len(r2_present)} patches presents | tuiles incompletes : {n_r2_missing}")
    return mapping


def main():
    print("=" * 60)
    print("GENERATION DU MAPPING MULTI-RESOLUTION R0 -> R1 / R2")
    print("=" * 60)
    print(f"R0 : {R0_DATASET}")
    print(f"R1 : {R1_DATASET}")
    print(f"R2 : {R2_DATASET}")

    full_mapping = {}
    for subfolder in MAPPING_SUBFOLDERS:
        m = build_mapping_for_subfolder(subfolder)
        if m is not None:
            full_mapping[subfolder] = m

    with open(MAPPING_JSON, "w") as f:
        json.dump(full_mapping, f)

    size_mb = Path(MAPPING_JSON).stat().st_size / 1e6
    print(f"\n[SUCCES] Mapping sauvegarde : {MAPPING_JSON} ({size_mb:.1f} Mo)")


if __name__ == "__main__":
    main()
