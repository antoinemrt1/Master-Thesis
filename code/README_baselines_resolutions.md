# Baselines mono-résolution (nouvelle direction — étape 1)

> **Contexte (juillet 2026)** : le modèle unique multi-échelle est abandonné (résultats
> de la grille : le mélange R0+R1 dégrade systématiquement). Nouvelle approche :
> **un modèle au sol par résolution** — R0, R1 (Split4), R2 (Split16) — chacun
> initialisé sur le Train_Init stratifié 20 % de SA résolution, puis fine-tuné
> progressivement avec les nouvelles tuiles de SA résolution.
> L'étape 2 (choix d'une bande passante réaliste basée sur de vraies données
> satellite) viendra ensuite.

## 1. État des lieux — vérifier ce qui existe

```bash
# R0 : doit exister (le champion, mAP50 = 0,3664)
ls /linux/antoimartin/v2/code/trained_models/baseline_stratified_20pct_yolov8l_1024/weights/

# R1/R2 : probablement JAMAIS entraînés (aucune mAP rapportée à ce jour)
ls /linux/antoimartin/v2/code/trained_models_splits/ 2>/dev/null || echo "-> vide : à entraîner"
```

## 2. Entraînements à lancer (si absents)

⚠️ Les défauts de `train_baseline_splits.py` sont maintenant **alignés sur la baseline
R0** (YOLOv8-**L** @ **1024** px) — indispensable pour que les perfs par résolution
soient comparables. Ne pas repasser en yolov8s/640.

```bash
# En parallèle sur 2×2 GPUs :
python train_baseline_splits.py --split split4  --device 0,1   # ≈ 13 200 imgs/epoch, ~10-15 h
python train_baseline_splits.py --split split16 --device 2,3   # ≈ 52 800 imgs/epoch, ~1-2 jours
# batch par défaut 8 (= 4/GPU sur 2 GPUs, footprint validé à 1024px).
# patience=20 : l'arrêt anticipé jouera bien avant les 200 epochs (le R0 a convergé à ~102).
```

Sorties : `code/trained_models_splits/baseline_init_split4_yolov8l_1024/weights/best.pt`
(idem `split16`).

## 3. Évaluation des performances initialisées

Après les entraînements (fonctionne aussi partiellement avant — il saute les modèles absents) :

```bash
python evaluate_baselines_resolutions.py --device 0
```

Produit `code/baseline_resolutions_report.csv` avec, par résolution :
- **`native_mAP50`** : perf sur le test set de SA résolution → c'est la référence
  d'initialisation que le fine-tuning progressif devra dépasser mission après mission ;
- **`R0frame_mAP50`** (+ small/medium/large) : les trois modèles évalués par le **même
  évaluateur** sur les **mêmes images test R0** (R1/R2 : inférence sur les 4/16 patches,
  reprojection des boîtes vers R0, NMS classe-agnostique) → seule colonne qui autorise
  une comparaison inter-résolutions.

Options : `--skip_reprojection` (rapide, natif seulement), `--only r1 r2`,
`--test_subset 500` (éval reprojetée sur un sous-ensemble), `--r1_model <path>` (chemins alternatifs).

⚠️ Ne pas comparer les `native_mAP50` entre résolutions : les test sets diffèrent
(objets coupés aux bords des patches, distribution des tailles changée par le zoom).
C'est précisément le rôle de la colonne `R0frame`.

## 4. Points de vigilance sur la correction des entraînements splits

- Les YAML (`data_split4.yaml`, `data_split16.yaml`) pointent `Train_Init_Stratified` /
  `Val` / `Test` des datasets splittés — mêmes images sources que le 20 % R0,
  découpées : l'« information vue » est comparable. ✓
- Les labels des patches ont été validés visuellement (rapport, figure « visual_proof_splits »). ✓
- Les patches vides ont un label vide → images de fond pour YOLO, comportement normal. ✓
- Early stopping sur la `Val` de la même résolution. ✓

## 5. Suite (étape 2, à venir)

- Choix d'une bande passante réaliste « satellite » suffisante pour l'apprentissage.
- Boucles de fine-tuning progressif par résolution (chaque modèle ne reçoit que les
  tuiles de sa résolution) — l'infrastructure existe déjà : chaînes de modèles du
  runner (variante 1b), mapping R0→R1/R2, comptabilité bande passante, replay.
