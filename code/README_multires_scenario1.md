# Scénario 1 — Allocation de résolution par tuile (single-pass)

> ⚠️ **PROTOCOLE ACTUEL (juillet 2026) : grille réaliste bande passante × annotation.**
> Suite au cadrage avec les encadrants, le budget par mission est désormais borné par
> un lien radio réaliste : volume = `bw_kbps × annot_min` ET annotation ≤ `annot_min`.
> Grille : {22, 88, 176} kbps × {3, 10, 30} min = 9 cellules.
>
> | Mo/mission | 3 min | 10 min | 30 min |
> |---|---|---|---|
> | **22 kbps**  | 0,50 | 1,65 | 4,95 |
> | **88 kbps**  | 1,98 | 6,60 | 19,80 |
> | **176 kbps** | 3,96 | 13,20 | 39,60 |
>
> (Repères : 1 tuile R0 ≈ 0,19 Mo ; 1 groupe R1 = 4 patches ≈ 0,81 Mo ;
> 1 groupe R2 = 16 patches ≈ 3,3 Mo — mesurés sur les runs de juillet.)
>
> **Lancement de la grille (3 GPUs, ~3 cellules chacun) :**
> ```bash
> # AVANT TOUT : libérer le disque (les anciennes campagnes ont causé l'ENOSPC)
> du -sh /linux/antoimartin/v2/code/AL_Multires_S1/*
> rm -rf /linux/antoimartin/v2/code/AL_Multires_S1/S1_1a_autoQ* \
>        /linux/antoimartin/v2/code/AL_Multires_S1/S1_1b_autoQ*   # CSV déjà sauvegardés en local
> df -h /linux/antoimartin/v2
>
> python run_multires_sweep.py --device 0 --variant 1a --bw_list "22"  --annot_list "3,10,30" --size_eval_final_only
> python run_multires_sweep.py --device 1 --variant 1a --bw_list "88"  --annot_list "3,10,30" --size_eval_final_only
> python run_multires_sweep.py --device 2 --variant 1a --bw_list "176" --annot_list "3,10,30" --size_eval_final_only
> # GPU 3 (optionnel) : cellule centrale en 5 missions pour la dynamique longue
> python run_multires_scenario1.py --device 3 --variant 1a --auto_thresholds \
>        --bw_kbps 88 --annot_min 10 --missions 5 --size_eval_final_only --campaign_suffix m5
> ```
> En mode grille, les baselines `ceal_r0`/`random_r0` utilisent directement le même
> budget exogène (octets + minutes) — plus besoin d'appariement a posteriori.
> Le mode historique « seuils purs sans budget » reste disponible (`--bw_kbps 0`).

Extension de la pipeline CEAL : le drone décide, tuile par tuile et **en un seul
passage**, à quelle résolution transmettre (poubelle / R0 / R1 / R2). Le budget
principal devient le **volume cumulé transmis en Mo** vs mAP@50 sur le test set.

## Fichiers

| Fichier | Rôle |
|---|---|
| `multires_common.py` | Module partagé : chemins R0/R1/R2, mapping, score CEAL, bande passante, reprojection, NMS, mAP custom par taille, équilibrage sampler |
| `build_multires_mapping.py` | Génère **une fois pour toutes** le mapping `r0_id → r1_ids/r2_ids` → `/linux/antoimartin/v2/multires_mappings.json` |
| `run_multires_scenario1.py` | Runner principal : boucle AL multirés + variantes 1a/1b + baselines à budget Mo apparié |
| `run_multires_sweep.py` | Sweep sur ≥3 couples (τ₁, τ₂) |
| `plot_multires_results.py` | Figure principale mAP vs Mo, figures secondaires, table de résultats |

## Ordre de lancement (machine externe)

### 0. Prérequis (une seule fois)
```bash
python build_multires_mapping.py
```
Vérifie l'existence physique des 4 patches R1 et 16 patches R2 de chaque tuile R0
(datasets `dataset_split4` / `dataset_split16` générés par `create_super_res_dataset.py`)
et sauvegarde le JSON. Les runners le chargent automatiquement.

### 1. Expérience complète (exemple : les 3 budgets sur 3 GPUs, variante 1a)
```bash
# GPU 0 : budget bande passante BAS (seuils hauts)
python run_multires_sweep.py --device 0 --variant 1a --quantile_pairs "60:95" --missions 3 --runs 3

# GPU 1 : budget MOYEN (les percentiles de l'énoncé : τ1=P40, τ2=P90)
python run_multires_sweep.py --device 1 --variant 1a --quantile_pairs "40:90" --missions 3 --runs 3

# GPU 2 : budget HAUT
python run_multires_sweep.py --device 2 --variant 1a --quantile_pairs "25:80" --missions 3 --runs 3

# GPU 3 : variante 1b (spécialiste HR) au budget moyen, pour comparaison 1a vs 1b
python run_multires_sweep.py --device 3 --variant 1b --quantile_pairs "40:90" --missions 3 --runs 3
```
Chaque campagne lance automatiquement, **dans le même run** (mêmes chunks de pool,
même seed) : la stratégie multirés, puis `ceal_r0` (CEAL classique tout-R0, tri par
efficacité) et `random_r0`, toutes deux **au même budget Mo par mission** que ce que
le run multirés a réellement consommé.

### Robustesse GPU partagé (OOM) et Replay Buffer

**CUDA OOM** : si un autre process occupe le GPU (`nvidia-smi` pour identifier le PID,
`kill <PID>` si c'est un zombie de vos anciens runs), le runner est maintenant blindé :
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` auto, `model.val()` au batch 4
(au lieu du défaut 16, source probable de l'OOM), et retry automatique du fine-tuning
avec batch divisé par 2 (4 → 2 → 1) en cas d'out-of-memory.

**`--replay_n N`** (défaut 0 = off) : injecte N images de `Train_Init_Stratified`
(déjà à la station sol → **coût bande passante et annotation NUL**) dans chaque
fine-tuning. Levier anti « oubli catastrophique » pour les petites cellules de la
grille — à activer en campagne complémentaire (`_repN` apparaît dans le nom) pour
mesurer son effet vs le protocole nu. C'est aussi l'analogue contrôlé de l'effet
Replay attendu du scénario 2 (le flux R0 de la pass 1).

### 1bis. Suivre une campagne en cours

Chaque campagne écrit maintenant :
- `campaign.log` — journal horodaté de chaque étape (scoring, allocation, Mo, fine-tuning, mAP) ;
- `all_runs_raw_metrics.csv` / `summary_metrics.csv` — **réécrits après chaque mission** de
  chaque stratégie (plus besoin d'attendre la fin du run).

Pour vérifier l'avancement sur le serveur :
```bash
tail -f code/AL_Multires_S1/S1_*/campaign.log        # progression en direct
nvidia-smi                                            # les process tournent-ils encore ?
ls code/AL_Multires_S1/S1_*/multires/run_1/           # train_m1/, eval_m1/... apparaissent au fil des missions
```
⏱️ **Ordre de grandeur** : sans `--pool_cap`, une mission Q25-80 fine-tune sur plusieurs
milliers d'images à 1024px (batch 4) → plusieurs heures **par mission**, et une campagne
complète (3 runs × [multirés + 2 baselines] × 3 missions) peut prendre plusieurs jours.
Le premier CSV utile apparaît dès la fin de la mission 1. Les plots sont générables à tout
moment sur un CSV partiel.

### 2. Figures + table
```bash
python plot_multires_results.py
```
Produit dans `code/AL_Multires_S1/` :
- `multires_s1_map_vs_mb.png` — **figure principale** : mAP@50 vs Mo cumulés, avec baselines ;
- `multires_s1_resolution_proportions.png` — proportion de tuiles par résolution ;
- `multires_s1_map_by_size.png` — mAP par taille d'objet (biais de scale) ;
- `multires_s1_results_table.{csv,md}` — table par variante/seuil avec écarts-types.

## Paramètres clés de `run_multires_scenario1.py`

- **Seuils** : `--tau1/--tau2` (valeurs absolues d'efficacité) **ou**
  `--auto_thresholds --q1 40 --q2 90` (quantiles empiriques du batch courant,
  calculés sur les tuiles à score > 0 ; les tuiles sans détection sont jetées,
  comme dans le CEAL existant).
- **`--extreme_score`** (+ `--extreme_percentile 98`) : le top 2 % du batch part
  en R2 (16 patches) au lieu de R1.
- **`--variant 1a`** : modèle unique multi-échelle sur le mélange R0+R1(+R2).
  Équilibrage du sampler via `--balance_mode {none,oversample}` (oversample =
  duplication par liens symboliques des résolutions minoritaires, équivalent à un
  weighted sampling inverse ; plafonné par `--balance_cap`).
- **`--variant 1b`** : modèle « global » fine-tuné sur R0 seul + « spécialiste »
  sur R1(+R2) seul. Éval par inférence des deux modèles, **reprojection** des
  boîtes du spécialiste dans le référentiel R0 (logique inverse exacte du découpage
  de `create_super_res_dataset.py`) puis **NMS classe-agnostique** (`--fusion_iou`).
- **Bande passante** : chaque image transmise est re-encodée en JPEG à qualité
  fixe (`--jpeg_quality 85`, identique pour toutes les résolutions → comparaison
  honnête) et ses octets sont ajoutés au compteur cumulé (ventilé R0/R1/R2).
- **Coût humain** : loggé en parallèle (2 s d'ouverture + 3,5 s par boîte GT de
  chaque unité transmise, patchs comptés individuellement). Les deux budgets
  restent indépendants.
- **`--pool_cap N`** : limite le pool par run pour réduire le coût de calcul
  (avec q1=40, ~60 % du pool est transmis → datasets d'entraînement très gros ;
  commencer par ex. avec `--pool_cap 3000` pour calibrer les temps).
- `--missions 3` (5 idéalement), `--runs 3`, hyperparamètres de fine-tuning
  identiques au champion (`epochs=10, freeze=10, lr0=0.001, imgsz=1024, batch=4`).
- `--base_model` : par défaut la baseline stratifiée 20 %
  (`baseline_stratified_20pct_yolov8l_1024`), la même que le champion CEAL.

## Métriques loggées (CSV `all_runs_raw_metrics.csv` par campagne)

`mAP50` (officielle `model.val()`, comparable aux résultats CEAL précédents),
`mAP50_95`, `mAP50_custom` (évaluateur interne — c'est la métrique headline de la
variante 1b car la fusion NMS n'est pas évaluable par `model.val()`),
`mAP50_small/medium/large` (bornes COCO 32²/96² px dans le référentiel R0),
`MB_mission`, `MB_cum` (+ ventilation R0/R1/R2), `N_trash/N_tiles_r0/r1/r2`,
`N_units_tx`, `Tau1/Tau2/Tau_ext` (valeurs absolues utilisées à chaque mission),
`Human_min_mission`, `Human_min_cum`.

## Choix d'implémentation à connaître

1. **Mapping** : nommage déterministe `{stem}_{i}_{j}` hérité de
   `create_super_res_dataset.py` ; le JSON matérialisé sert d'index vérifié
   (fallback : reconstruction à la volée avec warning).
2. **Fichiers transmis préfixés** `r0_`/`r1_`/`r2_` dans le dataset cumulatif →
   traçabilité de la résolution pour l'équilibrage et les stats.
3. **Baselines à budget apparié** : remplissage glouton du budget en octets
   (une image trop grosse est sautée, on continue — même logique que le budget
   temps du CEAL existant).
4. **Éval par taille + fusion 1b** : évaluateur mAP@50 interne (interpolation
   all-points, GT hors-bin ignorées façon COCO). Les valeurs absolues peuvent
   différer légèrement de `model.val()` → comparer les courbes **au sein du même
   évaluateur** ; `mAP50` officielle toujours loggée en ancre.
5. **1b — drone** : le modèle redéployé sur le drone entre les missions est le
   modèle « global » (c'est lui qui score les tuiles R0).
6. **1b — spécialiste à l'éval** : inférence sur les 4 patches R1 du test set
   (le R2 sert à l'entraînement si `--extreme_score`, pas à l'éval, pour garder
   un coût d'éval raisonnable).
