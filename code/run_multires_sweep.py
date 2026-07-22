# run_multires_sweep.py
# -*- coding: utf-8 -*-
"""
SWEEP sur les couples de seuils (tau1, tau2) du Scenario 1.

Chaque couple correspond a un budget bande passante different :
seuils hauts -> peu de tuiles transmises -> budget bas, et inversement.
Les baselines ceal_r0/random_r0 sont lancees automatiquement au budget
apparie DANS chaque campagne (cf. run_multires_scenario1.py).

Deux modes :
  - quantiles empiriques (recommande, robuste inter-missions) :
        python run_multires_sweep.py --device 0 --variant 1a \
            --quantile_pairs "60:95,40:90,25:80"
  - valeurs absolues d'efficacite :
        python run_multires_sweep.py --device 0 --variant 1a \
            --tau_pairs "0.03:0.09,0.02:0.06,0.01:0.04"

Le sweep est sequentiel sur un GPU. Pour paralleliser sur les 4 cartes,
lancer une instance par GPU avec des couples differents, ex. :
    GPU0 : --quantile_pairs "60:95"
    GPU1 : --quantile_pairs "40:90"
    GPU2 : --quantile_pairs "25:80"
    GPU3 : --quantile_pairs "40:90" --variant 1b
"""
import argparse
import copy

from run_multires_scenario1 import build_parser, validate_args, run_campaign


def parse_pairs(spec):
    """'a:b,c:d' -> [(a, b), (c, d)]"""
    pairs = []
    for chunk in spec.split(","):
        lo, hi = chunk.strip().split(":")
        pairs.append((float(lo), float(hi)))
    return pairs


def parse_list(spec):
    """'22,88,176' -> [22.0, 88.0, 176.0]"""
    return [float(x) for x in spec.split(",") if x.strip()]


def main():
    sweep_parser = argparse.ArgumentParser(add_help=False)
    sweep_parser.add_argument("--quantile_pairs", default=None,
                              help='Couples de percentiles "q1:q2,q1:q2,..." (mode auto_thresholds)')
    sweep_parser.add_argument("--tau_pairs", default=None,
                              help='Couples de seuils absolus "tau1:tau2,..."')
    sweep_parser.add_argument("--bw_list", default=None,
                              help='GRILLE REALISTE : debits en kbps, ex "22,88,176"')
    sweep_parser.add_argument("--annot_list", default=None,
                              help='GRILLE REALISTE : budgets annotation en minutes, ex "3,10,30"')

    # On reutilise le parser du runner pour tous les autres parametres
    base_parser = build_parser()
    sweep_args, remaining = sweep_parser.parse_known_args()
    base_args = base_parser.parse_args(remaining)

    grid_mode = sweep_args.bw_list or sweep_args.annot_list

    # 1. Variantes de seuils demandees (sinon : seuils du runner tels quels)
    threshold_variants = []
    if sweep_args.quantile_pairs:
        for q1, q2 in parse_pairs(sweep_args.quantile_pairs):
            threshold_variants.append({"auto": True, "q1": q1, "q2": q2})
    if sweep_args.tau_pairs:
        for t1, t2 in parse_pairs(sweep_args.tau_pairs):
            threshold_variants.append({"auto": False, "tau1": t1, "tau2": t2})
    if not threshold_variants:
        if grid_mode:
            # Grille realiste : un seul jeu de seuils (celui du runner, auto par defaut)
            threshold_variants = [{"auto": True, "q1": base_args.q1, "q2": base_args.q2}]
        else:
            # Sweep legacy pur seuils : 3 budgets bas / moyen / haut
            threshold_variants = [{"auto": True, "q1": q1, "q2": q2}
                                  for q1, q2 in parse_pairs("60:95,40:90,25:80")]
            print("[INFO] Aucun couple fourni -> defaut : 60:95, 40:90, 25:80")

    # 2. Grille (bw, annot) realiste, ou cellule unique en mode legacy
    if grid_mode:
        bw_values = parse_list(sweep_args.bw_list or "22,88,176")
        annot_values = parse_list(sweep_args.annot_list or "3,10,30")
        grid_cells_list = [(bw, t) for bw in bw_values for t in annot_values]
    else:
        grid_cells_list = [(base_args.bw_kbps, base_args.annot_min)]

    campaigns = []
    for bw, annot in grid_cells_list:
        for thr in threshold_variants:
            a = copy.deepcopy(base_args)
            a.bw_kbps, a.annot_min = bw, annot
            if thr["auto"]:
                a.auto_thresholds = True
                a.q1, a.q2 = thr["q1"], thr["q2"]
                a.tau1, a.tau2 = None, None
            else:
                a.auto_thresholds = False
                a.tau1, a.tau2 = thr["tau1"], thr["tau2"]
            campaigns.append(a)

    print("=" * 60)
    print(f"SWEEP SCENARIO 1 : {len(campaigns)} campagne(s) sur le GPU {base_args.device}")
    if grid_mode:
        for a in campaigns:
            vol = a.bw_kbps * 1000 / 8 * a.annot_min * 60 / 1e6
            print(f"  - {a.bw_kbps:g} kbps x {a.annot_min:g} min -> {vol:.2f} Mo/mission")
    print("=" * 60)

    for idx, a in enumerate(campaigns, 1):
        thr = f"Q{a.q1}-{a.q2}" if a.auto_thresholds else f"tau{a.tau1}-{a.tau2}"
        cell = f", cellule {a.bw_kbps:g}kbps x {a.annot_min:g}min" if a.bw_kbps > 0 else ""
        print(f"\n>>> CAMPAGNE {idx}/{len(campaigns)} : variante {a.variant}, seuils {thr}{cell}")
        run_campaign(validate_args(a))

    print("\n[SUCCES] Sweep termine. Lancer plot_multires_results.py pour les figures.")


if __name__ == "__main__":
    main()
