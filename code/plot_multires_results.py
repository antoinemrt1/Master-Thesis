# plot_multires_results.py
# -*- coding: utf-8 -*-
"""
Figures et tables du Scenario 1 (multi-resolution).

Lit les CSV produits par run_multires_scenario1.py / run_multires_sweep.py
dans AL_Multires_S1/ et genere :

  1. FIGURE PRINCIPALE : mAP@50 en fonction du volume cumule transmis (Mo),
     avec les baselines CEAL classique (tout R0) et Random au meme budget.
     Un point par mission, barres d'erreur = ecart-type sur les runs.
  2. Figure secondaire : proportion de tuiles par resolution et par mission.
  3. Figure secondaire : mAP@50 par taille d'objet (small / medium / large).
  4. Table des resultats par variante et par seuil (mission finale),
     avec ecarts-types -> multires_s1_results_table.csv + .md

Usage :
    python plot_multires_results.py [--results_root .../AL_Multires_S1]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

DEFAULT_ROOT = Path("/linux/antoimartin/v2/code/AL_Multires_S1")

STRATEGY_STYLE = {
    "multires":           {"label": "Multi-resolution (Scenario 1)", "color": "#27ae60", "marker": "s"},
    "baseline_ceal_r0":   {"label": "CEAL classique (tout R0)",      "color": "#2980b9", "marker": "o"},
    "baseline_random_r0": {"label": "Random (meme budget Mo)",       "color": "#e67e22", "marker": "^"},
}


def load_campaigns(root):
    """{nom_campagne: df_raw} pour chaque campagne trouvee."""
    campaigns = {}
    for csv in sorted(root.glob("S1_*/all_runs_raw_metrics.csv")):
        df = pd.read_csv(csv)
        if not df.empty:
            campaigns[csv.parent.name] = df
    return campaigns


def summarize(df):
    """Aggregation (Strategy, Mission) -> moyennes/std sur les runs."""
    return df.groupby(["Strategy", "Mission"]).agg(
        mAP50_mean=("mAP50", "mean"), mAP50_std=("mAP50", "std"),
        mAP50_custom_mean=("mAP50_custom", "mean"), mAP50_custom_std=("mAP50_custom", "std"),
        mAP50_small=("mAP50_small", "mean"), mAP50_medium=("mAP50_medium", "mean"),
        mAP50_large=("mAP50_large", "mean"),
        MB_cum_mean=("MB_cum", "mean"), MB_cum_std=("MB_cum", "std"),
        r0=("N_tiles_r0", "mean"), r1=("N_tiles_r1", "mean"),
        r2=("N_tiles_r2", "mean"), trash=("N_trash", "mean"),
        Human_min_cum=("Human_min_cum", "mean"),
    ).reset_index().sort_values("Mission")


def headline_map_columns(campaign_name):
    """La 1b se juge sur la fusion custom ; la 1a sur la mAP officielle."""
    if "_1b_" in campaign_name:
        return "mAP50_custom_mean", "mAP50_custom_std"
    return "mAP50_mean", "mAP50_std"


def plot_map_vs_mb(campaigns, out_path):
    """FIGURE PRINCIPALE : mAP@50 vs Mo cumules, toutes campagnes + baselines."""
    plt.figure(figsize=(12, 7.5))
    linestyles = ["-", "--", "-.", ":"]

    for c_idx, (name, df) in enumerate(campaigns.items()):
        s = summarize(df)
        ls = linestyles[c_idx % len(linestyles)]
        map_col, std_col = headline_map_columns(name)
        thr_tag = name.replace("S1_", "").split("_bal")[0]

        for strat, style in STRATEGY_STYLE.items():
            sub = s[s["Strategy"] == strat]
            if sub.empty:
                continue
            # Pour les baselines, la mAP headline est toujours l'officielle
            mcol, scol = (map_col, std_col) if strat == "multires" else ("mAP50_mean", "mAP50_std")
            label = f"{style['label']} [{thr_tag}]" if strat == "multires" else \
                    (style["label"] if c_idx == 0 else None)
            plt.errorbar(sub["MB_cum_mean"], sub[mcol], yerr=sub[scol].fillna(0),
                         xerr=sub["MB_cum_std"].fillna(0),
                         color=style["color"], marker=style["marker"], linestyle=ls,
                         capsize=3, linewidth=2, markersize=7,
                         alpha=1.0 if strat == "multires" else 0.75, label=label)

    plt.xlabel("Volume cumule transmis (Mo)")
    plt.ylabel("mAP@50 (Test Set)")
    plt.title("Scenario 1 : Allocation de resolution par tuile\nmAP@50 vs bande passante consommee")
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[FIGURE] {out_path}")


def plot_resolution_proportions(campaigns, out_path):
    """Proportion de tuiles poubelle / R0 / R1 / R2 par mission (barres empilees)."""
    n = len(campaigns)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)

    for ax, (name, df) in zip(axes[0], campaigns.items()):
        s = summarize(df)
        sub = s[(s["Strategy"] == "multires") & (s["Mission"] > 0)]
        if sub.empty:
            continue
        missions = sub["Mission"].values
        total = (sub["trash"] + sub["r0"] + sub["r1"] + sub["r2"]).values
        total = np.maximum(total, 1)
        bottom = np.zeros(len(missions))
        for col, color, lab in [("trash", "#95a5a6", "Non transmis"),
                                ("r0", "#2980b9", "R0"),
                                ("r1", "#27ae60", "R1 (x4 patches)"),
                                ("r2", "#c0392b", "R2 (x16 patches)")]:
            vals = sub[col].values / total * 100.0
            ax.bar(missions, vals, bottom=bottom, color=color, label=lab, width=0.6)
            bottom += vals
        ax.set_title(name.replace("S1_", ""), fontsize=10)
        ax.set_xlabel("Mission")
        ax.set_ylabel("% des tuiles scorees")
        ax.set_xticks(missions)
        ax.legend(fontsize=8)

    plt.suptitle("Repartition des tuiles par resolution transmise")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[FIGURE] {out_path}")


def plot_map_by_size(campaigns, out_path):
    """mAP@50 par taille d'objet -- verifie que le melange R0/R1 n'introduit pas de biais de scale."""
    n = len(campaigns)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)

    for ax, (name, df) in zip(axes[0], campaigns.items()):
        s = summarize(df)
        for strat, style in STRATEGY_STYLE.items():
            sub = s[s["Strategy"] == strat]
            if sub.empty or sub["mAP50_small"].isna().all():
                continue
            for col, ls, lab in [("mAP50_small", "-", "small"),
                                 ("mAP50_medium", "--", "medium"),
                                 ("mAP50_large", ":", "large")]:
                ax.plot(sub["Mission"], sub[col], color=style["color"], linestyle=ls,
                        marker=style["marker"], markersize=5,
                        label=f"{strat.replace('baseline_', '')} {lab}")
        ax.set_title(name.replace("S1_", ""), fontsize=10)
        ax.set_xlabel("Mission")
        ax.set_ylabel("mAP@50")
        ax.legend(fontsize=7)

    plt.suptitle("mAP@50 par taille d'objet (small / medium / large)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[FIGURE] {out_path}")


def plot_grid_heatmap(campaigns, out_path):
    """
    GRILLE REALISTE : heatmap bande passante x temps d'annotation.
    Gauche : mAP@50 finale du multires. Droite : delta vs CEAL classique (R0)
    au meme budget. Ne fait rien si moins de 2 campagnes en mode grille.
    """
    import re
    cells = {}
    for name, df in campaigns.items():
        m = re.search(r"BW([\d.]+)k_T([\d.]+)m", name)
        if not m:
            continue
        bw, tmin = float(m.group(1)), float(m.group(2))
        last = df["Mission"].max()
        get = lambda strat: df[(df["Strategy"] == strat) & (df["Mission"] == last)]["mAP50"].mean()
        cells[(bw, tmin)] = {"multires": get("multires"), "ceal": get("baseline_ceal_r0")}

    if len(cells) < 2:
        return

    bws = sorted({k[0] for k in cells})
    tms = sorted({k[1] for k in cells})
    grid_map = np.full((len(bws), len(tms)), np.nan)
    grid_delta = np.full((len(bws), len(tms)), np.nan)
    for (bw, tmin), v in cells.items():
        i, j = bws.index(bw), tms.index(tmin)
        grid_map[i, j] = v["multires"]
        if not np.isnan(v.get("ceal", np.nan)):
            grid_delta[i, j] = v["multires"] - v["ceal"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, title, cmap, fmt in [
            (axes[0], grid_map, "mAP@50 finale (multires)", "viridis", "{:.4f}"),
            (axes[1], grid_delta, "Delta mAP vs CEAL classique (meme budget)", "RdYlGn", "{:+.4f}")]:
        vmax = np.nanmax(np.abs(data)) if "Delta" in title else None
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=-vmax if vmax else None, vmax=vmax)
        ax.set_xticks(range(len(tms)), [f"{t:g} min" for t in tms])
        ax.set_yticks(range(len(bws)), [f"{b:g} kbps" for b in bws])
        ax.set_xlabel("Budget annotation / mission")
        ax.set_ylabel("Debit du lien")
        ax.set_title(title, fontsize=11)
        for i in range(len(bws)):
            for j in range(len(tms)):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                            fontsize=10, color="white",
                            path_effects=None)
        fig.colorbar(im, ax=ax, shrink=0.85)

    plt.suptitle("Scenario 1 -- Grille realiste bande passante x temps d'annotation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[FIGURE] {out_path}")


def build_results_table(campaigns, root):
    """Table finale par variante et par seuil, mission finale, std sur les runs."""
    rows = []
    for name, df in campaigns.items():
        last_mission = df["Mission"].max()
        for strat in df["Strategy"].unique():
            sub = df[(df["Strategy"] == strat) & (df["Mission"] == last_mission)]
            if sub.empty:
                continue
            map_col = "mAP50_custom" if ("_1b_" in name and strat == "multires") else "mAP50"
            rows.append({
                "Campagne": name,
                "Strategie": strat,
                "Mission_finale": int(last_mission),
                "mAP50_mean": sub[map_col].mean(),
                "mAP50_std": sub[map_col].std(),
                "MB_cum_mean": sub["MB_cum"].mean(),
                "MB_cum_std": sub["MB_cum"].std(),
                "Human_min_cum_mean": sub["Human_min_cum"].mean(),
                "N_runs": len(sub),
            })

    table = pd.DataFrame(rows).sort_values(["Campagne", "Strategie"])
    csv_path = root / "multires_s1_results_table.csv"
    table.to_csv(csv_path, index=False)

    md_path = root / "multires_s1_results_table.md"
    with open(md_path, "w") as f:
        f.write("# Scenario 1 -- Resultats par variante et par seuil\n\n")
        f.write("| Campagne | Strategie | mAP@50 (± std) | Mo cumules (± std) | Cout humain (min) | Runs |\n")
        f.write("|---|---|---|---|---|---|\n")
        for _, r in table.iterrows():
            std = 0.0 if pd.isna(r["mAP50_std"]) else r["mAP50_std"]
            mb_std = 0.0 if pd.isna(r["MB_cum_std"]) else r["MB_cum_std"]
            f.write(f"| {r['Campagne']} | {r['Strategie']} "
                    f"| {r['mAP50_mean']:.4f} ± {std:.4f} "
                    f"| {r['MB_cum_mean']:.1f} ± {mb_std:.1f} "
                    f"| {r['Human_min_cum_mean']:.1f} | {int(r['N_runs'])} |\n")

    print(f"[TABLE] {csv_path}")
    print(f"[TABLE] {md_path}")
    print("\n" + table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default=str(DEFAULT_ROOT))
    args = parser.parse_args()

    root = Path(args.results_root)
    campaigns = load_campaigns(root)
    if not campaigns:
        print(f"[ERREUR] Aucune campagne S1_* trouvee dans {root}")
        return
    print(f"Campagnes trouvees : {list(campaigns.keys())}")

    plot_map_vs_mb(campaigns, root / "multires_s1_map_vs_mb.png")
    plot_resolution_proportions(campaigns, root / "multires_s1_resolution_proportions.png")
    plot_map_by_size(campaigns, root / "multires_s1_map_by_size.png")
    plot_grid_heatmap(campaigns, root / "multires_s1_grid_heatmap.png")
    build_results_table(campaigns, root)


if __name__ == "__main__":
    main()
