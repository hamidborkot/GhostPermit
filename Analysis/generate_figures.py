from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "analysis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def save_fig(fig, name):
    path = FIG_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

def fig1_ghostpermit_heatmap():
    path = DATA_DIR / "ghostpermit_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    model_col = pick_col(df, ["model"])
    combo_col = pick_col(df, ["combo"])
    asr_col = pick_col(df, ["asr", "success"])
    piv = df.pivot_table(index=model_col, columns=combo_col, values=asr_col, aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_title("GhostPermit ASR by Model and Permission Combination")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Attack Success Rate")
    save_fig(fig, "figure1_ghostpermit_heatmap.png")

def fig2_pars_scatter():
    path = DATA_DIR / "ghostpermit_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    combo_col = pick_col(df, ["combo"])
    asr_col = pick_col(df, ["asr", "success"])
    pars_col = pick_col(df, ["pars"], required=False)
    if pars_col is None:
        print("Skipped Figure 2: no PARS column found.")
        return
    g = df.groupby(combo_col).agg(mean_asr=(asr_col, "mean"), pars=(pars_col, "mean")).reset_index()
    x = g["pars"].values
    y = g["mean_asr"].values
    coef = np.polyfit(x, y, 1)
    fit_y = coef[0] * x + coef[1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y)
    ax.plot(x, fit_y)
    for _, row in g.iterrows():
        ax.annotate(row[combo_col], (row["pars"], row["mean_asr"]), fontsize=8)
    ax.set_xlabel("PARS")
    ax.set_ylabel("Mean ASR")
    ax.set_title("PARS vs. Attack Success Rate")
    save_fig(fig, "figure2_pars_scatter.png")

def fig3_agentworm_hops():
    path = DATA_DIR / "agentworm_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    hop_col = pick_col(df, ["hop", "hops", "step"])
    success_col = pick_col(df, ["success", "infected", "propagated", "asr"])
    g = df.groupby(hop_col)[success_col].mean().reset_index().sort_values(hop_col)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g[hop_col], g[success_col], marker="o")
    ax.set_xlabel("Hop")
    ax.set_ylabel("Propagation Success Rate")
    ax.set_title("AgentWorm Multi-Hop Propagation")
    save_fig(fig, "figure3_agentworm_hops.png")

def fig4_defense_bars():
    path = DATA_DIR / "defense_results.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    detector_col = pick_col(df, ["detector", "guard_model", "defense_model", "model"])
    detected_col = pick_col(df, ["attack_detected", "detected", "flagged", "blocked", "is_flagged"])
    g = df.groupby(detector_col)[detected_col].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(g[detector_col], g[detected_col])
    ax.set_ylabel("Detection Rate")
    ax.set_title("Defense Detection Rate")
    ax.tick_params(axis="x", rotation=25)
    save_fig(fig, "figure4_defense_detection.png")

def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    fig1_ghostpermit_heatmap()
    fig2_pars_scatter()
    fig3_agentworm_hops()
    fig4_defense_bars()

if __name__ == "__main__":
    main()
