from pathlib import Path
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "analysis" / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def main():
    gp = DATA_DIR / "ghostpermit_results.csv"
    if not gp.exists():
        raise FileNotFoundError(f"Missing {gp}")
    df = pd.read_csv(gp)
    model_col = pick_col(df, ["model"])
    asr_col = pick_col(df, ["asr", "success"])
    arch_col = pick_col(df, ["arch"], required=False)
    combo_col = pick_col(df, ["combo"])
    pars_col = pick_col(df, ["pars"], required=False)
    results = []
    grouped = [g[asr_col].values for _, g in df.groupby(model_col) if len(g) > 0]
    if len(grouped) >= 2:
        stat, p = kruskal(*grouped)
        results.append({"test": "kruskal_model_asr", "statistic": stat, "p_value": p, "note": "ASR differences across models"})
    if arch_col is not None:
        tmp = df.copy()
        tmp["is_reasoning"] = tmp[arch_col].astype(str).str.contains("Reason", case=False, na=False).astype(int)
        g1 = tmp[tmp["is_reasoning"] == 1][asr_col].values
        g0 = tmp[tmp["is_reasoning"] == 0][asr_col].values
        if len(g1) > 0 and len(g0) > 0:
            stat, p = mannwhitneyu(g1, g0, alternative="two-sided")
            results.append({"test": "mannwhitney_reasoning_vs_nonreasoning", "statistic": stat, "p_value": p, "note": "Reasoning vs non-reasoning ASR"})
    if pars_col is not None:
        combo_df = df.groupby(combo_col).agg(mean_asr=(asr_col, "mean"), pars=(pars_col, "mean")).reset_index()
        rho, p = spearmanr(combo_df["pars"], combo_df["mean_asr"])
        results.append({"test": "spearman_pars_vs_combo_asr", "statistic": rho, "p_value": p, "note": "Monotonic association between PARS and mean combo ASR"})
    out = pd.DataFrame(results)
    out_file = OUT_DIR / "statistical_tests.csv"
    out.to_csv(out_file, index=False)
    print("\n== Statistical Tests ==")
    print(out.to_string(index=False))
    print(f"\nSaved: {out_file}")

if __name__ == "__main__":
    main()
