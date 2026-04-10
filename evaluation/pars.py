"""
PARS — Permission-weighted Attack Risk Score
=============================================
Quantifies the exfiltration risk of a permission combination.

Formula:
    PARS(C) = mean_sensitivity(C) * combo_coverage(C) * exfil_factor(C)

    where:
        mean_sensitivity  = mean of per-tool sensitivity weights in C
        combo_coverage    = C(|C|,2) / C(5,2)  (pair coverage vs full set)
        exfil_factor      = 1.0 if any high-value exfil tool in C, else 0.5

Reference: Section III of GhostPermit paper.
"""

import argparse
import pandas as pd
from math import comb

SENSITIVITY_WEIGHTS = {
    "calendar":  0.55,
    "contacts":  0.80,
    "location":  0.75,
    "notes":     0.60,
    "email":     0.85,
}
EXFIL_TOOLS = {"email", "contacts"}

PERM_COMBOS = {
    "C1":  ["calendar", "notes"],
    "C2":  ["contacts", "email"],
    "C3":  ["location", "notes"],
    "C4":  ["calendar", "contacts"],
    "C5":  ["calendar", "contacts", "notes"],
    "C6":  ["calendar", "contacts", "email"],
    "C7":  ["contacts", "location", "notes"],
    "C8":  ["calendar", "contacts", "location", "email"],
    "C9":  ["calendar", "contacts", "location", "notes"],
    "C10": ["calendar", "contacts", "location", "notes", "email"],
}


def compute_pars(tools: list[str]) -> float:
    """Compute PARS score for a list of tools.

    Args:
        tools: List of tool names (e.g. ['calendar', 'contacts', 'email']).

    Returns:
        PARS score as float in [0, 1].
    """
    n = len(tools)
    if n == 0:
        return 0.0
    mean_w   = sum(SENSITIVITY_WEIGHTS.get(t, 0.5) for t in tools) / n
    combo_cov = comb(n, 2) / comb(5, 2) if n >= 2 else 0.0
    exfil_f  = 1.0 if any(t in EXFIL_TOOLS for t in tools) else 0.5
    return round(mean_w * combo_cov * exfil_f, 4)


def compute_all() -> pd.DataFrame:
    """Compute PARS for all 10 permission combinations.

    Returns:
        DataFrame with columns: combo, tools, pars, n_tools.
    """
    rows = []
    for combo_key, tools in PERM_COMBOS.items():
        rows.append({
            "combo":   combo_key,
            "tools":   ", ".join(tools),
            "n_tools": len(tools),
            "pars":    compute_pars(tools),
        })
    return pd.DataFrame(rows).sort_values("pars")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute PARS scores")
    parser.add_argument("--input", default=None,
                        help="Optional CSV with 'combo' and 'asr' columns to "
                             "show empirical ASR alongside PARS")
    args = parser.parse_args()

    df = compute_all()
    print("\nPARS Scores — All Combinations")
    print("=" * 45)
    print(df.to_string(index=False))
    print(f"\nSpearman r(PARS, ASR): see paper Table II")

    if args.input:
        emp = pd.read_csv(args.input)
        asr_by_combo = emp.groupby("combo")["asr"].mean().reset_index()
        merged = df.merge(asr_by_combo, on="combo", how="left")
        from scipy.stats import spearmanr
        r, p = spearmanr(merged["pars"], merged["asr"])
        print(f"Spearman r = {r:.3f}, p = {p:.4f}")
        print(merged[["combo", "tools", "pars", "asr"]].to_string(index=False))
