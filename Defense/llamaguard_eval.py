from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "defense_results.csv"

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing data file: {DATA}")

    df = pd.read_csv(DATA)

    detector_col = pick_col(df, ["detector", "guard_model", "defense_model", "model"])
    detected_col = pick_col(df, ["attack_detected", "detected", "flagged", "blocked", "is_flagged"])

    label_col = pick_col(df, ["label", "gt", "is_attack", "malicious"], required=False)
    if label_col is not None:
        attack_df = df[df[label_col].astype(str).isin(["1", "True", "true", "attack", "malicious"])]
        if len(attack_df) == 0:
            attack_df = df.copy()
    else:
        attack_df = df.copy()

    attack_df = attack_df.copy()
    attack_df[detected_col] = attack_df[detected_col].astype(int)

    summary = (
        attack_df.groupby(detector_col)[detected_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={
            detector_col: "detector",
            "count": "n_trials",
            "sum": "n_detected",
            "mean": "detection_rate"
        })
    )

    summary["detection_rate_pct"] = (summary["detection_rate"] * 100).round(2)

    out_dir = ROOT / "analysis" / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "defense_summary.csv"
    summary.to_csv(out_file, index=False)

    print("\n== LlamaGuard / Defense Evaluation ==")
    print(summary.to_string(index=False))
    print(f"\nSaved: {out_file}")

if __name__ == "__main__":
    main()
