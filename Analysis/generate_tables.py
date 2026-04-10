from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "analysis" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def pick_col(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried: {candidates}")
    return None

def write_tex(df, name, caption, label):
    path = TABLE_DIR / name
    tex = df.to_latex(index=False, escape=False, caption=caption, label=label)
    path.write_text(tex, encoding="utf-8")
    print(f"Saved: {path}")

def main():
    gp = DATA_DIR / "ghostpermit_results.csv"
    if gp.exists():
        df = pd.read_csv(gp)
        model_col = pick_col(df, ["model"])
        arch_col = pick_col(df, ["arch"], required=False)
        asr_col = pick_col(df, ["asr", "success"])
        combo_col = pick_col(df, ["combo"])
        pars_col = pick_col(df, ["pars"], required=False)
        by_model = df.groupby([c for c in [model_col, arch_col] if c]).agg(Trials=(asr_col, "size"), ASR=(asr_col, "mean")).reset_index()
        by_model["ASR"] = (by_model["ASR"] * 100).round(2).astype(str) + "\\%"
        write_tex(by_model, "table_models.tex", "GhostPermit attack success rate by model.", "tab:models")
        by_combo = df.groupby(combo_col).agg(Trials=(asr_col, "size"), ASR=(asr_col, "mean"), PARS=(pars_col, "mean") if pars_col else (asr_col, "mean")).reset_index()
        by_combo["ASR"] = (by_combo["ASR"] * 100).round(2).astype(str) + "\\%"
        by_combo["PARS"] = by_combo["PARS"].round(3)
        write_tex(by_combo, "table_combos.tex", "GhostPermit attack success rate by permission combination.", "tab:combos")
    defense = DATA_DIR / "defense_results.csv"
    if defense.exists():
        df = pd.read_csv(defense)
        detector_col = pick_col(df, ["detector", "guard_model", "defense_model", "model"])
        detected_col = pick_col(df, ["attack_detected", "detected", "flagged", "blocked", "is_flagged"])
        out = df.groupby(detector_col)[detected_col].agg(["size", "sum", "mean"]).reset_index()
        out.columns = ["Detector", "Trials", "Detected", "DetectionRate"]
        out["DetectionRate"] = (out["DetectionRate"] * 100).round(2).astype(str) + "\\%"
        write_tex(out, "table_defense.tex", "Defense detection rates.", "tab:defense")

if __name__ == "__main__":
    main()
