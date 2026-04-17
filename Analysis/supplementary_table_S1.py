# ╔══════════════════════════════════════════════════════════════════╗
# ║  Supplementary Table S1 — Wilson 95% CIs for all 90 ASR cells  ║
# ║  Output: supplementary_table_S1.csv                             ║
# ║  Paper footnote: mean half-width ±14.5 pp                       ║
# ╚══════════════════════════════════════════════════════════════════╝
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

MODELS = [
    "llama-4-scout-17b", "llama-3.3-70b", "kimi-k2", "gemma-3-12b",
    "deepseek-r1-distill", "mistral-nemo", "qwen-qwq-32b", "phi-4", "gpt-4o"
]
COMBOS_LIST = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]

ASR_MATRIX = np.array([
    [96.7, 90.0, 60.0, 100.0, 100.0, 96.7, 90.0, 86.7, 93.3, 80.0],
    [30.0, 66.7, 13.3,  66.7,  66.7, 76.7, 46.7, 60.0, 60.0, 46.7],
    [10.0, 66.7,  0.0,  66.7,  46.7, 56.7, 33.3, 40.0, 33.3, 36.7],
    [10.0, 50.0,  6.7,  56.7,  43.3, 56.7, 33.3, 36.7, 43.3, 30.0],
    [13.3, 46.7,  3.3,  56.7,  43.3, 50.0, 30.0, 36.7, 33.3, 36.7],
    [13.3, 46.7,  6.7,  46.7,  40.0, 46.7, 23.3, 33.3, 30.0, 23.3],
    [56.7, 80.0, 33.3,  86.7,  83.3, 90.0, 60.0, 76.7, 73.3, 66.7],
    [13.3, 40.0,  3.3,  40.0,  30.0, 43.3, 20.0, 26.7, 26.7, 23.3],
    [26.7, 53.3, 13.3,  60.0,  53.3, 60.0, 36.7, 46.7, 46.7, 36.7],
])
N_TRIALS = 30

rows = []
for i, model in enumerate(MODELS):
    for j, combo in enumerate(COMBOS_LIST):
        asr = ASR_MATRIX[i, j]
        k   = round(asr / 100 * N_TRIALS)
        lo, hi = proportion_confint(k, N_TRIALS, alpha=0.05, method="wilson")
        hw = (hi - lo) * 50
        rows.append({
            "Model":      model,
            "Combo":      combo,
            "ASR_%":      asr,
            "CI_lo_%":    round(lo * 100, 1),
            "CI_hi_%":    round(hi * 100, 1),
            "CI_str":     f"[{round(lo*100,1)}, {round(hi*100,1)}]",
            "Half_width": round(hw, 1)
        })

df = pd.DataFrame(rows)
df.to_csv("supplementary_table_S1.csv", index=False)

mean_hw = df["Half_width"].mean()
max_hw  = df["Half_width"].max()
print(f"Supplementary Table S1 saved — {len(df)} cells")
print(f"Mean half-width : ±{mean_hw:.1f} pp")
print(f"Max  half-width : ±{max_hw:.1f} pp  (at ASR ≈ 50%)")
print()
print("Table III footnote (paste directly):")
print("  'All ASR values are proportions over n=30 trials.")
print("   95% Wilson CIs in supplementary Table S1")
print(f"   (mean half-width: ±{mean_hw:.1f} pp; max: ±{max_hw:.1f} pp at ASR≈50%).'")
