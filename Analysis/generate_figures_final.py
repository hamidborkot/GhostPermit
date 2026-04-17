# ╔══════════════════════════════════════════════════════════════════╗
# ║  Generate All Paper Figures — Final Publication-Ready Version   ║
# ║  Outputs: figure2_pars_sensitivity.png                          ║
# ║           figure3_agentworm_propagation.png                     ║
# ║           figure4_defense_evaluation.png                        ║
# ║  Run: python Analysis/generate_figures_final.py                 ║
# ╚══════════════════════════════════════════════════════════════════╝
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Figure 2: PARS Sensitivity ────────────────────────────────────────
combos    = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10"]
pars_base = [0.029,0.083,0.034,0.034,0.098,0.220,0.107,0.443,0.203,0.710]
pars_min  = [0.026,0.078,0.031,0.031,0.093,0.210,0.102,0.428,0.195,0.690]
pars_max  = [0.031,0.088,0.036,0.036,0.103,0.230,0.112,0.458,0.210,0.730]
mean_asr  = [22.6, 43.6, 22.6, 54.0, 49.9, 56.5, 36.5, 49.2, 44.6, 41.4]

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    name="PARS Score", x=combos, y=pars_base,
    marker_color=px.colors.sequential.Blues[3:],
    error_y=dict(
        type="data", symmetric=False,
        array=[hi - b for b, hi in zip(pars_base, pars_max)],
        arrayminus=[b - lo for b, lo in zip(pars_base, pars_min)],
        color="gray", thickness=1.5, width=4
    ),
    text=[f"{v:.3f}" for v in pars_base],
    textposition="outside", yaxis="y"
))
fig2.add_trace(go.Scatter(
    name="Mean ASR (%)", x=combos, y=mean_asr,
    mode="markers+lines",
    marker=dict(size=9, color="#e74c3c"),
    line=dict(color="#e74c3c", width=2),
    yaxis="y2"
))
fig2.update_layout(
    title={"text": "PARS Scores with Sensitivity Bounds vs Mean ASR"},
    xaxis_title="Permission Combo",
    yaxis=dict(title="PARS Score", side="left"),
    yaxis2=dict(title="Mean ASR (%)", side="right",
                overlaying="y", range=[0, 115]),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.05, xanchor="center", x=0.5),
)
fig2.write_image("figure2_pars_sensitivity.png")
print("figure2_pars_sensitivity.png saved")

# ── Figure 3: AgentWorm Propagation ──────────────────────────────────
hops_labels = ["Hop 0 (Seed)", "Hop 1", "Hop 2"]
exfil  = [93.3, 66.7, 93.3]
ci_lo  = [78.7, 48.8, 78.7]
ci_hi  = [98.2, 80.8, 98.2]

fig3 = make_subplots(
    rows=1, cols=2,
    subplot_titles=["Exfiltration Rate per Hop",
                    "Seed Re-propagation Rate"],
    horizontal_spacing=0.15
)
fig3.add_trace(go.Scatter(
    x=hops_labels, y=exfil,
    mode="markers+lines",
    name="Exfil Rate",
    marker=dict(size=14, color="#2980b9"),
    line=dict(width=3, color="#2980b9"),
    error_y=dict(
        type="data", symmetric=False,
        array=[hi - m for m, hi in zip(exfil, ci_hi)],
        arrayminus=[m - lo for m, lo in zip(exfil, ci_lo)],
        color="#2980b9", thickness=2, width=8
    )
), row=1, col=1)
fig3.update_yaxes(range=[0, 115], title_text="Rate (%)", row=1, col=1)
fig3.update_xaxes(title_text="Propagation Hop", row=1, col=1)
for h, v in zip(hops_labels, exfil):
    fig3.add_annotation(x=h, y=v + 7, text=f"{v}%",
                        showarrow=False, font=dict(size=13, color="#2980b9"),
                        row=1, col=1)
fig3.add_trace(go.Bar(
    x=["Hop 1", "Hop 2"], y=[50.0, 53.3],
    name="Re-propagation",
    marker_color=["#e67e22", "#e67e22"],
    text=["50.0%", "53.3%"], textposition="outside",
    width=0.4
), row=1, col=2)
fig3.add_hline(y=50, line_dash="dot", line_color="gray", row=1, col=2)
fig3.update_yaxes(range=[0, 115], title_text="Rate (%)", row=1, col=2)
fig3.update_xaxes(title_text="Propagation Hop", row=1, col=2)
fig3.update_layout(
    title={"text": "AgentWorm: LangChain Real Implementation (n=30 trials)"},
    showlegend=False
)
fig3.write_image("figure3_agentworm_propagation.png")
print("figure3_agentworm_propagation.png saved")

# ── Figure 4: Defense Evaluation ─────────────────────────────────────
defenses  = ["PromptGuard-2", "PARS-Gated", "Quarantine Filter", "CTIA Audit Log"]
precision = [0.0,  73.6, 90.8, 97.8]
recall    = [0.0,  43.3, 70.0, 81.9]
fpr       = [1.8,  39.6, 18.2,  4.6]

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    name="Precision (%)", x=defenses, y=precision,
    marker_color=["#e74c3c", "#f39c12", "#3498db", "#27ae60"],
    text=[f"{v:.1f}%" for v in precision],
    textposition="outside", width=0.25, offsetgroup=0
))
fig4.add_trace(go.Bar(
    name="Recall (%)", x=defenses, y=recall,
    marker_color=["#c0392b", "#d68910", "#2471a3", "#1e8449"],
    text=[f"{v:.1f}%" for v in recall],
    textposition="outside", width=0.25, opacity=0.75, offsetgroup=1
))
fig4.add_trace(go.Scatter(
    name="FPR (%)", x=defenses, y=fpr,
    mode="markers+lines",
    marker=dict(size=11, symbol="diamond", color="#7f8c8d"),
    line=dict(dash="dot", color="#7f8c8d", width=2),
))
fig4.update_layout(
    title={"text": "Defense Evaluation: Precision, Recall & FPR"},
    barmode="group",
    xaxis_title="Defense Mechanism",
    yaxis=dict(title="Rate (%)", range=[0, 120]),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.05, xanchor="center", x=0.5),
)
fig4.write_image("figure4_defense_evaluation.png")
print("figure4_defense_evaluation.png saved")
print("\nAll figures complete.")
