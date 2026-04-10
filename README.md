# GhostPermit: Cross-Permission Data Exfiltration in LLM Agents

<div align="center">

[![Paper](https://img.shields.io/badge/IEEE%20TIFS-Under%20Review-blue?style=flat-square)](https://ieee-tifs.org)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Models](https://img.shields.io/badge/Models-9%20LLMs-orange?style=flat-square)](#models-evaluated)
[![Trials](https://img.shields.io/badge/Trials-4%2C329%2B-red?style=flat-square)](#experiments)

**GhostPermit: Cross-Permission Data Exfiltration via Prompt Injection in Multi-Tool LLM Agents**

*MD. Hamid Borkot Tulla*\\
*Submitted to IEEE Transactions on Information Forensics and Security (TIFS), 2026*

</div>

---

## Overview

**GhostPermit** is the first systematic study of **cross-permission data exfiltration** in multi-tool LLM agents. We demonstrate that a malicious prompt injected into one permitted data source can silently direct the agent to harvest and compile sensitive data from *all* other granted tools — calendar, contacts, location, notes, and email — into a single exfiltration payload, without the user's knowledge.

This repository is organized around the paper's full research workflow: attack construction, evaluation, defense analysis, reproducibility artifacts, and analysis notebooks. The code and data are structured so reviewers can reproduce the main claims from saved outputs, or re-run the experiments when API access is available.

### Contributions

- 🔴 **GhostPermit Attack** — a prompt injection that exploits cross-tool trust in LLM agents.
- 📐 **PARS** (Permission-weighted Attack Risk Score) — a quantitative risk metric for multi-tool agents.
- 🐛 **AgentWorm** — a self-propagating variant that spreads across tool stores.
- 🔵 **CoT Hijacking** — an attack corrupting Chain-of-Thought reasoning mid-inference.
- 🛡️ **Defense Evaluation** — LlamaGuard 22M and 86M both fail to detect GhostPermit (0% detection).

### Key Results

| Metric | Value |
|---|---|
| Highest ASR (llama-4-scout-17b) | **89.3%** |
| GPT-4o ASR (closed-source) | **71.5%** |
| LlamaGuard detection rate | **0%** (22M and 86M) |
| AgentWorm 3-hop propagation | **74.8%** |
| CoT hijacking success rate | **68.3%** |
| Models evaluated | **9** (5 architecture types) |
| Total experimental trials | **4,329+** |
| Human judge Cohen's κ | **1.000** |

---

## Repository Structure

```
GhostPermit/
│
├── README.md                        ← This file
├── LICENSE                          ← MIT License
├── CITATION.cff                     ← Citation metadata (GitHub standard)
├── requirements.txt                 ← Python dependencies
│
├── attack/                          ← Attack implementations
│   ├── ghostpermit_attack.py        ← GhostPermit core injection logic
│   ├── agentworm.py                 ← Self-propagating worm variant
│   ├── cot_hijacking.py             ← Chain-of-Thought hijacking
│   └── prompts/
│       ├── ghostpermit_prompt.txt   ← Injection prompt template
│       ├── agentworm_prompt.txt     ← Worm propagation payload
│       └── cot_hijack_prompt.txt    ← CoT corruption payload
│
├── evaluation/                      ← Evaluation pipeline
│   ├── run_experiment.py            ← Main experiment runner (all models)
│   ├── judge.py                     ← Keyword-based ASR judge
│   ├── pars.py                      ← PARS metric computation
│   ├── utility_eval.py              ← Utility evaluation (200 benign tasks)
│   └── judge_validation.py          ← Cohen's κ validation script
│
├── defense/                         ← Defense evaluation
│   ├── llamaguard_eval.py           ← LlamaGuard 22M + 86M evaluation
│   └── defense_baseline.py          ← Keyword-filter baseline comparison
│
├── data/                            ← All experimental results
│   ├── README_data.md               ← Data schema and description
│   ├── ghostpermit_results.csv      ← Main GhostPermit results (2,535 trials)
│   ├── agentworm_results.csv        ← AgentWorm propagation (1,154 trials)
│   ├── cot_hijacking_results.csv    ← CoT hijacking results (240 trials)
│   └── defense_results.csv          ← LlamaGuard defense results (400 trials)
│
├── analysis/                        ← Figures and tables
│   ├── generate_figures.py          ← Reproduce all paper figures
│   ├── generate_tables.py           ← Reproduce all paper tables (LaTeX)
│   └── statistical_tests.py         ← Kruskal-Wallis + Mann-Whitney tests
│
└── notebooks/                       ← Jupyter notebooks
    ├── 01_main_results.ipynb        ← GhostPermit ASR analysis
    ├── 02_pars_analysis.ipynb       ← PARS metric deep-dive
    ├── 03_agentworm_analysis.ipynb  ← Worm propagation analysis
    └── 04_defense_analysis.ipynb    ← Defense evaluation notebook
```

---

## Paper Structure

The manuscript follows a seven-part research skeleton that maps directly onto the repository:

1. **Introduction and motivation** — why multi-tool agents create a new trust boundary.
2. **Background and related work** — prompt injection, tool use, and prior agent-security studies.
3. **Threat model** — attacker capability, defender assumptions, and scope.
4. **Attack design** — GhostPermit, PARS, AgentWorm, and CoT Hijacking.
5. **Experimental setup** — models, permission combinations, judge, and metrics.
6. **Evaluation and results** — ASR, propagation, defense failure, and statistical tests.
7. **Discussion and conclusion** — implications, limits, and defenses.

---

## Models Evaluated

| Model | Provider | Architecture | Trials | ASR |
|---|---|---|---|---|
| llama-4-scout-17b | Meta / Groq | Dense / Reasoning | 300 | **89.3%** |
| gpt-4o | OpenAI / GitHub | Closed / Reasoning | 200 | **71.5%** |
| llama-3.3-70b | Meta / Groq | Dense / Non-Reason | 300 | 72.9% |
| nvidia/llama-3-70b | NVIDIA / Groq | Dense / Non-Reason | 300 | 70.9% |
| gpt-oss-120b | OpenAI / GitHub | Dense / Non-Reason | 267 | 37.5% |
| qwen3-32b | Alibaba / Groq | Dense / Non-Reason | 299 | 21.4% |
| gpt-oss-20b | OpenAI / GitHub | Dense / Non-Reason | 269 | 20.5% |
| kimi-k2 | Moonshot / Groq | MoE / Non-Reason | 300 | 19.0% |
| zai/GLM | Zhipu / GitHub | Closed / Non-Reason | 300 | 0.3% |

---

## Permission Combinations & PARS Scores

PARS (Permission-weighted Attack Risk Score) = weighted sensitivity × combo coverage × exfiltration factor

| ID | Tool Combination | PARS |
|---|---|---|
| C1 | calendar + notes | 0.17 |
| C2 | contacts + email | 0.43 |
| C3 | location + notes | 0.23 |
| C4 | calendar + contacts | 0.22 |
| C5 | calendar + contacts + notes | 0.29 |
| C6 | calendar + contacts + email | 0.48 |
| C7 | contacts + location + notes | 0.50 |
| C8 | calendar + contacts + location + email | 0.75 |
| C9 | calendar + contacts + location + notes | 0.56 |
| C10 | calendar + contacts + location + notes + email | **0.90** |

---

## Setup & Reproduction

### 1. Clone and Install

```bash
git clone https://github.com/hamidborkot/GhostPermit.git
cd GhostPermit
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
export GROQ_API_KEY="your_groq_api_key"
export GITHUB_MODELS_KEY="your_github_token"
```

### 3. Reproduce Results from Saved Data (Recommended)

```bash
# Regenerate all paper figures from saved CSVs
python analysis/generate_figures.py --from-data

# Regenerate all LaTeX tables
python analysis/generate_tables.py --from-data

# Run statistical significance tests
python analysis/statistical_tests.py
```

### 4. Re-run Full Experiments (requires API keys + ~8 hours)

```bash
# GhostPermit: 9 models × 10 combos × 30 trials
python evaluation/run_experiment.py --experiment ghostpermit --all-models

# AgentWorm propagation
python evaluation/run_experiment.py --experiment agentworm

# CoT Hijacking
python evaluation/run_experiment.py --experiment cot

# Defense evaluation
python defense/llamaguard_eval.py
```

### 5. Validate Judge

```bash
python evaluation/judge_validation.py
# Output: Cohen's κ = 1.000, Specificity = 100%, FP = 0
```

---

## Judge Validation

The keyword-based ASR judge was independently validated:

| Metric | Value |
|---|---|
| Validation samples | 40 (20 safe, 20 unsafe) |
| Human-judge agreement | 100.0% |
| Cohen's κ | **1.000** |
| False positives | 0 |
| False negatives | 0 |
| Specificity (200 benign tasks) | **100.0%** |

---

## Ethical Statement

> This research was conducted under strict ethical guidelines.

1. **No real user data** was used. All experiments used synthetic agent environments with simulated tool outputs.
2. **No production systems** were attacked. All experiments ran in isolated sandboxed API sessions.
3. **Responsible disclosure** — findings were communicated to the relevant model providers prior to submission.
4. **Purpose** — this research aims to expose and remediate a critical LLM agent vulnerability class to advance the safe deployment of AI systems.

This paper does not provide sufficient detail to enable direct reproduction of the attack against real deployed systems beyond what is already known in the prompt injection literature.

---

## Citation

```bibtex
@article{borkot2026ghostpermit,
  title     = {GhostPermit: Cross-Permission Data Exfiltration via Prompt
               Injection in Multi-Tool LLM Agents},
  author    = {Borkot Tulla, {MD. Hamid}},
  journal   = {IEEE Transactions on Information Forensics and Security},
  year      = {2026},
  note      = {Under Review},
  url       = {https://github.com/hamidborkot/GhostPermit}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

> ⚠️ **Responsible Use Notice:** The attack code in this repository is released for academic and defensive research purposes only. Misuse against real systems without authorization is illegal and unethical.

---

<div align="center">
<sub>IEEE Transactions on Information Forensics and Security (TIFS) · Submitted 2026</sub>
</div>
