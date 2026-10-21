# CRISP + FRESCO: freshness-aware classification and age-driven fog scheduling

Reference implementation for *"Freshness-Aware Data Classification and
Age-Driven Fog Scheduling for Time-Critical Smart Agriculture IoT"*.

The premise: a fog-cloud framework that answers in 20 ms using a 40-minute-old
soil reading is fast and wrong. Both algorithms treat **information age** as a
first-class quantity.

* **CRISP** - Criticality and Refresh-Interval Scoring with Prioritisation.
  Every stream carries a value-of-information decay constant. CRISP converts
  age into lost decision confidence, then spends a bounded budget of
  on-demand refresh polls on exactly the items whose urgency verdict is both
  close to an agronomic bound and badly aged.
* **FRESCO** - Freshness-aware Elastic Scheduling with Cloud Offloading.
  Priority = criticality x residual freshness + starvation-free ageing term +
  slack term. A short-horizon arrival forecast spills predicted overflow
  (non-urgent only) to the cloud before the burst lands.

## Quick start

```bash
pip install -r requirements.txt
python run_all.py            # ~12 min
```

Outputs land in `results/` (CSV + `summary.json`) and `figures/` (300 dpi PNG).
Every number quoted in the paper comes from this script.

## Layout

| path | contents |
|---|---|
| `src/data.py` | soil/environmental streams, staleness model, urgency labels |
| `src/crisp.py` | CRISP plus threshold, naive Bayes and gradient-boosting baselines |
| `src/scheduler.py` | fog simulator: FCFS, EDF, LSF, AoI-greedy, FRESCO; AoI accounting |
| `src/experiments.py` | classification, staleness sweep, rate sweep, node scalability, Wilcoxon |
| `src/figures.py` | Figs. 1-6 of the paper |
| `results/` | generated CSVs and `summary.json` |
| `figures/` | generated figures |

## Data

* **Dataset 1** - L. Belli et al., "Smart agriculture dataset in a tomato
  cultivation under different irrigation regimes", *Data in Brief* 60, 111521
  (2025), doi:10.1016/j.dib.2025.111521; raw files doi:10.17632/35wh56287y.1
* **Dataset 2** - Azure Functions Trace 2019,
  <https://github.com/Azure/AzurePublicDataset> (Shahrad et al., USENIX ATC
  2020)

**Important.** The default run uses `data.synthesize()`, whose statistics are
calibrated to Dataset 1; `data.load_real()` reads the downloaded CSVs. See
`data/README.md`. Re-run on the real files before quoting the numbers as
measurements on the published data.

## Headline results (10 seeds, chronological 70/15/15 split)

Classification on 7,776 held-out items:

| Method | Precision | Recall | F1 | Accuracy | Missed urgent |
|---|---|---|---|---|---|
| Fixed threshold | 0.890 | 0.890 | 0.890 | 0.926 | 10.98 % |
| Naive Bayes | 0.752 | 0.808 | 0.778 | 0.845 | 19.20 % |
| Gradient boosting | 0.912 | 0.874 | 0.893 | 0.929 | 12.61 % |
| **CRISP** | **0.955** | **0.944** | **0.949** | **0.966** | **5.57 %** |

CRISP refreshes 12 % of items. Fog scheduling at 30 tasks/s, 6 nodes:

| Policy | Peak AoI (s) | Mean AoI (s) | Latency (ms) | Urgent (ms) | Exec (s) | Deadline miss |
|---|---|---|---|---|---|---|
| FCFS | 18.20 | 8.85 | 8307.5 | 8182.7 | 130.2 | 85.6 % |
| EDF | 18.25 | 8.73 | 8307.5 | 7028.5 | 130.2 | 85.4 % |
| LSF | 15.93 | 4.68 | 8307.5 | 357.9 | 130.2 | 63.2 % |
| AoI-greedy | 5.02 | 1.14 | 8307.5 | 8305.8 | 130.2 | 34.9 % |
| **FRESCO** | **3.61** | **0.98** | **2758.6** | **353.4** | **119.0** | **21.2 %** |

Wilcoxon signed-rank: p = 0.002 for FRESCO vs EDF and vs AoI-greedy on peak
AoI and latency; p = 0.031 for CRISP vs gradient boosting on F1.


## Citation

Sneha, P. D. Singh, V. Tripathi, "Freshness-Aware Data
Classification and Age-Driven Fog Scheduling for Time-Critical Smart
Agriculture IoT" (submitted).

## License

MIT - see `LICENSE`.
