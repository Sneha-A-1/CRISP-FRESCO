"""Generates the five figures of the CRISP + FRESCO paper (300 dpi)."""

from __future__ import annotations
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = f"{ROOT}/results", f"{ROOT}/figures"
CW = 3.4                                  # IEEE column width, inches
plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2, "axes.linewidth": 0.6,
    "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "legend.fontsize": 6.3,
    "axes.labelsize": 7.4, "axes.titlesize": 7.6, "savefig.dpi": 300,
    "figure.dpi": 300, "lines.linewidth": 1.0, "legend.frameon": False,
})
GREY = "#4d4d4d"
POL = ["FCFS", "EDF", "LSF", "AOIG", "FRESCO"]
PNAME = {"FCFS": "FCFS", "EDF": "EDF", "LSF": "LSF",
         "AOIG": "AoI-greedy", "FRESCO": "FRESCO"}
PCOL = {"FCFS": "#9e9e9e", "EDF": "#4c72b0", "LSF": "#55a868",
        "AOIG": "#e08214", "FRESCO": "#c0392b"}
PMK = {"FCFS": "v", "EDF": "s", "LSF": "^", "AOIG": "D", "FRESCO": "o"}


def _s():
    return json.load(open(f"{RES}/summary.json"))


# ----------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(CW, 3.05))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11.6); ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=6.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                                    linewidth=0.6, edgecolor=GREY, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                linespacing=1.25)

    def arr(x1, y1, x2, y2, ls="-", c=GREY, lw=0.7):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=lw,
                                     linestyle=ls, color=c))

    box(0.15, 9.25, 9.7, 2.15, "", "#f2f2f2")
    ax.text(0.35, 11.18, "LAYER 1  IoT sensing layer", fontsize=6.6,
            fontweight="bold", va="top")
    for k, lab in enumerate(["soil EC\n$\\tau$=180 min", "soil moisture\n$\\tau$=90 min",
                             "air temp.\n$\\tau$=25 min", "leaf wetness\n$\\tau$=20 min"]):
        box(0.45 + 2.36 * k, 9.45, 2.05, 1.25, lab, "white", fs=5.5)
    ax.text(0.45, 8.62, "update $(x, t_{gen})$, age $a$", fontsize=5.2,
            ha="left", va="center", style="italic")

    for x in (2.0, 5.0, 8.0):
        arr(x, 9.25, x, 8.2)

    box(0.15, 6.15, 9.7, 2.0, "", "#e8e8e8")
    ax.text(0.35, 7.93, "LAYER 2  CRISP freshness-aware classifier",
            fontsize=6.6, fontweight="bold", va="top")
    box(0.45, 6.32, 2.9, 1.20, "VoI decay\n$V=e^{-a/\\tau}$  (1)", "white", fs=5.5)
    box(3.55, 6.32, 2.9, 1.20, "refresh value\n$\\Gamma$  (4), budget $B$", "white", fs=5.5)
    box(6.65, 6.32, 2.9, 1.20, "guard band\n$\\tau_{ad}$  (3)", "white", fs=5.5)
    arr(3.35, 6.92, 3.55, 6.92); arr(6.45, 6.92, 6.65, 6.92)
    ax.text(6.15, 8.62, "on-demand refresh poll", fontsize=5.2, ha="left")
    arr(6.0, 8.30, 6.0, 9.25, ls=(0, (2, 1.4)), c="#7a7a7a")

    arr(2.6, 6.15, 2.6, 5.05); ax.text(2.72, 5.55, "urgent", fontsize=5.4)
    arr(7.9, 6.15, 7.9, 2.35, ls=(0, (2, 1.4)))
    ax.text(8.02, 4.5, "non-urgent\n+ predicted\noverflow", fontsize=5.3)

    box(0.15, 2.95, 6.6, 2.1, "", "#dedede")
    ax.text(0.35, 4.83, "LAYER 3  FRESCO fog scheduler", fontsize=6.6,
            fontweight="bold", va="top")
    box(0.45, 3.12, 2.8, 1.28, "age-weighted\npriority  (5)", "white", fs=5.5)
    box(3.45, 3.12, 2.9, 1.28, "forecast\nadmission  (6)", "white", fs=5.5)
    arr(3.25, 3.76, 3.45, 3.76)
    ax.text(7.0, 3.76, "actuation", fontsize=5.4, va="center")
    arr(6.75, 3.76, 6.95, 3.76)

    box(0.15, 0.25, 9.7, 1.95, "", "#f2f2f2")
    ax.text(0.35, 1.98, "LAYER 4  Cloud layer", fontsize=6.6,
            fontweight="bold", va="top")
    for k, lab in enumerate(["overflow\nprocessing", "seasonal\nanalytics",
                             "$\\tau$ re-estimation"]):
        box(0.45 + 3.15 * k, 0.42, 2.9, 1.1, lab, "white", fs=5.5)

    arr(0.15, 1.0, 0.15, 10.2, ls=(0, (1.6, 1.2)), c="#7a7a7a")
    ax.text(0.04, 5.6, "decay constants", fontsize=5.2, rotation=90,
            va="center", color="#5a5a5a")

    fig.tight_layout(pad=0.15)
    fig.savefig(f"{FIG}/fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig2_classification():
    s = _s()["classification"]
    names = ["THRESH", "NB", "GB", "CRISP"]
    lab = ["threshold", "naive Bayes", "grad. boosting", "CRISP"]
    metrics = ["precision", "recall", "f1", "accuracy"]
    mlab = ["Precision", "Recall", "F1-score", "Accuracy"]
    hatch = ["", "//", "..", "xx"]
    cols = ["#d9d9d9", "#a6a6a6", "#737373", "#c0392b"]

    x = np.arange(4); w = 0.2
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for i, n in enumerate(names):
        v = [s[n][m] for m in metrics]
        ax.bar(x + (i - 1.5) * w, v, w, label=lab[i], color=cols[i],
               edgecolor="k", linewidth=0.4, hatch=hatch[i])
    ax.set_xticks(x); ax.set_xticklabels(mlab)
    ax.set_ylim(0.70, 1.04); ax.set_ylabel("Score")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.04),
              handlelength=1.1, columnspacing=0.7, handletextpad=0.35,
              fontsize=5.6)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig2_classification.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig3_peak_aoi():
    s = _s()["scheduling"]
    rates = sorted(int(k) for k in s)
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for p in POL:
        y = [s[str(r)][p]["peak_aoi_s"] for r in rates]
        ax.plot(rates, y, PMK[p] + "-", color=PCOL[p], ms=3.0,
                label=PNAME[p], mfc="none" if p != "FRESCO" else PCOL[p])
    ax.set_yscale("log")
    ax.set_xlabel("Task arrival rate (tasks/s)")
    ax.set_ylabel("Peak AoI (s)")
    ax.legend(ncol=2, loc="upper left", handlelength=1.7)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig3_peak_aoi.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig4_latency_util():
    s = _s()["scheduling"]
    rates = sorted(int(k) for k in s)
    fig, ax = plt.subplots(1, 2, figsize=(CW * 2.02, 1.95))
    for p in POL:
        ax[0].plot(rates, [s[str(r)][p]["latency_ms"] for r in rates],
                   PMK[p] + "-", color=PCOL[p], ms=3.0, label=PNAME[p],
                   mfc="none" if p != "FRESCO" else PCOL[p])
        ax[1].plot(rates, [s[str(r)][p]["stale_viol_pct"] for r in rates],
                   PMK[p] + "-", color=PCOL[p], ms=3.0,
                   mfc="none" if p != "FRESCO" else PCOL[p])
    ax[0].set_yscale("log")
    ax[0].set_xlabel("Task arrival rate (tasks/s)")
    ax[0].set_ylabel("Mean task latency (ms)")
    ax[0].set_title("(a) latency")
    ax[1].set_xlabel("Task arrival rate (tasks/s)")
    ax[1].set_ylabel("Staleness violations (%)")
    ax[1].set_title("(b) freshness SLA violations")
    ax[0].legend(ncol=5, loc="upper center", bbox_to_anchor=(1.1, 1.30),
                 handlelength=1.5, columnspacing=1.0)
    for a in ax:
        a.grid(alpha=0.25, lw=0.4, which="both")
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(top=0.80)
    fig.savefig(f"{FIG}/fig4_latency_staleness.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig5_scalability():
    s = _s()
    nodes = sorted(int(k) for k in s["scalability"])
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for p in POL:
        y = [s["scalability"][str(n)][p] for n in nodes]
        ax.plot(nodes, y, PMK[p] + "-", color=PCOL[p], ms=3.0,
                label=PNAME[p], mfc="none" if p != "FRESCO" else PCOL[p])
    ax.set_xlabel("Number of fog nodes")
    ax.set_ylabel("Execution time (s)")
    ax.set_xticks(nodes)
    ax.legend(ncol=2, loc="upper right", handlelength=1.7)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig5_scalability.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
def fig6_staleness_sweep():
    s = _s()["staleness_sweep"]
    ages = sorted(int(k) for k in s)
    names = ["THRESH", "NB", "GB", "CRISP"]
    lab = ["threshold", "naive Bayes", "grad. boosting", "CRISP"]
    cols = ["#000000", "#7f7f7f", "#4c72b0", "#c0392b"]
    mk = ["s", "^", "D", "o"]
    fig, ax = plt.subplots(figsize=(CW, 1.95))
    for i, n in enumerate(names):
        ax.plot(ages, [s[str(a)][n] for a in ages], mk[i] + "-", color=cols[i],
                ms=3.0, label=lab[i], mfc="none" if n != "CRISP" else cols[i])
    ax.set_xlabel("Mean information age at decision time (min)")
    ax.set_ylabel("F1-score (urgent class)")
    ax.legend(ncol=2, loc="lower right", handlelength=1.7)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.2)
    fig.savefig(f"{FIG}/fig6_staleness_sweep.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig1_architecture(); fig2_classification(); fig3_peak_aoi()
    fig4_latency_util(); fig5_scalability(); fig6_staleness_sweep()
    print("figures written to", FIG)
