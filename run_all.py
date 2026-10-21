#!/usr/bin/env python3
"""Reproduce every table and figure of the CRISP + FRESCO paper."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import experiments, figures

if __name__ == "__main__":
    print("[1/3] classification, staleness sweep, scheduling, scalability ...")
    s = experiments.main()
    print("[2/3] figures ...")
    figures.fig1_architecture(); figures.fig2_classification()
    figures.fig3_peak_aoi(); figures.fig4_latency_util()
    figures.fig5_scalability(); figures.fig6_staleness_sweep()
    print("[3/3] done. results/ and figures/ updated.")
    c = s["classification"]
    print("  F1: threshold %.3f | GB %.3f | CRISP %.3f"
          % (c["THRESH"]["f1"], c["GB"]["f1"], c["CRISP"]["f1"]))
    k = s["scheduling"]["30"]
    print("  peak AoI @30/s: EDF %.2f s -> FRESCO %.2f s"
          % (k["EDF"]["peak_aoi_s"], k["FRESCO"]["peak_aoi_s"]))
