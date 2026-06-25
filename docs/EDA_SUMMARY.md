# EDA Summary

> **Data note.** The concrete numbers below come from the **synthetic**
> CIC-IDS2017 stand-in (`netsentry/data/synthetic.py`, seed 42, 40k rows), used
> so the analysis is reproducible without the licensed download. They illustrate
> the *methodology and the shape of the problem*; they are **not** real-world
> results. Re-run `notebooks/01_eda.ipynb` on the real dataset to reproduce the
> equivalent figures. The modelling implications hold for the real data too.

## Class distribution — severe imbalance
- ~**78% benign**, ~22% attack overall (13 classes after consolidating the three
  `Web Attack` variants).
- The tail is brutal: in a 40k sample, `Infiltration` ≈ 30 rows and `Heartbleed`
  ≈ 9 rows, versus `DoS Hulk` ≈ 2.4k and `PortScan` ≈ 2.1k.
- **Implication:** accuracy is meaningless (predicting all-benign scores ~78%).
  Lead with **PR-AUC, per-class recall, and TPR@fixed-FPR**. Use **class
  weights**, and expect rare-class recall to be low and high-variance — report it
  honestly rather than hiding it in a macro average alone.

## Missingness — concentrated and explainable
- NaNs appear almost entirely in `Flow Bytes/s` and `Flow Packets/s` (~2%, from
  Inf on zero-duration flows) and a little in `Flow IAT Std` (~0.5%).
- **Implication:** impute with the **median, fit on train only**, inside the
  feature pipeline. Do not drop rows (the Inf flows are real, short flows).

## Feature signal — present but not separable
- The features most associated with "is attack" are flow-rate and volume
  statistics: `Flow Packets/s`, `Flow Bytes/s`, `SYN Flag Count`, and the packet
  counts (point-biserial |r| ≈ 0.2–0.32). No single feature is close to a perfect
  separator.
- **Implication:** a gradient-boosted tree should do well but **not** report
  ~99.9%. If it does, **suspect leakage** and investigate before believing it.

## `Destination Port` — the leakage trap, made visible
- Attack rate by destination port is wildly uneven: ~**0.62 on port 80** versus
  ~**0.00 on ports 25/110/143/993**. A model given the raw port can "detect"
  attacks by memorising which ports the capture happened to use.
- **Implication:** **drop `Destination Port` from the headline model.** Train a
  second variant that keeps it (as a categorical) and report the gap — that gap
  is part of the leakage story, not a win.

## Takeaways for modelling
1. Metrics: PR-AUC + per-class P/R/F1 + TPR@{0.1%, 1%} FPR; never accuracy alone.
2. Imbalance: class weights over default resampling; watch rare-class recall.
3. Pipeline: median impute (train-fit), scale, drop identifiers, drop port from
   the headline variant.
4. Honesty gate: a too-good number is a bug — audit for leakage first.
