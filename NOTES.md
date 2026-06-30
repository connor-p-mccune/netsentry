# Engineering notes

A running log of decisions, surprises, and self-audits. The guiding habit:
**treat a too-good number as a bug** — when a metric looks great, stop and
investigate leakage before celebrating, and write down what was found.

## Phase 0 — scaffolding & tooling

- **Dependency strategy.** The core install is intentionally light (numpy,
  pandas, scikit-learn, pydantic, typer, matplotlib). The heavy ML stack
  (LightGBM, SHAP, MLflow, imbalanced-learn, Optuna, seaborn, and PyTorch) lives
  in optional extras (`train`, `ae`). Every module that uses them imports lazily
  and has a graceful fallback, so the package imports and the fast test path is
  green even without them.
  - *Designed-in fallbacks:* LightGBM → scikit-learn `HistGradientBoosting`;
    SHAP → permutation importance; MLflow → local JSON run log; autoencoder →
    Isolation Forest only.
- **Surprise (good one).** The local interpreter is Python 3.14, newer than the
  3.11 target. I expected some heavy wheels (especially PyTorch) to be missing,
  but a probe showed **the entire stack installs on 3.14**. So the local demo can
  use the real premium models, while CI still pins 3.11/3.12 for portability.
- **Config-first.** All knobs (seed, paths, split strategy, model
  hyperparameters, FPR targets) live in `configs/default.yaml` and the typed
  `Settings` model. No magic numbers are allowed in code.
- **Reproducibility primitive.** `seed_everything` seeds Python, NumPy, and (if
  present) Torch from one config seed; estimators get the same seed via
  `random_state`.

## Phase 2 — cleaning & EDA

- The synthetic EDA shows the shape we expect: ~78% benign, a long rare-class
  tail (Heartbleed/Infiltration in the dozens), missingness concentrated in the
  rate columns (Inf→NaN), and moderate feature signal (top |corr| ≈ 0.3). No
  single feature separates attacks — which is the honest baseline we want.
- `Destination Port` attack-rate is wildly uneven (≈0.62 on port 80, ≈0 on
  several others). That is the leakage trap made visible: it justifies dropping
  the port from the headline model.

## Phase 3 — splits & the leakage firewall

- **Surprise worth stating loudly.** With the temporal split (train Mon–Wed, test
  Thu–Fri), the attack *classes* are largely **disjoint across the boundary**:
  Patator + DoS + Heartbleed are in train; Web Attack + Infiltration + Bot +
  PortScan + DDoS are in test. A multiclass classifier literally cannot *name* a
  class it never saw.
  - **Consequence for framing:** the temporal HEADLINE is naturally a **binary**
    (attack vs benign) *generalization* test — can the model flag later-day,
    partly-novel attacks at all? "Name the attack" (multiclass per-class) is the
    honest job of the **stratified** reference split. And this disjointness is
    precisely why the unsupervised **anomaly detector** (Phase 6) earns its keep.
- **Firewall:** the feature `ColumnTransformer` uses `remainder="drop"`, so only
  explicitly-listed feature columns are ever modelled; a test injects identifier
  columns and asserts they vanish from `get_feature_names_out()`.
- **Fit-on-train-only** is structural: the single `Pipeline` is fit on the train
  split, and a test asserts the imputer's learned median equals the *train*
  median, not the combined one. Validation is always carved from train.

## Phase 4 — baseline & supervised model

Synthetic-data results (seed 42), binary attack-vs-benign:

| split | model PR-AUC | majority baseline | macro-F1 |
|---|---|---|---|
| **temporal** (honest headline) | **0.478** | 0.253 | 0.62 |
| stratified (optimistic ref) | 0.729 | 0.221 | 0.78 |

- **This is the thesis in one table.** The optimistic stratified split looks ~50%
  better than the honest temporal split. That gap is the leakage/over-optimism
  story other CIC-IDS projects hide.
- **Honesty checks pass:** the model beats its majority baseline on both splits
  (so it genuinely learns), and *nothing* is suspiciously high — no metric near
  1.0, so no leakage to chase. (Real CIC-IDS2017 numbers will differ; these are
  the synthetic stand-in.)
- **Determinism** is enforced (LightGBM `deterministic=True`; a test trains twice
  and asserts identical probabilities).
- **Tracking robustness:** recent MLflow gates the file store, so `track_run`
  opts in via `MLFLOW_ALLOW_FILE_STORE` and, crucially, falls back to a local
  JSON run log on *any* MLflow error — tracking can never break a training run.

## Phase 5 — evaluation framework

- The report leads with **operational** metrics. On the synthetic temporal split,
  the operating points are sobering and honest: ~**2.4% detection at a 0.1% FP
  budget**, rising to ~13% at a 1% budget (~11k false alerts/day). Low detection
  at a tight FP budget is the *expected* shape for cross-day-type generalisation —
  and exactly why the anomaly detector (Phase 6) and threshold tuning matter.
- The report renders the temporal-vs-stratified PR-AUC gap (+0.250) front and
  centre, with figures (PR/ROC/threshold/confusion) saved to `docs/figures/`.
- Metric correctness is unit-tested on hand-computed confusion matrices so an
  averaging/off-by-one bug can't silently invalidate every downstream number.

## Phase 6 — anomaly detection (novel attacks)

- Leave-one-attack-out (synthetic): benign-only detectors catch ~**10% of a
  never-seen attack class at a 1% benign FP budget** (Isolation Forest and the
  autoencoder land close). Modest — detecting truly unseen behaviour at a tight
  FP budget is genuinely hard, and saying so is the honest thing.
- **The ensemble result is the payoff.** On the temporal test (later-day, partly
  novel attacks), PR-AUC is: supervised-only 0.478, anomaly-only 0.393,
  **combined 0.506** — the rank-mean ensemble beats *both*. That is the argument
  for pairing a classifier with a benign-only detector: neither alone covers both
  known and novel regimes.
- The autoencoder standardises on benign-train stats and early-stops on a benign
  holdout; thresholds for both detectors are calibrated to a benign FP budget, so
  "how much benign noise fires" is an explicit operator choice.

## Phase 7 — explainability

- SHAP `TreeExplainer` on the headline LightGBM model powers both a global
  importance figure (in the eval report) and top-k per-prediction contributions
  (returned by the API in Phase 8). A flagged flow without a reason isn't
  actionable, so explanations are treated as part of the contract.
- Robustness: recent SHAP returns a *list* of arrays for binary LightGBM; the
  reducer handles list / 2-D / 3-D outputs. If SHAP isn't installed, the
  explainer degrades to model feature importances so the API still returns
  `top_features` (a documented approximation).

## Phase 8 — serving

- The served bundle is the **multiclass** model (so it can name attacks) plus a
  benign-fit Isolation Forest for the anomaly score. The honest temporal-split
  numbers remain the reported headline; the served model is the namer/scorer.
- **Live latency** (single flow, local, SHAP per request): p50 ≈ 47 ms, p95 ≈
  56 ms, p99 ≈ 76 ms, ~21 req/s single-process. SHAP is the dominant cost — the
  expected trade-off for per-prediction explanations.
- **Contract coherence fix.** `is_attack` is the actionable thresholded decision;
  at the strict 0.1%-FPR profile a flow can have a high `attack_probability` yet
  not be flagged. To avoid a confusing "named an attack but is_attack=false"
  response, `predicted_class` now agrees with the decision (benign when not
  flagged), while `attack_probability` is still reported for transparency.
- Input is validated against the real feature columns (unknown key or non-numeric
  value -> 422); missing features are imputed by the fitted pipeline.

## Phase 9 — containerization & CI

- Multi-stage, slim, non-root images: a lean `serve` image (serving + LightGBM +
  SHAP, no MLflow/Torch) and a full `train` image. The serve entrypoint builds a
  synthetic demo bundle on first start, so `docker compose up` works standalone.
- CI gains a **train-smoke** job (full `download → prep → train → eval → anomaly`
  on a tiny synthetic dataset via `configs/ci.yaml`) plus the slow integration
  tests, a non-blocking `pip-audit`, and a serving-image build.
- The fast `quality` job runs on `[dev]` only — the graceful fallbacks
  (hist_gbdt, importance-based explanations) mean the full non-slow suite passes
  without LightGBM/SHAP/Torch installed.
- Caveat: Docker can't run in this build environment, so the Dockerfiles are
  written to best practice and the compose/CI YAML and the exact smoke commands
  were validated locally; the image build itself is exercised in CI.

## Phase 10 — docs & canonical results

The committed report/figures and the README headline come from one canonical run
on the **default 60k-row synthetic** dataset (seed 42). These supersede the
smaller dev-run numbers noted in earlier phases:

| Metric (synthetic) | Value |
|---|---|
| Binary PR-AUC — temporal (honest) | 0.529 (baseline 0.250) |
| Binary PR-AUC — stratified (optimistic) | 0.786 |
| Over-optimism gap | +0.257 |
| Detection @ 0.1% / 1% FPR (temporal) | 9.1% / 21.0% |
| Anomaly LOAO avg detection @ 1% FPR | 8.5% (AE), 4.3% (iForest) |
| Ensemble vs best single (temporal PR-AUC) | 0.537 vs 0.529 |
| Latency p50 / p95 | ~47 / ~56 ms |

- The story holds at scale: the honest temporal split is ~0.26 PR-AUC below the
  optimistic shuffled split, nothing is suspiciously near 1.0, and the autoencoder
  out-detects the Isolation Forest on novel attacks while the ensemble edges past
  either alone.
- These are **synthetic** numbers (clearly labelled everywhere); the real
  CIC-IDS2017 run uses the identical commands.

## Stretch S5 — ONNX export + quantization

- `netsentry onnx` exports the gradient-boosted classifier to ONNX (via skl2onnx +
  onnxmltools; the `ai.onnx.ml` opset is pinned to 3 because the LightGBM converter
  emits v5) and benchmarks ONNX Runtime vs the Python path.
- **Honest results (synthetic, 2000-flow batch):** probabilities match sklearn to
  ~1e-7 (100% argmax agreement); ONNX Runtime is ~1.4x the Python throughput (76k
  vs 53k flows/s). **Dynamic quantization is a no-op for trees** — a
  `TreeEnsembleClassifier` has no quantizable matmul weights, so the quantized model
  is the same size and not faster (slightly slower here). The report states this
  plainly rather than claiming a quantization win; quantization is the lever for the
  autoencoder, not the trees.

## Stretch S3 — Streamlit demo dashboard

- `netsentry demo` (or `streamlit run netsentry/demo/dashboard.py`) serves a live
  dashboard: pick/edit a flow → verdict, attack probability, anomaly score, and the
  SHAP top features, reusing the exact `InferenceEngine` the API uses.
- Structured for testability: the data layer (`demo/core.py`: sample flows +
  `predict_flow`) is Streamlit-free and unit-tested; the UI (`dashboard.py`) is
  verified headless with Streamlit's `AppTest` (builds a tmp bundle, asserts the
  app renders without exception). `streamlit` is an optional `demo` extra.

## Stretch S4 — vulnpipe integration

- `netsentry triage` re-ranks vulnerability findings by fused risk: base severity
  (or CVSS) + NetSentry attack probability + anomaly flag. The point is to connect
  the two projects — vulnpipe finds holes, NetSentry says which are being leaned on.
- On the sample findings, a **critical** CVE on a quiet host (db01) drops below two
  **high** CVEs on hosts with attack-like traffic, and a **medium** CVE on an
  attacked host nearly ties the critical. That is the intended behaviour: triage by
  what's being exploited, not CVSS alone. Weights live in config (`triage.*`).
- `VulnFinding` is a documented contract; real vulnpipe output wires in by mapping
  severity/CVSS + asset and attaching the host's network-flow features.

## Stretch S1 — cross-dataset generalization

- Added `netsentry crosseval`: score the CIC-trained bundle, unchanged, on a
  synthetic **NetFlow-schema** foreign dataset adapted into CIC features (most
  features absent → imputed; detection transfers only through shared rate/volume
  behaviour). In-domain vs cross is contrasted honestly.
- **A self-audit caught a too-good result.** The first foreign generator made
  attacks trivially separable (4-10x volume, very short flows) → cross PR-AUC ≈
  0.96, *higher* than in-domain. That is exactly the "if it looks too good,
  investigate" rule firing: the stand-in's attacks were a giveaway, not a
  generalization win. Retuned to modest, overlapping signal (mirroring the main
  synthetic generator) and made the report prose **sign-aware**, so it can never
  contradict its own numbers and flags a too-good cross score as a stand-in artifact.
- **The honest read now:** PR-AUC 0.529 → 0.517 (ranking transfers via the few
  shared features) but TPR@0.1%FPR collapses 11.9% → 1.2% — the operating point
  does *not* transfer across schemas. Precisely the nuance a real UNSW-NB15 study
  would surface; the synthetic magnitude is illustrative, the method is the point.

## Stretch S2 — drift monitoring (PSI)

- Added `netsentry/monitoring`: a Population Stability Index implementation
  (quantile-binned, reference-fit), a `DriftReport`, a `netsentry drift` report
  (feature + model-score PSI, default temporal train-vs-test), and a rolling
  in-serving monitor that exports `netsentry_feature_drift_psi_max`/`_mean` gauges.
- **The honest read on the synthetic data:** across the Mon-Wed → Thu-Fri
  boundary, max feature PSI ≈ 0.11 and **score drift ≈ 0.16** (both "moderate").
  That is the same phenomenon the temporal split punishes, now measured directly —
  later-day traffic really does drift, so a model tuned on earlier days is partly
  extrapolating. Input/score drift is the signal you watch in production to catch
  this *before* the labels (and the damage) arrive.
- The serving monitor is deliberately bounded and safe: a tumbling window, two
  gauges (no per-feature label cardinality), and every step wrapped so monitoring
  can never break a prediction. The reference travels inside the bundle, so a
  deployed model self-monitors without the processed dataset.

## Probability calibration

- `ml.md` §4 requires that any threshold/probability claim be backed by
  *calibrated* probabilities, and the config carried `thresholds.calibrate` /
  `calibration_method` — but nothing fit a calibrator. That was a real gap: the
  served `attack_probability` and the FPR thresholds were raw LightGBM scores,
  which rank well but are not probabilities. Closed it.
- The calibrator is a monotonic 1-D map (isotonic by default, Platt/sigmoid
  optional) fit on the **validation** attack scores and applied to both the served
  probability and the decision thresholds (which now live on the calibrated scale,
  so serving calibrates before comparing).
- **The subtlety worth stating:** I claimed "calibration is monotonic so ranking
  metrics are unchanged" — then a test caught that *isotonic* introduces ties, so
  PR-AUC can move by a hair (Platt is strictly monotone and is exactly invariant).
  Fixed the claim to be precise rather than convenient: calibration preserves the
  *ordering* of flows, so the model's discriminative power is untouched; only the
  score→probability map changes. The headline PR-AUC is computed on the raw score
  and is unaffected regardless.
- Synthetic temporal-split result: isotonic calibration improves every diagnostic
  (Brier 0.175 → 0.171, ECE 0.121 → 0.106, MCE 0.315 → 0.138). The big MCE drop is
  the point — the worst-case over-confident bin is roughly halved.

## Invariants I am holding myself to (from the project rules)

1. No identifier/timestamp column (`Flow ID`, IPs, ports, `Timestamp`) ever
   reaches a model. A test will enforce it.
2. Transformers are fit on the training split only; never compute a statistic
   over the full dataset.
3. The headline evaluation is the **temporal/by-day split**; the stratified
   number is reported only as an optimistic reference, with the gap called out.
4. Lead with PR-AUC, per-class P/R/F1, and TPR@fixed-FPR — never accuracy.
5. Every run is reproducible from logged config + seed.
