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

## Invariants I am holding myself to (from the project rules)

1. No identifier/timestamp column (`Flow ID`, IPs, ports, `Timestamp`) ever
   reaches a model. A test will enforce it.
2. Transformers are fit on the training split only; never compute a statistic
   over the full dataset.
3. The headline evaluation is the **temporal/by-day split**; the stratified
   number is reported only as an optimistic reference, with the gap called out.
4. Lead with PR-AUC, per-class P/R/F1, and TPR@fixed-FPR — never accuracy.
5. Every run is reproducible from logged config + seed.
