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

## Invariants I am holding myself to (from the project rules)

1. No identifier/timestamp column (`Flow ID`, IPs, ports, `Timestamp`) ever
   reaches a model. A test will enforce it.
2. Transformers are fit on the training split only; never compute a statistic
   over the full dataset.
3. The headline evaluation is the **temporal/by-day split**; the stratified
   number is reported only as an optimistic reference, with the gap called out.
4. Lead with PR-AUC, per-class P/R/F1, and TPR@fixed-FPR — never accuracy.
5. Every run is reproducible from logged config + seed.
