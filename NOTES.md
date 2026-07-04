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

## MITRE ATT&CK enrichment

- A class label ("DoS Hulk") isn't actionable; a tactic/technique ("Impact / T1499")
  is. Added a curated CIC-IDS2017 → ATT&CK mapping, returned live in the `mitre`
  field of every attack prediction and summarised in `netsentry intel` (12 classes →
  6 tactics, 8 techniques). One source of truth shared by serving and the report.
- Kept it honest: the mapping is **indicative** of the dataset's capture scenarios
  (CIC-IDS2017 isn't natively ATT&CK-labelled), stated plainly in the report and the
  README, and keyed on the *consolidated* model labels with raw web-attack variants
  aliased so `technique_for` works on either. A test asserts every attack class the
  model can emit has a mapping, so a new class can't silently ship unmapped.

## Statistical significance (bootstrap CIs)

- The project's whole claim is the *gap* between the honest temporal split and the
  optimistic stratified one. A single number invites over-reading, so the gap now
  carries a percentile-bootstrap CI and a p-value, and PR-AUC / TPR@FPR carry CIs.
- Got the comparison *shape* right: model-vs-baseline on one test set is **paired**
  (resample indices once, score both), but temporal-vs-stratified is across **two
  different test sets**, so it's an **independent** bootstrap (resample each, take the
  difference). Conflating the two is a common stats slip.
- Synthetic result: gap +0.257, 95% CI [+0.239, +0.276], p < 0.001, and the temporal
  PR-AUC CI [0.518, 0.541] excludes the majority baseline (0.250). The headline now
  reads as "significantly better than chance, and significantly worse than the
  optimistic split" — both halves of the honesty story, with uncertainty attached.
- A test caught an over-strict assertion: two independent draws from the *same*
  distribution still have a fixed finite-sample gap, so "no gap" isn't guaranteed for
  arbitrary seeds; the regression test compares a set against itself (symmetric about
  zero) instead.

## Per-class detection slices

- Broke the temporal detection number down by attack class to show *which* novel
  later-day attacks are caught. The honest read: **DDoS ~53%** (it behaves like the
  Mon-Wed DoS family the model trained on) but PortScan/Bot/Web Attack/Infiltration
  are near-zero — a clean known-vs-novel picture, and the per-class case for the
  benign-only anomaly detector.
- **Self-audit that fired.** First run showed 0% for *every* class, impossible given
  the 21% aggregate. Root cause: for the binary task `result.y_val` is already the
  0/1 target, but I re-derived it as `y_val.astype(str) != "BENIGN"` (always true for
  "0"/"1"), so validation looked all-attack and `threshold_at_fpr` returned +inf.
  Fixed to use the binary target directly; per-class now sums to the headline 21%.
  Also kept slices on the *raw* score (detection is a ranking property) to match the
  eval report's operating points and dodge isotonic's tie artifacts.

## API security hardening

- Added optional API-key auth + a per-client fixed-window rate limit on the
  prediction endpoints, config-gated so dev stays open and prod locks down via env.
- **Bug worth remembering.** I first wrote these as FastAPI dependencies with a
  `request: Request` param. Under `from __future__ import annotations` every
  annotation is a *string*, and FastAPI resolves it via the function's module
  globals — but FastAPI is imported *inside* the app factory, so `Request` wasn't
  resolvable and FastAPI treated `req` as a required **query** param (422 on every
  predict, even the batch contract test). Moved the checks into a Starlette
  middleware, which receives `request` positionally and needs no type resolution —
  clean fix, and `/health` + `/metrics` stay unguarded for probes.

## Serving integration of the new ML rigor

- The calibration / cost / conformal work was at risk of being "report-only". Wired
  it into the live API so it's product, not paperwork: the serving bundle now carries
  a `cost_optimal` threshold profile and class-conditional conformal thresholds (both
  fit on the *stratified* val split, where exchangeability makes the conformal
  guarantee valid), and `/predict` returns a `prediction_set` + `recommended_action`
  (auto_alert / auto_clear / review). `?profile=cost_optimal` is selectable next to
  the FPR profiles. The conformal set uses the calibrated attack probability, so the
  whole chain (calibrate → threshold/cost → conformal action) is consistent end to end.
- Kept it additive and safe: new response fields are optional (old clients unaffected),
  and the profile/conformal computation in the bundle build is wrapped so an extra
  operating point can never break the core artifact.

## Observability stack (Prometheus + Grafana)

- The API already exposed `/metrics`; this completes the loop with a Prometheus +
  Grafana compose profile and a pre-provisioned dashboard, so "it has metrics"
  becomes "here's the operations console". Added model-behaviour collectors
  (predictions-by-decision, anomaly count, attack-probability histogram) alongside
  the existing request/latency/drift metrics, all emitted best-effort so metrics can
  never break a prediction and all bounded-cardinality.
- Alert rules encode the concerns this project actually argues about: major input
  drift (PSI > 0.25) ties straight to the drift-monitoring work; an attack-flag-rate
  spike catches a broken/over-firing model; error-rate and a p99 latency SLO cover
  serving health. The drift alert closes the loop from the PSI gauges to an action.
- Couldn't run Docker in this environment, so the compose/Prometheus/Grafana YAML and
  the dashboard JSON are validated as parseable and the metric names are covered by a
  serving test that asserts they appear in `/metrics` after a prediction.

## Hyperparameter optimization (Optuna)

- The config carried `supervised.tune` / `tune_trials` and `optuna` was a declared
  dependency, but nothing used them. Wired it: `netsentry train tune` runs a TPE
  study and `train supervised` honours the flag.
- **Leakage discipline is the whole point here.** The pipeline is fit on train, each
  trial trains on train with early stopping on validation and is scored by validation
  PR-AUC, and the test split is never read during the search — tuning against test is
  the most common silent leak in "I got 0.99" write-ups. The search is also seeded
  (TPE sampler seed + a seeded random-search fallback) so a study reproduces.
- Output is a `configs/tuned.yaml` override (with `tune: false` baked in so reusing
  it doesn't recurse), keeping the workflow config-driven rather than mutating code.

## Conformal prediction & selective alerting

- Added class-conditional (Mondrian) split-conformal: a per-flow prediction set with
  a finite-sample, distribution-free coverage guarantee, with the four set shapes
  ({benign}/{attack}/both/empty) mapped to SOC actions (auto-clear, auto-alert,
  human-review, human-review-because-novel). The empty set is the conformal echo of
  the anomaly detector's "looks like nothing I trained on".
- **The result is the best kind of honest.** On the *exchangeable* stratified split
  the guarantee holds for both classes (benign 93.5%, attack 92.0% vs a 90% target).
  On the *temporal* split benign coverage holds (93.1%) but attack coverage collapses
  to 64.4% — because conformal assumes exchangeability and the temporal split breaks
  it on purpose (later-day attacks are novel). I almost shipped prose claiming
  "coverage meets the target"; caught that it didn't for attacks and reframed it: the
  shortfall is conformal correctly *detecting* the shift, so conformal coverage on a
  recent window is a drift signal that complements PSI. The unit test confirms the
  guarantee under exchangeability so the breakage is clearly attributable to drift.

## Cost-sensitive thresholds (SOC economics)

- A fixed-FPR threshold is honest but arbitrary. Attaching a cost to each outcome
  (analyst time per alert, expected loss per miss) makes the operating point a
  decision-theoretic optimum. The closed form for a *calibrated* probability —
  alert iff `p >= cost_per_alert/cost_per_miss` — is the clean tie-in to the
  calibration work: the threshold only means something if the score is a probability.
- **Two subtleties I had to get right rather than hand-wave:**
  1. The synthetic test split is ~22% attack, absurdly higher than production. Using
     that base rate makes "alert almost everything" optimal (degenerate). Fixed by
     costing at a configurable production prior (default 1%) via base-rate-reweighted
     conditional TPR/FPR — the correct way to extrapolate a test set to deployment.
     Under reweighting the optimal *single* threshold is no longer the closed form
     (the benign pool dominates the FP cost), so I scoped that claim instead of
     overstating it.
  2. The val-chosen cost-optimal threshold does **not** always win on test — on the
     synthetic run the fixed-1%-FPR point edges it, because validation (earlier days)
     and test (later days) differ. Rather than paper over the contradiction, the
     report states it: operating points drift on later-day traffic, the very effect
     the temporal split is built to expose. Re-tune on recent data in prod.

## Adversarial evasion robustness

- The model card listed "not adversarially robust — not evaluated against adaptive
  attackers" as a limitation. Turned that into a measurement: `netsentry
  robustness` runs a **mimicry** attack (interpolate the attacker-controllable
  features toward the benign centroid) and an **adaptive query search** (random-
  restart, L2-bounded, on controllable features only — the right tool since trees
  aren't differentiable), then plots detection vs effort.
- **The result is sobering and honest.** On the synthetic temporal model, mimicry
  drops detection 82.6% → 0% as the controllable features are moved fully toward
  benign; the adaptive search reaches ~10% detection at an L2 budget of 3. The
  most-exploitable single features are Flow Duration, packet counts, and flow
  rates — exactly the SHAP top features, which is the point: the model leans on
  spoofable volume/timing signal. This is the concrete case for the benign-only
  anomaly detector, not a footnote.
- **Self-audit that fired.** The first run produced perfectly flat curves
  (detection never moved). Treated "too clean to be true" as a bug and found it:
  `ColumnTransformer.get_feature_names_out()` prefixes names (`numeric__Flow
  Duration`), so the controllable-feature match found nothing and every attack was
  a silent no-op. A test (`mimicry at fraction 0 == baseline`, curves monotone)
  caught it; fixed with a prefix-stripping `base_feature_name`.

## Prequential streaming (drift → retrain → recover)

- The drift monitor *measures* decay; this closes the loop to the *action*. Replay
  the later-day (temporal test) flows as a time-ordered stream and compare a
  **static** model (frozen at deploy) against one **retrained** on each labeled
  batch, scored **prequentially** (interleaved test-then-train: score the batch,
  then learn from it). One operating threshold fixed on clean validation, so model
  freshness is the only moving part.
- **The result tells a genuinely rich story.** Mean batch PR-AUC rises 0.43 (static)
  → 0.54 (retrained). More telling than the mean is the *shape*: early-Thursday
  batches (rare Web Attack/Infiltration, few positives) are hard for both, but by the
  Friday batches the retrained model — having folded in the earlier novel attacks —
  hits ~0.90 PR-AUC while the static model plateaus at ~0.66. The retrained model
  literally learns the later-day attack types the frozen one never trained on.
- **The drift signal and the failure coincide, which is the whole point.** Per-batch
  model-score PSI starts *major* (0.55) on the earliest, most-novel batches and
  subsides as the stream approaches the training regime — so the batches where the
  static model is worst are exactly the ones the PSI alert would fire on. That is the
  closed loop: PSI rises → major-PSI batch is the retrain trigger (same threshold as
  the Prometheus drift alert) → retrain on recent labels → recover.
- Ties the batch together: retraining's *cost* is labels for the new attacks, which
  is precisely the analyst budget the active-learning study prices, and the recovery
  is the flip side of the ablation finding that later-day attacks don't transfer.

## Feature-group ablation

- SHAP says which features a *prediction* leaned on (attribution); it can't say what
  the model would lose if a whole family were gone, because a high-SHAP feature may
  be redundant with a fallback. Ablation answers that causal question: partition the
  CICFlowMeter stats into behavioural families (timing/IAT, flow rates, packet size,
  TCP flags, volume/counts, header/window) and refit with each family removed. Done
  on the fitted feature matrix (drop columns, refit model) so the leakage-safe
  pipeline and every other column's train-fit stats are untouched.
- **The result is the most interesting honest finding of this batch.** Removing
  **flow rates** collapses temporal PR-AUC 0.529 → 0.224 (detection 21% → 0.8%) — the
  honest signal leans overwhelmingly on the rate family. But removing **volume/counts
  *improved* PR-AUC to 0.641** and timing/IAT removal helped too. That is not "prune
  them"; it is the fingerprint of **overfitting to the temporal shift** — volume and
  duration magnitudes differ between the Mon–Wed training attacks and the Thu–Fri
  test attacks, so the model learns day-specific thresholds that mislead later, while
  the rate *ratios* transfer. A third independent view of the project's core thesis.
- **Credibility guard I put in deliberately.** "Removing X improves the test number"
  is one keystroke from "so I selected features on the test set" — the exact leakage
  the project is about. The report states plainly that acting on this means selecting
  on *validation*, never the test split; ablation only says where to look. Without
  that sentence the feature would undercut the whole project's thesis.
- Ties to robustness: the rate/timing families ablation shows carry the transferable
  signal are the attacker-controllable ones the evasion study exploits — the same
  argument for pairing the classifier with the benign-only anomaly detector, now from
  a second direction.

## Active learning (analyst labeling budget)

- The whole project frames the SOC's binding constraint as analyst time; this makes
  it literal on the *training* side. Labels are the scarce resource, so the question
  is label efficiency: from a small seed, does querying the flows the model is least
  sure about (**uncertainty sampling**, |p−0.5| smallest) beat labeling at **random**?
  Refit and score test after each round; the gap between the curves is time saved.
- **Deliberate split choice, and a clean tie-in to the thesis.** Ran it on the
  **stratified** split, because active learning assumes the pool and test are
  exchangeable — the same assumption conformal needs, and the same one the temporal
  split breaks on purpose. So AL is the reference-split technique and the report says
  so, rather than overclaiming it on the honest split. It's the training-time mirror
  of the conformal work: both order analyst attention by model uncertainty.
- **Leakage discipline:** only *labels* are scarce, so the unsupervised feature
  pipeline (impute/scale) is fit on the whole pool once — legitimate, and how you'd
  actually deploy — while the model only ever trains on the labeled subset.
- **Reporting subtlety I fixed in the loop.** My first efficiency metric ("labels to
  reach 98% of the best PR-AUC") landed in the *flat* top of both curves, so it
  reported "curves are close" and buried a real win. Switched to the standard AL
  metric — labels needed to reach the quality random only reaches at the *end* of its
  budget — which surfaces the honest result: uncertainty hits random's full-budget
  0.759 PR-AUC with 3,500 labels vs 4,500 (a 22% saving), and leads at every
  mid-budget round. The lesson: on a saturating curve, *where* you measure the gap
  decides whether you see it.

## Provenance & supply-chain (SBOM + model manifest)

- The auto model card answers "what is this model"; this answers the two questions
  a security reviewer or a deploy gate actually asks about a binary artifact that
  decides maliciousness: **what went into it** (a CycloneDX SBOM) and **has it been
  altered** (a SHA-256 model manifest + a `verify` integrity gate). It rounds out
  the governance story from "documented" to "attestable".
- **Design call: hand-built the CycloneDX 1.5 JSON rather than going through
  `cyclonedx-python-lib`** (which is installed). The library's model API churns
  across majors; a spec is stable. A dependency-free emitter that writes valid
  CycloneDX keeps the artifact durable and the code reviewable, and a test asserts
  the structure + `purl` form scanners key on. Scoped the SBOM to *declared*
  dependencies resolved to installed versions — a bounded, meaningful BOM — rather
  than the full transitive environment (hundreds of env-specific rows).
- **Manifest portability bug I caught in the loop.** First `verify` run failed:
  the manifest records only the bundle *name* (deliberately — absolute paths in a
  committed artifact are non-portable), so `verify_manifest` resolved it next to
  the manifest (docs/reports/) instead of models/. Kept `verify_manifest` pure
  (hash in, compare) and moved the "where do bundles live" knowledge into the CLI,
  which has settings — so it resolves the name against `models_dir`. Tampering a
  byte flips the hash and `verify` exits non-zero, which is the whole point.
- `netsentry verify` is a real CI/deploy gate: recompute the bundle SHA-256, fail
  loudly on a swap or corruption — the model-serving analogue of checking a package
  signature, and the natural consumer of the manifest the pipeline now emits.

## Training-set poisoning study

- The robustness study covers the inference-time adversary (evasion); this covers
  the training-time one. Two attacks, each aimed at the component it actually
  threatens: **label flips** (attack rows relabeled benign) against the supervised
  model, and **benign-pool contamination** (attack rows injected into the
  "benign-only" pool) against the anomaly detector. Degradation is always measured
  on the *clean* test ground truth while train/val carry the poison — the
  operator's real position, since they only ever hold the labels they were given.
- **The finding I did not expect, and the reason the study is worth shipping: PR-AUC
  lies here in the opposite direction from usual.** At a 50% flip rate PR-AUC barely
  moves (0.529 → 0.465) because it is a *ranking* metric on the raw score and
  boosting still orders attacks above benign — but detection at the operator's
  threshold collapses 21.0% → 1.8%. The threshold is chosen on the *poisoned*
  validation labels, so the flips move the operating point even where they leave the
  ranking intact. A study reporting only PR-AUC would call this model
  poison-resistant and be wrong about the number that ships. That is a clean second
  instance of the project's thesis (operating point ≠ ranking; report the one that
  matters), now in the security dimension.
- Anomaly contamination degrades detection 7.3% → 2.8% at 20% injection, via a
  double mechanism the report names: injected attacks widen the learned "normal"
  *and* the calibration quantile computed on the contaminated benign pool inflates
  the threshold. Note the realized benign FPR actually *drifts down* as
  contamination rises — the threshold is being set too high — which is the tell.
- Kept the prose sign-aware (three branches: ranking/operating-point split,
  PR-AUC-dominated, tolerant) so it renders a correct story on real data too, and
  tied the defences to components already here (validate gates + PSI drift), with
  the honest caveat that slow poisoning stays under both.

## Rules-vs-model baseline

- Every ML-IDS write-up implicitly compares against "no detection"; the real
  incumbent is a signature engine. Added a config-driven ruleset (six
  Suricata-style threshold rules, port-scoped like real signatures) and a report
  that compares it with the classifier **at a matched FP budget** — the model's
  threshold is chosen on validation at the FPR the ruleset actually spends, so
  neither system touches test before the comparison.
- **The honest result is not the one I expected: the rules *win* the single
  operating point** (21.1% vs 19.6% detection at ~0.5% FPR on the synthetic
  temporal test). It makes sense on inspection — the temporal test mix is
  dominated by DDoS + PortScan, precisely the two patterns with signatures, and
  PortScan is a Friday-only class the Mon–Wed-trained model has literally never
  seen. Reporting that instead of burying it is the point of the study.
- The per-class table is the real finding: the model catches DDoS at 49.6% (it
  generalises from the trained DoS family) but PortScan at 0.2%; the rules catch
  PortScan at 20.7% but Bot/Infiltration/Web Attack at exactly 0% (no signature →
  invisible). The hybrid (rules OR model) beats both at 24.3%. Complements, not
  rivals — and the sign-aware prose renders correctly whichever side wins.
- Also visible: signatures for attack classes not present in the traffic
  (ssh-bruteforce, slow-drip-dos) fire only on benign flows — dead rules don't
  just do nothing, they *spend the FP budget*. That is the maintenance-cost
  argument against rulesets, measured.

## Per-service detection parity (subgroups)

- The per-class slices answer "which *attacks* are caught"; a SOC also needs
  "does one global threshold treat each *service* fairly" — services are what
  alerts are routed by, and the attack class is unknown when the alert fires.
  Added `netsentry subgroups`: temporal-test flows grouped by the service implied
  by `Destination Port`, with per-service detection and FPR at the single global
  threshold. It is an equalized-odds fairness audit transplanted to security.
- The port is the one field the project pointedly *drops* from the model (the
  memorisation leak), which is exactly what makes it safe here: it only labels
  the slice, never enters a prediction. Grouping by a deliberately-excluded
  column to audit the model is a nice closure of the port-leakage story.
- First draft over-read the FPR spread (0.57–1.11% across services around the 1%
  budget). At ~2–4k benign flows per service that spread is mostly binomial
  noise, so I added Wilson 95% intervals to every rate and made the prose
  interval-aware: it only claims a service "genuinely exceeds the budget" when
  its whole interval sits above it (on the current data IMAP straddles, so the
  report says so). The structural point survives regardless — a global threshold
  constrains only the aggregate FPR; nothing pins any single service's queue.
- The detection side needs no hedging: HTTP-bound attacks are caught at 42.1%
  [40.4, 43.9] while ephemeral-port traffic (PortScan spray + Infiltration) sits
  at 0.3% [0.2, 0.6] — non-overlapping intervals, consistent with the per-class
  slices, and stated as signal. Alert-share per service ("IMAP alone raises 27%
  of all false positives") is the queue-level view of alert fatigue.

## Novelty distance (the split gap, decomposed)

- The project's spine is the temporal-vs-stratified gap; this study builds the
  instrument that says what the gap is *made of*. Distance from each test attack
  to its nearest training attack (in the pipeline's standardized space) is a
  direct novelty measure; binning detection by it on shared quantile edges for
  both splits separates two stories that "shuffled splits leak" conflates:
  **composition** (the shuffled split's test attacks sit near training twins) and
  **at-distance shift** (later days are harder even at matched novelty).
- The decomposition is a one-line counterfactual: apply stratified per-bin
  detection to the temporal distance mix. On the stand-in it says composition is
  ~nothing (-1.1 pts) and at-distance is ~everything (+27.8 pts of the 26.7-pt
  gap) — which is *correct*, because the generator draws flows independently and
  so has no burst near-twins. The report says that plainly and states the
  real-data expectation (twin bar ≈ the leakage) instead of pretending the
  stand-in showed it.
- My first draft's prose assumed the classic story (nearer = flatter,
  detection decays with distance). The data said otherwise on both counts: the
  medians nearly coincide (6.87 vs 7.15) and detection **rises** with distance in
  both splits (+20/+24 pts) — far-from-training attacks are volumetric extremes
  that are easy *because* they are extreme; the hard attacks hug the benign
  manifold, exactly where the evasion study's mimicry pushes. I rewrote the
  render with three-way sign-aware branches (material-difference margins, not
  bare `<`) so the report can never claim a mechanism its own table contradicts.
- Also fixed the figure: quantile bins have a heavy-tailed last range whose
  midpoint squashed everything left; x is now bin index and the table carries the
  ranges. Reference/query caps and the twin epsilon are config
  (`novelty.*`), seeded subsampling keeps it deterministic.

## Leave-one-day-out (temporal sensitivity)

- The ml.md rules name two honest temporal designs ("train on earlier days,
  test on later days, OR do leave-one-day-out"); only the first existed. Added
  the second as `netsentry lodo`, reusing `temporal_split` per fold with
  rotated `train_days`/`test_days` so the val-carve and disjointness discipline
  is inherited rather than re-implemented.
- Two structural facts make the study more than an error bar. Each CIC-IDS2017
  attack class lives on exactly one capture day, so holding a day out removes
  its classes from training entirely — every fold is zero-shot class detection.
  And Monday is benign-only, so its fold is a pure quiet-day false-alarm audit:
  0.94% realized FPR ≈ 9.4k alerts/day, the number a SOC pays on the days when
  nothing happens (most days).
- The spread is the finding: novel-family detection runs 1.5% (Thursday:
  Web Attack/Infiltration — subtle, no behavioural cousins in training) to
  25.4% (Wednesday: the DoS family, which generalises from Friday's DDoS and
  vice versa), mean 13%. Every rotation supports the headline conclusion
  (novel families are hard at a fixed FP budget); which families are hard is
  the per-family difficulty profile the fixed cut can only show once.
- `fold_metrics` returns NaN (not 0.0) for undefined sides — a benign-only
  day's "detection" must not read as "caught nothing"; `rates_at_threshold`'s
  0.0 convention would have. Unit tests lock the NaN semantics.

## Invariants I am holding myself to (from the project rules)

1. No identifier/timestamp column (`Flow ID`, IPs, ports, `Timestamp`) ever
   reaches a model. A test will enforce it.
2. Transformers are fit on the training split only; never compute a statistic
   over the full dataset.
3. The headline evaluation is the **temporal/by-day split**; the stratified
   number is reported only as an optimistic reference, with the gap called out.
4. Lead with PR-AUC, per-class P/R/F1, and TPR@fixed-FPR — never accuracy.
5. Every run is reproducible from logged config + seed.
