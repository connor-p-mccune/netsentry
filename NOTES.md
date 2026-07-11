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

## Label-noise audit (find it, don't assume it)

- The poisoning study prices label corruption; this finds the corrupted rows —
  the pair mirrors Engelen et al.'s point that CIC-IDS2017 needed a corrected
  re-release at all. Implementation is confident-learning distilled to the
  binary case: out-of-fold scores (StratifiedKFold on the temporal *train* split
  only; the test split is never touched) and class-conditional mean thresholds.
  No new dependency — the Northcutt idea is a dozen lines here.
- The audit validates itself by planting flips, because a noise detector never
  tested against known noise is just an opinion. Recovery on the stand-in:
  58.8% recall, 19.8% precision — and the precision number is where honesty
  earned its keep twice. First, 19.8% must be read against the 1.2% planted
  base rate: a 16x concentration, i.e. a triage multiplier, not an oracle (the
  render computes the lift and refuses the triage framing below 2x). Second,
  the intrinsic flags (3.2% of benign-labeled, 18.1% of attack-labeled rows)
  are on labels that are correct by construction — they are the method's
  ambiguity floor caused by the generator's deliberate class overlap, and my
  first template called them "likely mislabeled" before I rewrote it. The same
  rows are the ones the per-class slices show being missed, which is a
  satisfying cross-check: hard-to-classify and hard-to-audit coincide.
- Two-pass design (audit as recorded, then plant-and-recover) doubles the k-fold
  cost; `label_audit.max_rows` caps it and the fold count is config.

## Per-service thresholds in serving (audit -> product)

- The subgroups report ends by saying per-service thresholds would pin each
  queue to its budget where one global cut cannot; leaving that as prose felt
  like stopping one step short, so the serving layer now ships it:
  `?profile=per_service`. The bundle builder calibrates a threshold per service
  on the same calibrated validation scores as every other profile; inference
  routes each flow to its service's cut.
- The leakage story stays intact and is the design's spine: `Destination Port`
  was already an accepted request field (it is in the schema's feature columns)
  that the model pipeline drops — so it rides as routing metadata and never
  enters a prediction. The port picks *which validation-calibrated threshold
  applies*; it contributes nothing to the score. Same rule as the audit, now
  enforced by a shared `data/services.py` map so the audit and serving cannot
  drift apart (the old private copy in subgroups.py moved there; `__all__`
  re-exports keep the audit's public surface unchanged).
- Failure modes decided conservatively: no port in the request, a service the
  bundle has no entry for (support floor `subgroups.min_support`, one-class
  validation traffic, or a budget no finite threshold can meet at the service's
  support - sklearn's roc_curve returns an inf sentinel there, which would have
  silently disabled detection for the service and broken strict JSON; caught on
  inspection of the first built bundle) -> the profile's global threshold. The
  profile degrades to the global cut; it never guesses. Storing the global
  under `bundle.thresholds["per_service"]` makes the profile selectable through
  the existing profile-validation path with zero app changes.

## Adversarial hardening (close the loop the robustness report opens)

- The robustness report measures the evasion weakness and ends by *naming*
  adversarial training as a direction; leaving it there felt like stopping one step
  short, exactly as the per-service note describes for the parity audit. So `harden`
  implements it: augment training with mimicry-perturbed attack rows (the attacker's
  own move toward the benign centroid, computed on train benign only - no leak),
  refit, and re-run the *same* evasion study on baseline vs hardened.
- The design decision that keeps it honest: calibration and the FPR thresholds are
  fit on the **clean** validation split for both models, so the two operating points
  are the same kind of thing. The only variable between them is the injected rows.
- The stand-in result is almost too clean (full-mimicry detection 0% -> ~100%),
  because the hardened model literally trains on fraction-1.0 mimicry examples. That
  is why the report leads with the **trade-off** (clean PR-AUC 0.529 -> 0.519) and
  states plainly that adversarial training only defends the perturbation it trains on
  - the standing argument for the anomaly detector. Reporting the win without the
  cost would have been the "too-good number" the whole project is a rebuke to.

## Statistical drift detectors (significance and timing, not just PSI magnitude)

- PSI is an effect size with a rule-of-thumb cutoff; it has no p-value and runs on
  static batches. `driftscan` adds a per-feature KS test with Benjamini-Hochberg FDR
  (so "5/76 drifted" is multiplicity-corrected, not 5% of stable features flagged by
  chance) and two online detectors (Page-Hinkley on the score stream, DDM on the
  error stream) that report *when*.
- **DDM warmup was the real lesson.** First cut fired drift at index 29 of a 10k
  stream: an all-correct warmup locks a (0,0) baseline, and `0 >= 0` trips the alarm
  on the first error. Guarded by not arming until `p > 0` *and* `n >= min_samples`.
  Even then DDM is genuinely jumpy at small n (the cumulative error rate is volatile
  and the 3-sigma band tightens as the stream grows), so `ddm_min_samples` is a
  substantial 2000 on the real-data stream - established empirically by sweeping it
  (200 -> spurious warning; 2000 -> warning 6013 / drift 7133, both cleanly past the
  5000 boundary). The unit test uses evenly-spaced errors so the baseline is stable
  by construction and the assertion is deterministic.
- **Which model to monitor** mattered. The temporal model's score mean does not rise
  on later days (novel attacks score *lower*), so Page-Hinkley on it never fired; the
  deployed stratified serving bundle is the honest subject anyway (you monitor what is
  deployed), and its score distribution *does* shift at the boundary. Both detectors
  now locate the same later-day boundary the temporal split embodies.

## ATT&CK Navigator layer (coverage in the framework the SOC already uses)

- `navigator` writes a real Navigator layer JSON, scored by support-weighted per-class
  recall. The one judgement call: **which split**. The temporal headline split only
  contains later-day classes, so it cannot color the whole matrix; per-class recall is
  only well-posed where every class appears in test, which is the **stratified**
  reference split (the same split NOTES already assigns the "name the attack" job).
  The split and operating point are written into the layer description + metadata, so
  the artifact is self-describing rather than quietly mixing splits.
- Tactic shortnames live next to the existing ATT&CK mapping (not a second copy in the
  navigator module), the same single-source-of-truth discipline the port->service map
  got, so the layer and the served `mitre` field cannot drift.

## Alert-queue capacity planning (detection per unit of analyst time)

- Distinct from `cost`: cost picks the expected-cost-minimising threshold; this reads
  detection off a *fixed* analyst budget. A budget of K alerts/day is the ROC point
  whose alert volume equals K at the production base rate, so recall/precision/lift
  come straight from the score ranking - no new model, just the operating-point
  arithmetic the cost report already reweights to a realistic 1% prior.
- The honest number is the **lift over random** (~50-60x on the stand-in): recall
  divided by random triage's K/flows hit rate. Precision is reported at the production
  prior, not the 22% test mix, so it reflects the benign-heavy queue an analyst really
  faces. Below ~500 alerts/day detection is 0% on the hard temporal split - reported
  as-is, because a capacity plan that hides its floor is worthless.

## Feature-importance stability (can you trust the shipped explanations?)

- The API returns SHAP top-features as a contract and the eval report shows a global
  ranking - both from a *single* fit. `importance` audits whether that ranking is
  signal or sampling luck: refit on bootstrap resamples, recompute importance, measure
  how much it moves (mean pairwise Spearman of the vectors; Jaccard of the top-k sets).
- The stand-in result is the honest kind: full-ranking Spearman ~0.40 (noisy) but
  top-10 Jaccard ~0.59 (the leaders mostly hold). My first `_verdict` collapsed that to
  a blanket "unstable", which under-reads it - the real, common pattern is *the tail is
  noise, the head holds*, so the verdict now says so and ties it to why the API returns
  only the top few features. A weak-signal iid generator (top |corr| ~0.3, no dominant
  feature) is exactly where the low-importance tail should reshuffle; on real CIC data
  with stronger drivers the Spearman would be higher, and the audit would say that too.
- Importance per refit uses the model's gain importances (LightGBM) with a model-
  agnostic permutation fallback for the HistGB path, so the audit runs on either
  backend. Kept the pure `stability_metrics` separate from the refitting so it unit-
  tests against hand-built matrices (identical -> 1.0; reversed -> negative Spearman,
  zero Jaccard) without training anything.

## Seed sensitivity (the error bar bootstrap CIs cannot see)

- First cut computed PR-AUC on the *calibrated* test scores and got 0.514 where the
  headline says 0.529 — isotonic calibration creates ties, and ties change average
  precision. Not a bug in either number, but a report whose rows don't line up with
  the eval report invites exactly the wrong kind of doubt, so the audit now mirrors
  eval's conventions (raw ranking for PR-AUC; thresholds re-chosen on each run's own
  raw validation scores). Seed 42's row now reproduces the headline exactly.
- The measured floor is small (PR-AUC sd 0.0017 across seeds 42-46) and data noise
  dominates (bootstrap half-width 0.0116) — worth knowing *before* choosing promotion
  margins, which is the point: the margins are evidence, not taste.
- Reproducibility and stability are asserted separately: same-seed refits are
  bit-identical (max score delta 0.0), different seeds move metrics by the floor.

## Release gate (the definition of done, as an exit code)

- The gate failed its own first run: I set `max_ece: 0.10` from taste, and the
  measured post-calibration ECE on the temporal test is 0.1057 — a validation-fit
  calibrator honestly degrades under temporal shift (raw is 0.1206). The bar moved
  to 0.15 with the reasoning written into the config docstring. Exactly the loop the
  gate exists to force: a bar you can't justify is a bar you tune with evidence.
- The inverted ceiling (PR-AUC > 0.999 **fails**) is the project's too-good-is-a-bug
  habit as machinery. The leakage firewall is re-checked on the *fitted artifact* at
  release, not only in unit tests — a config drift (e.g. shipping an experiment with
  the port encoded) passes every unit test and fails this gate.
- Hygiene stated in the report: a release gate touches the frozen test split, so it
  runs at release cadence, not per-commit; production runs the bars on a fresh window.

## Champion/challenger promotion (and the HOLD that earned its keep)

- The margins come from the seed audit (~3x sd for PR-AUC, ~2.5x for TPR), and the
  comparison is a *paired* bootstrap — one resample scores both models, so shared
  sampling noise cancels; the CI is several times tighter than the independent one.
  Detection is compared at each bundle's own validation-chosen threshold, because
  that is what each model would actually ship.
- The first real decision was a **HOLD**, and it is the best possible demo: a routine
  seed-43 retrain is PR-AUC-equivalent (+0.0001, CI [-0.0022, +0.0025]) yet credibly
  worse at the 0.1%-FPR operating point (-0.0149, CI [-0.0188, -0.0117], p = 1.000).
  A ranking metric said "same model"; the operating point said "ships 1.5pp less
  detection". The gate held the champion. That is the evaluation thesis (ranking vs
  operating point) resurfacing at the deployment layer, unprompted.
- Two policies because two questions: `non_inferiority` rolls routine retrains
  forward (freshness has measured value under drift — the streaming study);
  `superiority` demands the CI exclude zero for risky swaps. The champion is a
  SHA-256-pinned *snapshot*, so a later retrain overwriting the working bundle path
  cannot silently rewrite it; every decision appends to a JSONL history.

## Retrain triggers (the PSI alarm pulls the lever - and under-delivers)

- The drift-triggered policy captured essentially none of the retraining headroom
  (mean batch PR-AUC 0.413 vs the 0.534 every-batch ceiling; even the calendar
  periodic-3 hit 0.474 with the same two retrains) - and that is the finding, kept,
  not smoothed over. The trigger fires when later-day traffic first arrives, the
  redeploy resets its reference, PSI goes quiet, and it never fires again - while
  labeled retraining keeps buying quality for five more batches.
- The lesson generalises: PSI watches the score *distribution*, and a distribution
  can settle while quality is still being bought. An unsupervised trigger is a
  cost-saver against covariate shift, not a substitute for labels; the honest
  deployment pairs it with a periodic labeled cadence. My first render assumed the
  trigger would win and the text contradicted the numbers - rewritten with branches
  for win / partial / under-delivery so the report can never gaslight its own table.
- Faithfulness details that made the simulation defensible: prequential scoring,
  one validation-chosen threshold for every policy, and each policy's drift signal
  from its *own* deployed model with a reference that resets on redeploy (the same
  mechanics as the bundle-embedded serving reference).

## Behavioral canaries + shadow challenger (assurance at the serving edge)

- `verify` proves the artifact's bytes; nothing proved the *runtime*. Canaries close
  that: the bundle embeds a class-mixed handful of raw validation flows with its
  build-time calibrated scores, and load-time (plus `netsentry canary`, exit-coded)
  must reproduce them. NaN feature values round-trip as None. A missing canary exits
  2, distinct from a failing one - a check you didn't run is not a check you passed.
- Caught by my own CI wiring: the champion snapshot is a copy of the *training*
  bundle, which didn't embed canaries - so the deploy gate would have exit-2'd on
  the exact artifact promotion ships. Fixed by embedding canaries in every persisted
  training bundle (persist path only; the report generators' fit path pays nothing).
- The shadow challenger is `promote`'s evidence source moved to live traffic: a
  second bundle scores every request silently (champion answers, shadow is
  measured), emitting a score-delta histogram and a decision-disagreement counter,
  each model at its own threshold. Failure isolation is explicit - a broken shadow
  disables itself rather than taxing the champion. The integration test uses an
  identical-copy shadow and asserts the disagreement counter provably stays zero.

## Surrogate distillation (behavior compresses; the ranking does not)

- First cut distilled the *calibrated* score and the teacher row read 0.514 — the
  same isotonic-ties trap the seed audit hit. Switched the target to the raw
  ranking: the calibrator is monotone, so `calibrator(surrogate)` serves identical
  semantics, rank fidelity is well-behaved (no tie plateaus), and the teacher row
  now matches the headline 0.529 on sight. Same lesson twice, now a memoried rule:
  any new report's teacher/benchmark numbers must be computed on the scale the
  evaluation report uses.
- The stand-in result splits cleanly and honestly: decision agreement at matched
  alert volume climbs to 97.5% by 49 leaves while Spearman stalls at 0.61 and
  PR-AUC pays 0.529 -> 0.451. Read: the model's coarse alert/no-alert behavior fits
  in a page of rules; its fine ranking - which is what PR-AUC and tight budgets
  price - does not. Both numbers are needed to say that; either alone misleads.
- The visibly non-monotone surrogate TPR column (16.6% at depth 4 between 21.1%
  and 24.7%) is leaf-score quantization at a 1% budget, not noise - a K-leaf tree
  emits K scores, so operating points move in jumps. Stated in the report so nobody
  ships "the interpretable version" expecting the teacher's threshold granularity.
- Rendered rules strip the `numeric__` transformer prefixes; an auditor reads
  `Total Fwd Packets <= 0.28` (standardized units, said so), not pipeline plumbing.

## Packet ingestion (the wire, without a capture dependency)

- The whole capture stack is stdlib `struct` on purpose. scapy/dpkt would parse
  more protocols, but classic pcap + Ethernet/IPv4/TCP/UDP covers what the CIC
  features can express, and a zero-dependency reader keeps the core install
  promise ("runs anywhere") intact. pcapng raises a clear "convert first" error
  rather than a parse failure; per-packet garbage is counted and skipped — a NIDS
  ingest that dies on a malformed frame would be an irony too far.
- Feature fidelity is anchored to `data/schema.py`, not to a hand-kept list: the
  assembler asserts its row covers `FEATURE_COLUMNS` exactly, so schema drift
  breaks loudly in tests. Departures from CICFlowMeter are chosen, not accidental:
  bulk features emit 0 (they are ~always 0 upstream), zero-duration rates emit NaN
  to match cleaning's Inf policy (the *fitted pipeline* imputes them — reusing the
  train medians, no new statistics computed at serve time), and flows end on idle
  timeout or TCP close.
- The demo capture is built frame-by-frame with the same builders the tests use,
  so the parser is asserted against known on-wire values rather than against
  itself, and no binary fixture lands in git. Scoring the demo against the
  stand-in model gave the honest wrinkle worth keeping: the flood is flagged at
  the 1% budget, but the SYN sweep sails through — PortScan lives on Friday and
  the temporal model has never seen one, which is the slices report's finding
  resurfacing at the packet layer. The README says "mechanics, not detection
  claim" for exactly that reason.
- Scoring the capture output surfaced a real contract bug: `top_features` leaked
  `numeric__` pipeline prefixes to the analyst. Fixed centrally (the explainer,
  distill, and evasion now share one `display_feature_name`), with the test
  asserting no `__` ever reaches the response.

## Self-training (precision that lies, measured)

- The subtlest number in the study is the one that looks reassuring: pseudo-label
  precision is ~92% on *both* sides, and a practitioner monitoring only precision
  would call the pseudo-labels healthy. The damage lives in the composition of the
  8%: the benign-side errors are not random benign-ish flows, they are
  specifically the novel later-day attacks the model already scores lowest — 12.9%
  of the window's attacks, absorbed as training-set benign. Aggregate precision is
  the wrong lens for the same reason accuracy is: the errors are concentrated
  where the cost is.
- Design choices that keep the study honest: the eval window is the untouched
  *end* of the stream (a random holdout would let the adaptation window leak its
  own regime); each model gets its own validation-chosen threshold (a shared
  threshold flatters whichever model happens to match the static calibration); and
  the adaptation truth exists only inside the audit function — no model, pipeline,
  or threshold ever touches it.
- The oracle gap (0.653 → 0.844 eval-window PR-AUC) is deliberately reported on
  the eval window, not the full test split, so all three rows share one yardstick;
  it will not match the headline 0.529 and should not.
- Same-scale rule held again: pseudo-label taus and the reported thresholds are on
  the raw score scale the models emit, stated in the report header.

## Model-family leaderboard (the split picks the winner)

- The result I did not plan for: the honest split *inverts the ranking*. Naive
  Bayes (0.571) and logistic (0.569) beat the tuned LightGBM (0.529) temporally,
  while the optimistic split shows the familiar deep-leaderboard shape (LightGBM
  0.786 on top). The gap column is monotone in capacity — NB +0.067, LR +0.128,
  RF +0.243, GBDT +0.257 — which is the textbook bias/variance trade surfacing as
  an *evaluation* artifact: flexible models memorise the Mon–Wed regime and pay
  on Thu–Fri. The uncomfortable implication is stated in the report: model
  selection on the shuffled split ships the wrong model.
- Kept LightGBM as the deployed model anyway, and the reasoning belongs here: the
  headline number is the temporal PR-AUC *with* the rest of the system attached —
  calibration, thresholds, SHAP contracts, distillation — and the seed study +
  promotion gate price change on those terms. Swapping the family because one
  study's point estimate favors it would be tuning on the test split with extra
  steps; the leaderboard is a selection-protocol finding, not a promotion.
- The inversion sentence renders only when the two splits actually crown
  different winners (same discipline as the streaming/retrain branches: the text
  can never contradict the table it sits under).
- Baselines run at library defaults on purpose, and the scope section says the
  comparison *favors* the tuned deployed model — which makes the fact that it
  still loses the honest table more informative, not less.

## Property-based tests (and the drift blind spot they caught immediately)

- The suite is written around *contracts*, not functions: "the threshold chosen
  at an FPR budget never exceeds that budget on the selecting set" is the
  sentence the whole TPR@FPR story rests on, so it gets asserted for every
  labels/scores vector hypothesis can construct, not for three examples.
- It earned its keep before it was committed: the total-migration invariant
  ("shift everything off the reference support → PSI must read major") failed
  instantly on a constant reference. Root cause: overwriting the outer quantile
  edges with ±inf collapses a constant or two-valued feature into one bin —
  and CIC has such features *by construction* (bulk columns are ~always 0, flag
  counts are near-binary), so the deployed drift monitor had lanes it could
  never see. An example-based test would only have caught this if someone had
  already thought of it, which is precisely the case for property testing.
- Kept the suite honest about cost: bounded sizes, `deadline=None` (Windows CI
  timing), 30 examples for the pandas-heavy cleaning property, and the frames
  generator reuses the quirky-fixture defects so the generalization is of the
  known failure modes, not random noise.

## Campaign-level detection (two true statistics, one deployment)

- The flow-level and campaign-level numbers are *both* honest and they disagree
  by construction: 21% of hostile flows vs 5/5 operations alerted at the same
  threshold. The failure mode this report exists to prevent is quoting either
  one alone — the flow number undersells sustained attacks, the campaign number
  hides that PortScan ran 687 probes before the pager went off and that alert
  cost didn't move (benign traffic has no campaigns).
- First-alert latency counts the campaign's *own* flows, not stream position:
  an interleaved benign flow must not advance the counter, which is exactly the
  off-by-context bug a naive `np.where(alerts)[0]` over the whole stream would
  produce. The unit test plants a high-scoring benign flow between campaign
  flows to pin it.
- k_confirm exists because "one alert = detected" assumes an analyst (or a
  correlation layer) connects that alert to a campaign; on the stand-in the
  k=5 column drops Infiltration (2 alerts over 42 flows), which is the right
  conservative read of a signal that thin.

## Base-rate stress test (the fallacy cited by every IDS paper, finally computed)

- The study is deliberately cheap — one model fit, then pure Bayes arithmetic —
  because the point is not new measurement but a re-reading: the conditional
  TPR/FPR the evaluation report already establishes are prevalence-invariant, so
  the entire prevalence sweep is a closed form on two measured numbers. The
  interesting engineering was deciding what *not* to recompute.
- Used the **realized** test FPR (0.059% at the 0.1% budget), not the nominal
  budget, for every derived number. The val-chosen threshold undershoots its
  budget on later-day traffic — the same val→test threshold drift the cost report
  documents — and computing the fallacy on the nominal 0.1% would have made the
  queue look ~1.7x worse than the deployed system actually is. Honesty cuts both
  ways: this time the honest number is the *more* favorable one.
- The two inversions carry the report. Break-even prevalence (pi* = FPR/(TPR+FPR)
  = 0.64%) says where the queue flips majority-false, and the required-FPR
  inversion says the gap is ~5,800x at a 1e-5 prevalence — unclosable by
  thresholding, which is precisely why the suite's answer lives elsewhere
  (ranking, campaign aggregation, costs). The report ends by naming those layers
  rather than pretending a better model fixes arithmetic.
- Kept the assumed production rate (1%) *above* the computed break-even on the
  stand-in and said so, instead of quietly picking a prior that makes the
  dramatic majority-false story render. The prose branches on the comparison, so
  on real data it tells whichever story the numbers support.

## Adaptive conformal (the caveat, closed — and its price tag kept visible)

- The conformal study's best finding was a *shortfall* (attack coverage 64% on
  the temporal split, guarantee intact on the exchangeable one); leaving it as
  "conformal detects drift" felt like the per-service and hardening notes all
  over again — measuring a weakness and stopping one step short of acting on
  it. ACI is the act: steer alpha with the realized errors and the guarantee
  holds under arbitrary shift.
- The subtle correctness decision was the **quantile convention**. My first
  table used the textbook ceil((n+1)(1-alpha)) order statistic; the existing
  module computes the same level through `np.quantile(..., method="higher")`,
  which lands one order statistic higher. Either is defensible alone, but a
  static-vs-adaptive comparison must start from *identical* thresholds at the
  target alpha or the delta is partly convention, not adaptation — the unit
  test asserting table == conformal_quantile caught it. Same lesson as the
  seed/distill "same scale" rule, now for quantile arithmetic.
- Deliberately did **not** clamp alpha to [0, 1]. The excursions (alpha <= 0 =>
  include everything) are what make Gibbs-Candes assumption-free, and clamping
  quietly reintroduces a failure mode under persistent shift. The set
  constructor interprets the excursions instead; a test pins the semantics.
- The result is almost embarrassingly on-target (89.7% vs 90%) — which is what
  the theory promises for long-run coverage, so for once a clean number is
  *expected*, not suspicious. The honest cost is in the review column: +33pp of
  flows routed to a human. The report leads with that trade because "we
  restored the guarantee" without it would be the exact kind of free-lunch
  claim this project exists to reject.

## Threshold refresh (the study that answered "no" twice, usefully)

- I built this expecting the textbook story: the frozen threshold's realized
  FPR drifts off budget, the refresh pulls it back, and some detection comes
  along free. The stream said no twice. Detection: +0.1% (the drift cost is
  ranking — the model literally cannot score Friday's attack types, and no cut
  fixes blindness). Compliance: the frozen cut sits *closer* to the 1% budget
  (0.109%) than the refreshed one (0.156%), because the benign score
  distribution barely moves across these days while a 2-batch quantile estimate
  carries real sampling noise. My first compliance paragraph asserted the
  design intent ("the refresh keeps the promise honest"); the generated numbers
  contradicted it, so the paragraph is now branched on the measured distances —
  the same never-gaslight-the-table rule the retrain-policy and novelty renders
  learned.
- The value case still exists, and rather than manufacture it on the stand-in,
  it lives in the unit tests: a constructed benign-score shift where the frozen
  cut's realized FPR runs 50x over budget and the trailing-window refresh
  returns it under 5x. First version of that fixture failed for a great reason:
  perfectly separable toy scores park the FPR threshold at the *lowest attack
  score* (roc_curve collapses the collinear stretch), so no benign drift can
  ever spend the budget — the class overlap isn't test decoration, it is what
  makes an operating point a real object.
- Design rule that kept the study honest: refreshed cuts are chosen on the
  prequentially *emitted* scores (what the deployed model said before learning
  from the batch). Letting the retrained model re-pick its threshold on flows it
  had just trained on would be the quiet leak — realized FPR on trained-on rows
  is optimistic, and the threshold would inherit that optimism.

## Exemplar explanations (precedent as evidence, with its own audit)

- The design rule was *audit before serve*: the retrieval only became an API
  field after the report measured whether neighbour agreement means anything.
  It points the right way (89% vs 82% alert precision) but the disagreeing
  bucket is 44 alerts, so the render prints the bucket sizes ahead of the
  percentages and scopes the claim to triage ordering. Selling a 44-row gap as
  a re-ranker would be the base-rate mistake in miniature.
- The distance finding *failed* the intuitive expectation and matched the
  measured one: missed attacks are not farther from training than caught ones
  (7.8 vs 9.7 mean NN distance — backwards from the naive story), which is the
  novelty study's detection-rises-with-distance geometry showing up per flow.
  Two studies computing the same geometry from different directions and
  agreeing is worth more than either alone; the render branches so real
  burst-structured data can tell the opposite story.
- The examples table earned the feature its keep: the top novel-DDoS alerts
  retrieve Wednesday's DoS Hulk as nearest cases — cross-family precedent an
  analyst can actually pull, on a class the model never trained on.
- Serving decisions: class-balanced index (a proportional sample would be ~80%
  benign and rare classes would have no cases to match), float32 in bundle
  metadata (~1.4k rows), retrieval shares the anomaly scorer's pipeline
  transform, and the whole path is opt-in + best-effort — the explanation can
  vanish; the verdict cannot change.

## pcapng (closing a stated limitation without importing a parser)

- The original capture phase drew the line at classic pcap and made pcapng a
  documented error; Wireshark has defaulted to pcapng for years, so the "convert
  first" caveat was the stack's most-hit rough edge. The container is just
  length-prefixed blocks, so the stdlib-`struct` promise holds: the whole reader
  is one loop and two helpers.
- The two spec details that actually carry risk got the test attention:
  **byte order is per section** (the SHB type value is a palindrome precisely so
  it parses before the byte-order magic is known — the loop re-derives the order
  at every SHB and resets interface numbering, covered by a two-section test),
  and **timestamps are per interface** (`if_tsresol` has both a decimal and a
  binary encoding; both are converted to one ticks-per-second integer so the
  packet loop stays a single division — 10^-9 and 2^-10 fixtures pin it).
- Reused the classic reader's link-layer decode by factoring `_decode_frame`
  rather than copying it, so pcapng inherited the tested Ethernet/VLAN/raw-IP
  handling instead of re-implementing it — the same no-second-copy discipline as
  the services map and the ATT&CK tactic names.
- Judgement call: an interface with an unsupported link type skips *its* packets
  (counted, noted once) instead of failing the file — a multi-interface capture
  with one wifi NIC should still yield the Ethernet flows. And SPBs parse with a
  note that time features degrade, rather than inventing timestamps.

## Incident reports (the last mile from verdicts to a response artifact)

- Deliberately thin by design: `incident` computes nothing new — no model, no
  threshold, no metric. It re-reads the engine's own verdicts into the unit an
  analyst works, which is exactly why it can be trusted: there is no second
  scoring path to skew. The one algorithmic piece (contiguity grouping with a
  bridged-gap tolerance) is pure and tested on hand-built sequences, including
  the case that matters — a bridged benign row is *skipped over*, never absorbed
  into the incident.
- The first generated demo artifact was quietly embarrassing in a useful way:
  the stale local bundle predated the serving upgrades, so the report showed
  class "attack", no ATT&CK link, and "n/a" actions. Regenerating the bundle
  from current code fixed all three — and is a live demonstration of why the
  canary/verify machinery exists: an old artifact serves old behavior without
  erroring, and only the surrounding contract surfaces it.
- A CLI paper cut worth remembering: the incident command's `-o` short flag for
  `--output` collided with the global `-o/--override` option, and typer bound
  the output path as a config override (FileNotFoundError on a .md 'config').
  `score` had avoided the collision by convention (long flag only); now both do
  it deliberately.
- Fresh-bundle detail that improved the demo: the serving bundle is stratified/
  multiclass (it has seen PortScan), so the demo SYN sweep that the *temporal*
  stand-in misses is caught and named here — the same capture reads differently
  through the two models, which is the temporal story told from one more angle.

## Zeek ingestion (meet the data where it already lives)

- The design question was which columns to map, and the discipline was to map
  *fewer* than possible: only what a connection-total record can honestly say.
  It is tempting to synthesize IAT means from duration/packet counts (uniform
  spacing assumptions) — that would hand the model fabricated timing features
  indistinguishable from real ones. Missing + train-median imputation is the
  contract the pipeline already has for absent detail, and the cross-dataset
  study already measured what to expect from it; the module cites that instead
  of promising CIC-grade behaviour on conn.log input.
- Zeek's `history` string is event letters, not counters, so the flag-count
  mapping is labelled a lower bound in both the docstring and the README. A
  reviewer who knows Zeek would catch an unqualified "SYN Flag Count" mapping
  immediately — the qualification is the credibility.
- Parser notes: `#separator \x09` arrives space-separated and escape-encoded
  (everything after it uses the declared separator), and unset fields must be
  *dropped*, not parsed as the literal `-` (which floats to NaN by luck on
  numeric fields but would poison string fields like history). JSON-lines
  support is a few lines and covers the increasingly common json-streaming
  deployments.
- The pipe-masking near-miss is worth remembering: `black --check | tail -1`
  swallowed black's non-zero exit and the feat commit initially landed with an
  unformatted file — caught by reading the output, fixed by amending before
  push. Check chains must gate on the tool's exit code, not its last line.

## Invariants I am holding myself to (from the project rules)

1. No identifier/timestamp column (`Flow ID`, IPs, ports, `Timestamp`) ever
   reaches a model. A test will enforce it.
2. Transformers are fit on the training split only; never compute a statistic
   over the full dataset.
3. The headline evaluation is the **temporal/by-day split**; the stratified
   number is reported only as an optimistic reference, with the gap called out.
4. Lead with PR-AUC, per-class P/R/F1, and TPR@fixed-FPR — never accuracy.
5. Every run is reproducible from logged config + seed.
