# Model Card — NetSentry

> Fill in as models are trained. Keep it honest; the limitations section is a
> feature, not a weakness.

## Model details
- **Name / version:** NetSentry `<vX.Y.Z>`
- **Type:** Supervised multiclass classifier (LightGBM) + unsupervised anomaly
  detector (Isolation Forest + benign-only autoencoder)
- **Input:** pre-computed network-flow features (CIC-IDS2017 schema, identifier
  columns removed)
- **Output:** attack class + probability, anomaly score, SHAP top features

## Intended use
- **Intended:** education, research, and demonstration of ML-based intrusion
  detection on flow data; a reference for honest evaluation methodology.
- **Out of scope:** drop-in production NIDS; live packet/NetFlow capture;
  adversarial-evasion-robust deployment.

## Training data
- CIC-IDS2017 (see `DATA_CARD.md`). Known issues handled: leakage columns dropped,
  Inf/NaN imputed, duplicates removed, labels consolidated, imbalance addressed.

## Evaluation
- **Headline split:** temporal / by-day (honest). **Reference split:** stratified
  random (optimistic; reported to expose the leakage gap).
- **Metrics:** PR-AUC, per-class P/R/F1, macro-F1, TPR@0.1%/1% FPR, confusion
  matrix, anomaly leave-one-attack-out detection.
- **Results:** `<fill from netsentry eval report>`

## Limitations & ethical considerations
- Trained on 2017 traffic; concept drift means real-world performance will differ.
- False positives cause analyst alert fatigue; thresholds are tunable and the
  operating point is reported explicitly.
- Dataset reflects a specific testbed; class balance and attack mix are not
  representative of any particular network.
- Not evaluated against adaptive adversaries; do not rely on it as a sole control.
