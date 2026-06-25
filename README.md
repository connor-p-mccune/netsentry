# NetSentry — ML Network Intrusion Detection

> Fill the `<…>` placeholders with your **real** measured numbers as you complete
> the build. Do not invent results — the honest numbers are the impressive part.

**A reproducible, leakage-safe machine-learning pipeline that detects network
intrusions in flow data — pairing a supervised classifier for known attacks with
an unsupervised anomaly detector for novel ones, served behind a real-time API
with explainable predictions.**

![CI](https://img.shields.io/badge/CI-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

---

## Why this project is different

Most public CIC-IDS2017 projects report ~99.9% accuracy. That number is almost
always an artifact of **data leakage** (identifier columns + a naive random
split) and a metric (accuracy) that is meaningless on data that is ~80% benign.
NetSentry is built to be the project that does it right:

- **Leakage-safe by construction** — identifier/timestamp columns are dropped and
  all preprocessing is fit on the training split only. (A test enforces it.)
- **Honestly evaluated** — the headline result uses a **temporal / by-day split**,
  not a shuffled one, and the optimistic random-split number is reported beside
  it so the gap is visible.
- **Operational metrics** — leads with PR-AUC, per-class recall, and
  **detection rate at a fixed 0.1% false-positive budget**, because in a SOC the
  binding constraint is analyst time, not raw accuracy.
- **Detects the unknown** — a benign-only anomaly detector flags attack classes
  the supervised model never trained on (leave-one-attack-out).
- **Explainable** — every prediction returns the top contributing features (SHAP).

## Headline results

> _Temporal split (the honest number). See `docs/` for full report + figures._

| Metric | Score |
|---|---|
| PR-AUC (binary benign/attack) | `<…>` |
| Macro-F1 (multiclass) | `<…>` |
| Detection rate @ 0.1% FPR | `<…>` |
| Detection rate @ 1% FPR | `<…>` |
| Anomaly detector — held-out attack detection (avg) | `<…>` |
| Inference latency (p95, single flow) | `<…> ms` |
| Throughput | `<…> req/s` |

> Reference (optimistic) stratified-split PR-AUC: `<…>` — reported only to show
> the leakage gap; the temporal number above is the one that matters.

![Results](docs/figures/pr_curve.png)

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). In short: `download → clean →
honest split → leakage-safe feature pipeline → LightGBM (known) + Isolation
Forest/autoencoder (novel) → SHAP → MLflow`, then a FastAPI service that loads one
pipeline+model artifact and returns predictions with explanations.

## Tech stack

Python 3.11 · scikit-learn · LightGBM · PyTorch · SHAP · MLflow · FastAPI ·
pydantic · Prometheus · Docker · GitHub Actions · pytest/ruff/mypy.

## Quickstart

```bash
make install
netsentry download          # fetch CIC-IDS2017 into data/raw
netsentry prep              # clean + honest splits + features
netsentry train supervised  # train LightGBM, log to MLflow
netsentry train anomaly     # train benign-only anomaly detector
netsentry eval              # generate metrics report + figures
netsentry serve             # FastAPI on :8000
# or:
docker compose -f docker/docker-compose.yml up
```

Example prediction:

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d @examples/sample_flow.json
# → {"predicted_class":"DoS Hulk","attack_probability":0.97,
#    "anomaly_score":0.83,"top_features":[...],"model_version":"0.1.0"}
```

## Reproducibility

Every result is reproducible from a logged config + seed. `netsentry eval`
regenerates the report and figures; MLflow holds params, metrics, artifacts, and
the environment for each run.

## Limitations

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). NetSentry consumes pre-computed
flow features (not raw packets), is trained on a 2017 dataset, and is a rigorous
reference implementation and demo — not a drop-in production NIDS.

## License

MIT
