# NetSentry — Cross-Dataset Generalization

_Generated 2026-06-29 22:49 UTC. The CIC-trained model is scored, unchanged, on a foreign
**NetFlow-schema** dataset adapted into CIC features. Both datasets here are
synthetic stand-ins; the methodology — not the absolute number — is the point._

## Result

| dataset | PR-AUC | ROC-AUC | TPR @ 0.1% FPR | attack prevalence |
|---|---|---|---|---|
| in-domain (CIC temporal test) | 0.529 | 0.668 | 11.9% | 0.25 |
| cross (synthetic-netflow) | 0.517 | 0.709 | 1.2% | 0.30 |

- **PR-AUC: in-domain 0.529 → cross 0.517 (gap +0.012).**

In-domain and cross scores are close. Detection transfers through the few shared behavioural features (volumes and rates); the rest of the CIC feature space is absent from a NetFlow schema and imputed. Treat the synthetic magnitude as illustrative — real UNSW-NB15 / NF-*-v2 numbers are the ones to trust.

## Method

- Foreign data: `synthetic-netflow` — a NetFlow-style schema (in/out
  packets & bytes, duration, port) whose attacks are DoS/DDoS-like high-volume
  flows.
- Adapter: rename/unit-convert the shared quantities, derive a few CIC rates, and
  leave every unmatched CIC feature NaN for the fitted pipeline to impute. No
  retraining and no peeking — the production bundle is scored exactly as served.
- For real numbers, point the adapter at UNSW-NB15 or the NetFlow `NF-*-v2`
  releases; the commands and framing are identical.
