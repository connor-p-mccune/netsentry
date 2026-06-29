# NetSentry — ONNX Export & Quantized Inference

_Generated 2026-06-29 23:37 UTC. The trained classifier exported to ONNX and run under ONNX
Runtime, benchmarked against the Python sklearn/LightGBM path on 2000 flows
(76 features), median of 25. Preprocessing stays in the numpy
pipeline._

## Fidelity

- Max absolute probability difference vs sklearn: **1.45e-07**
- Argmax agreement: **100.0%**

ONNX inference matches sklearn to float32 rounding — a safe drop-in.

## Latency (batch of 2000 flows)

| backend | batch latency (ms) | throughput (flows/s) | model size (bytes) |
|---|---|---|---|
| sklearn / LightGBM | 37.83 | 52,864 | — |
| ONNX Runtime | 26.16 | 76,461 | 2,609,839 |
| quantized ONNX (dynamic) | 32.83 | 60,917 | 2,609,970 |

ONNX Runtime runs at roughly **1.4x** the Python path here, with no
accuracy cost — the case for exporting a tree model for a low-overhead or
non-Python serving target.

## On quantization (an honest note)

Dynamic quantization targets matmul/conv weights. A gradient-boosted tree exports
to a single `TreeEnsembleClassifier` op whose parameters are split thresholds and
leaf values — **there are no quantizable matmul weights** — so dynamic quantization
leaves size and latency essentially unchanged. The low-overhead win here is ONNX
Runtime versus the Python path, not quantization; quantization is the lever for
neural nets (e.g. the autoencoder), not trees.
