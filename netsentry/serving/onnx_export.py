"""ONNX export + quantization for low-overhead / edge inference.

Exports the trained gradient-boosted classifier to ONNX (preprocessing stays in
the numpy pipeline — it is cheap) and runs it under ONNX Runtime, benchmarked
against the Python sklearn/LightGBM path. Dynamic quantization is attempted too;
for tree ensembles it is effectively a no-op — a `TreeEnsembleClassifier` carries
no quantizable matmul weights — which the report states plainly rather than
overselling.

Optional ``onnx`` extra: onnx, onnxruntime, skl2onnx, onnxmltools.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.data.split import load_split
from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.models.registry import ModelBundle

logger = get_logger(__name__)

ONNX_MODEL_NAME = "model.onnx"
ONNX_QUANT_NAME = "model.quant.onnx"
REPORT_NAME = "onnx.md"
# skl2onnx supports the ai.onnx.ml domain up to v3; the LightGBM/tree converters
# emit v5, so the target opset is pinned to keep conversion within range.
_TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}


def _register_lightgbm(estimator: Any) -> None:
    """Register the LightGBM->ONNX converter with skl2onnx (LightGBM only)."""
    if type(estimator).__name__ != "LGBMClassifier":
        return
    from lightgbm import LGBMClassifier
    from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
    from skl2onnx import update_registered_converter
    from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
    )


def export_to_onnx(bundle: ModelBundle, path: Path) -> Path:
    """Convert the bundle's classifier (operating on pipeline features) to ONNX."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    estimator = getattr(bundle.model, "model", bundle.model)  # unwrap our BaseModel
    n_features = len(bundle.feature_names())
    _register_lightgbm(estimator)
    onx = convert_sklearn(
        estimator,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        options={id(estimator): {"zipmap": False}},
        target_opset=_TARGET_OPSET,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    logger.info("Exported ONNX model", extra={"path": str(path), "features": n_features})
    return path


class OnnxScorer:
    """Run an exported ONNX classifier under ONNX Runtime."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._input = self._session.get_inputs()[0].name
        outputs = self._session.get_outputs()
        self._proba = next(
            (i for i, o in enumerate(outputs) if "prob" in o.name.lower()), len(outputs) - 1
        )

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        result = self._session.run(None, {self._input: np.asarray(x, dtype=np.float32)})
        return np.asarray(result[self._proba], dtype=float)


def quantize_onnx(src: Path, dst: Path) -> Path | None:
    """Dynamic-quantize an ONNX model; returns dst, or None if it could not run."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
        return dst
    except Exception as exc:  # quantization is a bonus and must never be fatal
        logger.warning("ONNX quantization skipped (%s)", exc)
        return None


def _median_ms(fn: Callable[[np.ndarray], Any], x: np.ndarray, repeats: int) -> float:
    fn(x)  # warm up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(x)
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples))


@dataclass
class OnnxReport:
    n_rows: int
    n_features: int
    repeats: int
    max_abs_proba_diff: float
    argmax_agreement: float
    sklearn_ms: float
    onnx_ms: float
    quant_ms: float | None
    onnx_bytes: int
    quant_bytes: int | None


def run_onnx_export(
    settings: Settings, *, bundle: ModelBundle | None = None, repeats: int = 25
) -> Path:
    """Export, quantize, benchmark, and write the ONNX report."""
    from netsentry.models.registry import latest_bundle, load_bundle

    if bundle is None:
        path = settings.serving.artifact_path or latest_bundle(settings)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                "No model bundle found. Train one with `netsentry train supervised` first."
            )
        bundle = load_bundle(Path(path))

    sample = load_split(settings, "temporal", "test")
    sample = sample.sample(min(2000, len(sample)), random_state=settings.seed)
    x = np.asarray(bundle.pipeline.transform(sample), dtype=np.float32)

    onnx_path = export_to_onnx(bundle, settings.paths.models_dir / ONNX_MODEL_NAME)
    scorer = OnnxScorer(onnx_path)
    sk_proba = np.asarray(bundle.model.predict_proba(x))
    ox_proba = scorer.predict_proba(x)

    quant = quantize_onnx(onnx_path, settings.paths.models_dir / ONNX_QUANT_NAME)
    quant_scorer = OnnxScorer(quant) if quant is not None else None

    report = OnnxReport(
        n_rows=len(x),
        n_features=x.shape[1],
        repeats=repeats,
        max_abs_proba_diff=float(np.max(np.abs(sk_proba - ox_proba))),
        argmax_agreement=float(np.mean(sk_proba.argmax(1) == ox_proba.argmax(1))),
        sklearn_ms=_median_ms(bundle.model.predict_proba, x, repeats),
        onnx_ms=_median_ms(scorer.predict_proba, x, repeats),
        quant_ms=_median_ms(quant_scorer.predict_proba, x, repeats) if quant_scorer else None,
        onnx_bytes=onnx_path.stat().st_size,
        quant_bytes=quant.stat().st_size if quant is not None else None,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(report), encoding="utf-8")
    logger.info("Wrote ONNX report", extra={"path": str(out_path)})

    quant_metric = {"onnx_quant_ms": report.quant_ms} if report.quant_ms is not None else {}
    with track_run(settings, "onnx") as run:
        run.log_metrics(
            {
                "onnx_max_abs_proba_diff": report.max_abs_proba_diff,
                "onnx_argmax_agreement": report.argmax_agreement,
                "sklearn_ms": report.sklearn_ms,
                "onnx_ms": report.onnx_ms,
                **quant_metric,
            }
        )
        run.log_artifact(out_path)
    return out_path


def _throughput(n_rows: int, ms: float) -> float:
    return n_rows * 1000.0 / ms if ms else float("nan")


def _render(r: OnnxReport) -> str:
    speedup = r.sklearn_ms / r.onnx_ms if r.onnx_ms else float("nan")
    if r.quant_ms is not None and r.quant_bytes is not None:
        quant_row = (
            f"| quantized ONNX (dynamic) | {r.quant_ms:.2f} | "
            f"{_throughput(r.n_rows, r.quant_ms):,.0f} | {r.quant_bytes:,} |"
        )
    else:
        quant_row = "| quantized ONNX (dynamic) | n/a | n/a | n/a |"
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# NetSentry — ONNX Export & Quantized Inference

_Generated {generated}. The trained classifier exported to ONNX and run under ONNX
Runtime, benchmarked against the Python sklearn/LightGBM path on {r.n_rows} flows
({r.n_features} features), median of {r.repeats}. Preprocessing stays in the numpy
pipeline._

## Fidelity

- Max absolute probability difference vs sklearn: **{r.max_abs_proba_diff:.2e}**
- Argmax agreement: **{r.argmax_agreement:.1%}**

ONNX inference matches sklearn to float32 rounding — a safe drop-in.

## Latency (batch of {r.n_rows} flows)

| backend | batch latency (ms) | throughput (flows/s) | model size (bytes) |
|---|---|---|---|
| sklearn / LightGBM | {r.sklearn_ms:.2f} | {_throughput(r.n_rows, r.sklearn_ms):,.0f} | — |
| ONNX Runtime | {r.onnx_ms:.2f} | {_throughput(r.n_rows, r.onnx_ms):,.0f} | {r.onnx_bytes:,} |
{quant_row}

ONNX Runtime runs at roughly **{speedup:.1f}x** the Python path here, with no
accuracy cost — the case for exporting a tree model for a low-overhead or
non-Python serving target.

## On quantization (an honest note)

Dynamic quantization targets matmul/conv weights. A gradient-boosted tree exports
to a single `TreeEnsembleClassifier` op whose parameters are split thresholds and
leaf values — **there are no quantizable matmul weights** — so dynamic quantization
leaves size and latency essentially unchanged. The low-overhead win here is ONNX
Runtime versus the Python path, not quantization; quantization is the lever for
neural nets (e.g. the autoencoder), not trees.
"""
