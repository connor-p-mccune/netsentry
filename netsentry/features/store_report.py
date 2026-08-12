"""Price the temporal leak: the same host-context features, joined correctly and incorrectly.

Three detectors see the same flows and the same split. One gets no host context (the deployed
model). One gets context from an as-of join, which a serving path could reproduce. One gets
context aggregated over the whole capture, which it could not. The gap between the last two is
the inflation a notebook would have shipped as a result.

The synthetic stand-in cannot host this comparison, and the report says so with the measurement
that proves it: the generator draws a fresh random address for every flow, so no source is ever
observed twice and *every* host context is structurally empty. That is a property of the
stand-in, not of CIC-IDS2017, where a few hundred hosts generate hundreds of thousands of flows.
So the mechanism is demonstrated on a controlled stream built to have the host structure the
stand-in lacks — a scanner sweeping a subnet among ordinary hosts — where the correct and
incorrect joins can actually differ.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from netsentry.evaluation import plots
from netsentry.features.store import FEATURE_NAMES, as_of_features, leakage_gap, leaky_features
from netsentry.log import get_logger
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import FeatureStoreConfig

logger = get_logger(__name__)

REPORT_NAME = "feature_store.md"
FIGURE_NAME = "feature_store_gap.png"

ENTITY_COLUMN = "Source IP"
TIME_COLUMN = "Timestamp"
DEST_HOST_COLUMN = "Destination IP"
DEST_PORT_COLUMN = "Destination Port"
BASE_FEATURES = ("Flow Duration", "Total Fwd Packets", "Flow Bytes/s")


@dataclass
class VariantResult:
    """One detector: which context it was given, and what it scored."""

    name: str
    description: str
    pr_auc: float
    n_features: int


@dataclass
class StoreStudy:
    """The stand-in diagnostic, the controlled stream, and what each join buys on it."""

    n_raw_flows: int
    n_raw_hosts: int
    mean_prior_events: float
    variants: list[VariantResult]
    gaps: dict[str, float]
    lookback_seconds: float
    n_stream: int
    n_hosts: int
    n_scanners: int


def _raw_host_diagnostic(settings: Settings, cap: int) -> tuple[int, int]:
    """How many distinct sources the raw capture actually contains — the reason for the stream."""
    frames = []
    for path in sorted(Path(settings.paths.data_raw).glob("*.csv")):
        frames.append(pd.read_csv(path, usecols=[ENTITY_COLUMN], low_memory=False))
    if not frames:
        raise FileNotFoundError(f"no raw CSVs under {settings.paths.data_raw}")
    raw = pd.concat(frames, ignore_index=True).head(cap)
    return len(raw), int(raw[ENTITY_COLUMN].nunique())


def _controlled_stream(settings: Settings) -> pd.DataFrame:
    """A flow stream with the host structure the stand-in lacks, and a host-level attack in it.

    Ordinary hosts open occasional connections to a handful of servers. A small number of
    scanners sweep many destinations in tight bursts. The per-flow features deliberately overlap
    between the two — a single scan connection looks much like a single benign one — so the only
    reliable evidence is *how busy the source has been*, which is exactly the signal a per-flow
    model cannot see and a feature store exists to supply.
    """
    cfg = settings.feature_store
    rng = np.random.default_rng(settings.seed)
    hosts = [f"10.0.{i // 254}.{i % 254 + 1}" for i in range(cfg.n_hosts)]
    scanners = set(rng.choice(cfg.n_hosts, size=cfg.n_scanners, replace=False).tolist())

    rows = []
    for host_idx, host in enumerate(hosts):
        is_scanner = host_idx in scanners
        n_flows = cfg.scanner_flows if is_scanner else int(rng.integers(1, cfg.benign_flows))
        # A scanner's burst starts at a random moment and is tightly packed; benign traffic is
        # spread across the capture.
        start = float(rng.integers(0, cfg.stream_seconds))
        for j in range(n_flows):
            offset = (
                j * cfg.scan_gap_seconds
                if is_scanner
                else float(rng.integers(0, cfg.stream_seconds))
            )
            rows.append(
                {
                    ENTITY_COLUMN: host,
                    TIME_COLUMN: pd.Timestamp("2017-07-03 09:00:00")
                    + pd.Timedelta(seconds=(start + offset) if is_scanner else offset),
                    DEST_HOST_COLUMN: f"10.1.{rng.integers(0, 254)}.{rng.integers(1, 254)}",
                    DEST_PORT_COLUMN: (
                        int(rng.integers(1, 65535))
                        if is_scanner
                        else int(rng.choice([80, 443, 53, 22]))
                    ),
                    # Per-flow features overlap heavily between the two populations.
                    "Flow Duration": float(rng.lognormal(8.0 if is_scanner else 8.4, 1.2)),
                    "Total Fwd Packets": float(rng.poisson(3 if is_scanner else 5) + 1),
                    "Flow Bytes/s": float(rng.lognormal(6.0, 1.5)),
                    "label_binary": int(is_scanner),
                }
            )
    frame = pd.DataFrame(rows).sort_values(TIME_COLUMN).reset_index(drop=True)
    return frame


def _fit_and_score(
    settings: Settings,
    frame: pd.DataFrame,
    context: pd.DataFrame | None,
    name: str,
    description: str,
    serve_context: pd.DataFrame | None = None,
) -> VariantResult:
    """Fit on the earlier half of the stream and score the later half — a temporal split.

    serve_context models the consequence that actually bites: a model trained on features
    computed one way and served features computed another. That is not a hypothetical, it is what
    happens the moment a notebook's whole-capture aggregates meet a serving path that can only
    look backwards.
    """
    seed_everything(settings.seed)
    base = frame[list(BASE_FEATURES)].to_numpy(dtype=float)
    x_train = base if context is None else np.hstack([base, context.to_numpy(dtype=float)])
    at_serve = context if serve_context is None else serve_context
    x_serve = base if at_serve is None else np.hstack([base, at_serve.to_numpy(dtype=float)])
    x, y = x_train, frame["label_binary"].to_numpy().astype(int)
    cut = len(frame) // 2
    model = HistGradientBoostingClassifier(
        max_iter=settings.supervised.n_estimators, random_state=settings.seed
    )
    model.fit(x[:cut], y[:cut])
    scores = model.predict_proba(x_serve[cut:])[:, 1]
    return VariantResult(
        name=name,
        description=description,
        pr_auc=float(average_precision_score(y[cut:], scores)),
        n_features=int(x.shape[1]),
    )


def run_store_study(settings: Settings) -> StoreStudy:
    """Diagnose the stand-in, then price both joins on a stream that has host structure."""
    cfg: FeatureStoreConfig = settings.feature_store
    n_raw, n_hosts_raw = _raw_host_diagnostic(settings, cfg.max_rows)

    stream = _controlled_stream(settings)
    as_of = as_of_features(
        stream,
        entity_column=ENTITY_COLUMN,
        time_column=TIME_COLUMN,
        dest_host_column=DEST_HOST_COLUMN,
        dest_port_column=DEST_PORT_COLUMN,
        lookback_seconds=cfg.lookback_seconds,
    )
    leaky = leaky_features(
        stream,
        entity_column=ENTITY_COLUMN,
        dest_host_column=DEST_HOST_COLUMN,
        dest_port_column=DEST_PORT_COLUMN,
    )

    variants = [
        _fit_and_score(settings, stream, None, "no host context", "per-flow features only"),
        _fit_and_score(
            settings,
            stream,
            as_of,
            "point-in-time context",
            f"as-of join, {cfg.lookback_seconds:g}s lookback, strictly earlier events only",
        ),
        _fit_and_score(
            settings,
            stream,
            leaky,
            "whole-capture context",
            "the one-line groupby: each flow sees its host's totals, including the future",
        ),
        _fit_and_score(
            settings,
            stream,
            leaky,
            "whole-capture, served point-in-time",
            "trained on the leaky join, then deployed against features a serving path can "
            "actually compute",
            serve_context=as_of,
        ),
    ]
    return StoreStudy(
        n_raw_flows=n_raw,
        n_raw_hosts=n_hosts_raw,
        mean_prior_events=float(np.mean(as_of[FEATURE_NAMES[0]])),
        variants=variants,
        gaps=leakage_gap(as_of, leaky),
        lookback_seconds=cfg.lookback_seconds,
        n_stream=len(stream),
        n_hosts=cfg.n_hosts,
        n_scanners=cfg.n_scanners,
    )


def run_store_report(settings: Settings) -> Path:
    """Run the feature-store study and write the report + figure."""
    study = run_store_study(settings)
    fig = plots.plot_barh(
        labels=[v.name for v in study.variants],
        values=[v.pr_auc for v in study.variants],
        xlabel="held-out PR-AUC on the controlled stream",
        title="Host context: absent, joined correctly, and joined from the future",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, fig), encoding="utf-8")
    logger.info("Wrote feature-store report", extra={"path": str(out_path)})

    with track_run(settings, "feature_store") as run:
        run.log_metrics(
            {f"pr_auc_{re.sub(r'[^0-9a-zA-Z]+', '_', v.name)}": v.pr_auc for v in study.variants}
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _variant_table(study: StoreStudy) -> str:
    rows = ["| detector | what it sees | features | held-out PR-AUC |", "|---|---|---|---|"]
    for v in study.variants:
        rows.append(f"| {v.name} | {v.description} | {v.n_features} | {v.pr_auc:.3f} |")
    return "\n".join(rows)


def _gap_table(study: StoreStudy) -> str:
    rows = ["| context feature | leaky mean / point-in-time mean |", "|---|---|"]
    for name in FEATURE_NAMES:
        ratio = study.gaps.get(name, float("nan"))
        cell = "unbounded" if not np.isfinite(ratio) else f"{ratio:.1f}x"
        rows.append(f"| `{name}` | {cell} |")
    return "\n".join(rows)


def _read(study: StoreStudy) -> str:
    base, pit, leak, skewed = study.variants
    honest = pit.pr_auc - base.pr_auc
    inflation = leak.pr_auc - pit.pr_auc
    first = (
        f"Host context is worth **{honest:+.3f} PR-AUC** when it is computed correctly "
        f"({base.pr_auc:.3f} to {pit.pr_auc:.3f}): the per-flow features cannot separate a single "
        "scan connection from a single benign one, and the count of what the source did in the "
        "preceding minute can."
        if honest > 0.01
        else f"Correctly computed context moves PR-AUC {honest:+.3f}, which on this stream is "
        "not the interesting part."
    )
    if inflation > 0.01:
        second = (
            f"The incorrect join adds a further **{inflation:+.3f}** on top "
            f"({pit.pr_auc:.3f} to {leak.pr_auc:.3f}), and *that* number is not detection — it is "
            "the size of the lie. It comes entirely from telling a flow at 09:00 what its source "
            "would go on to do later in the capture, which no serving path can know. A model "
            f"benchmarked at {leak.pr_auc:.3f} would arrive in production delivering "
            f"{pit.pr_auc:.3f} at best, and the gap would be blamed on drift."
        )
    else:
        second = (
            f"The incorrect join adds only {inflation:+.3f} on top, because correctly computed "
            "context has already resolved this stream and there is little left to buy. That "
            "makes the offline comparison look almost harmless, which is exactly why the last "
            "row exists."
        )
    collapse = leak.pr_auc - skewed.pr_auc
    third = (
        f"**The last row is the one that matters.** A model trained on whole-capture context and "
        f"then deployed against features a serving path can actually compute scores "
        f"{skewed.pr_auc:.3f} — a **{collapse:.3f} collapse** from the {leak.pr_auc:.3f} it was "
        "benchmarked at, and "
        + (
            f"{skewed.pr_auc - base.pr_auc:+.3f} against having no context at all. "
            if skewed.pr_auc > base.pr_auc
            else f"{base.pr_auc - skewed.pr_auc:.3f} *worse* than having no context at all. "
        )
        + "That is the shape of the failure in production: not a model that is subtly "
        "optimistic, but one that learned to lean on a number nobody can supply at request "
        "time. It would be diagnosed as drift, investigated as drift, and never fixed, because "
        "the cause is a join written six months earlier."
        if collapse > 0.01
        else "Serving the same model point-in-time features costs it "
        f"{collapse:.3f}, so on this stream the train/serve skew is mild — the leaky and correct "
        "features happen to be close enough that the model's reliance on the first transfers to "
        "the second."
    )
    return f"{first} {second} {third}"


def _render(study: StoreStudy, fig: Path) -> str:
    return f"""# NetSentry — Point-in-Time Correctness: a Feature Store, and the Leak It Prevents

_Controlled stream: {study.n_stream:,} flows from {study.n_hosts} hosts, {study.n_scanners} of
them scanning. Lookback window {study.lookback_seconds:g}s. Trained on the earlier half of the
stream, scored on the later half._

## Why this report exists

The per-flow model is identity-blind by design — IPs are dropped before anything is modelled, so
it cannot memorise *which host* attacked instead of *what an attack looks like*. That firewall
costs something real: one flow cannot express "this source has opened four hundred connections in
the last minute", which is the first thing a human analyst would look at.

Host **context** recovers that signal without reintroducing identity, because a behaviour count is
not an address. Computing it correctly is where production ML infrastructure earns its keep. The
obvious implementation — group the whole capture by source, join the totals back — hands a flow at
09:00 information about what its host did at 17:00. That is a **temporal leak**: it scores well
offline and cannot be reproduced by a serving path, for which 17:00 has not happened. The as-of
join is the fix, and it is the defining guarantee of a feature store (Feast, Tecton). It is also
the same class of mistake as the identifier leakage this project was built around — one axis over,
and the one that survives dropping every identifier column.

## Why this runs on a controlled stream

The synthetic stand-in cannot host the comparison, and the measurement says so plainly: across
{study.n_raw_flows:,} raw flows it contains **{study.n_raw_hosts:,} distinct source addresses** —
one per flow. No source is ever observed twice, so every host context is structurally empty
and both joins return the same nothing. (The controlled stream below averages
{study.mean_prior_events:.1f} prior events per flow, for comparison.) That is a property of the
generator, not of CIC-IDS2017, where a few hundred hosts
produce hundreds of thousands of flows. Rather than quietly reporting a null result caused by the
data, the mechanism is demonstrated on a stream built to have the structure the stand-in lacks:
ordinary hosts making occasional connections, and a handful of scanners sweeping many
destinations in tight bursts, with per-flow features that deliberately overlap so the *only*
reliable evidence is host-level.

## How different are the two joins, before any model?

{_gap_table(study)}

These ratios are the discrepancy at the feature level — how much larger each flow's context looks
when the future is allowed to contribute to it. Nothing has been fitted yet.

## What does each join buy?

{_variant_table(study)}

{_read(study)}

![PR-AUC by context join](../figures/{fig.name})

## The design that makes this safe

The store computes context with a two-pointer sweep over time-sorted events per entity, so each
flow sees only its source's events in `[t - {study.lookback_seconds:g}s, t)` — strictly earlier,
never simultaneous. Excluding the flow's own instant matters more than it looks: with one-second
timestamp resolution ties are common, and including them would leak the label-bearing flow into
its own feature. The sweep is linear in the number of flows rather than the quadratic of a
per-row filter, and it is the same computation an online lookup would run against a live window —
which is the other half of the guarantee, since a store whose offline and online definitions can
drift has reintroduced the training/serving skew it exists to prevent.

Identifiers are used to *compute* the aggregates and never reach the model: what the model sees
is four behaviour counts, and `Source IP` stays behind the same `remainder="drop"` firewall as
always.

## Scope

Four aggregates over one entity type and one window is a deliberately small store; destination-side
context, per-(source, service) entities and multiple windows are the obvious extensions, and each
multiplies the join cost without changing the correctness argument. The controlled stream is a
demonstration of the *mechanism*, not a detection result — its absolute numbers mean nothing
about CIC-IDS2017, and the report would be dishonest if it presented them as if they did. Running
this on the real dataset is the natural follow-up and requires only that the capture have repeat
hosts, which the real one does. The store's correctness properties are pinned by unit tests
independently of any dataset: that a flow never sees its own instant, that events outside the
window are excluded, and that the leaky join demonstrably includes the future."""
