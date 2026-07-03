"""Per-service detection parity — does one global threshold treat services equally?

The headline evaluation picks a single global decision threshold and reports one
aggregate detection rate and one aggregate false-positive rate. That aggregate hides
*where* the false alarms land and *which* services go under-detected. This study
slices the honest temporal-split test set by the **service** implied by
``Destination Port`` and measures detection (TPR) and false-alarm (FPR) rate per
service at that one global threshold.

Two things make this more than a per-class rerun of ``slices.py``:

- It groups by a **non-label operational attribute** (the service a SOC actually
  routes alerts on), not by the attack class — which is unknown at routing time — and
  it includes *benign* flows, so it can measure per-service **FPR**, the quantity
  ``slices.py`` (attack-only) cannot.
- The grouping key is ``Destination Port``, a field the model **never sees**: it is
  dropped from the feature set precisely so the model cannot memorise "attack X hits
  port Y" (see ``.claude/rules/ml.md``). The port is used here only to *slice*, never
  to *predict* — so any parity gap reflects the model's behavioural generalisation,
  not a port lookup.

The framing is an equalized-odds fairness audit transplanted to network defence: a
large *unintended* TPR/FPR gap across services is where a per-service threshold would
beat one global cut, and it tells the SOC which service queue a single threshold
floods first (alert fatigue is where detection systems die, not misses).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import DESTINATION_PORT
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import positive_scores, rates_at_threshold, threshold_at_fpr
from netsentry.log import get_logger
from netsentry.training.tracking import track_run
from netsentry.training.train_supervised import fit_supervised

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "subgroups.md"

# Well-known TCP/UDP port -> coarse service. IANA-registered assignments; this is
# reference data (like the ATT&CK mapping) rather than a per-deployment knob. Ports
# outside the map fall into "other/ephemeral" — which is exactly where a port scan's
# sprayed destinations and odd C2 ports land, itself a meaningful bucket.
_PORT_SERVICE: dict[int, str] = {
    20: "FTP",
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    465: "SMTP",
    587: "SMTP",
    53: "DNS",
    80: "HTTP",
    8000: "HTTP",
    8080: "HTTP",
    110: "POP3",
    995: "POP3",
    143: "IMAP",
    993: "IMAP",
    443: "HTTPS",
    8443: "HTTPS",
    139: "SMB",
    445: "SMB",
    3389: "RDP",
}
OTHER_SERVICE = "other/ephemeral"


def service_of(port: float) -> str:
    """Map a destination port to a coarse service name (well-known ports; else 'other').

    Non-finite or non-integer ports (missing/garbled rows) fall into the 'other'
    bucket rather than raising — this is an evaluation slice, not a hard gate.
    """
    try:
        number = int(port)
    except (ValueError, TypeError, OverflowError):
        return OTHER_SERVICE
    return _PORT_SERVICE.get(number, OTHER_SERVICE)


@dataclass
class ServiceSlice:
    """Detection and false-alarm behaviour for one service at the global threshold."""

    service: str
    n_benign: int
    n_attack: int
    detection: float  # TPR within the service (NaN when the service has no attacks)
    fpr: float  # FPR within the service (NaN when the service has no benign flows)
    true_positives: int
    false_positives: int
    alert_share: float  # this service's share of ALL false positives raised


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate (95% by default).

    Per-service supports are a few thousand flows at best, so each observed rate
    carries visible binomial noise; Wilson stays honest near 0 and at small n where
    the naive normal interval collapses or escapes [0, 1].
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * float(np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def service_slices(
    y_true: np.ndarray,
    scores: np.ndarray,
    ports: np.ndarray,
    threshold: float,
    min_support: int,
) -> list[ServiceSlice]:
    """Per-service TPR/FPR at one global ``threshold`` (services with >= min_support flows)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    services = np.array([service_of(p) for p in np.asarray(ports)])
    flagged = scores >= threshold
    total_fp = int(np.sum(flagged & (y_true == 0)))

    out: list[ServiceSlice] = []
    for service in sorted(set(services.tolist())):
        mask = services == service
        if int(mask.sum()) < min_support:
            continue
        benign = mask & (y_true == 0)
        attack = mask & (y_true == 1)
        n_benign, n_attack = int(benign.sum()), int(attack.sum())
        tp = int(np.sum(flagged & attack))
        fp = int(np.sum(flagged & benign))
        detection = float(np.mean(flagged[attack])) if n_attack else float("nan")
        fpr = float(np.mean(flagged[benign])) if n_benign else float("nan")
        alert_share = (fp / total_fp) if total_fp else 0.0
        out.append(ServiceSlice(service, n_benign, n_attack, detection, fpr, tp, fp, alert_share))
    return out


def parity_gap(values: list[float]) -> float:
    """Max - min across the finite values — the equalized-odds-style disparity (0 if <2)."""
    finite = [v for v in values if np.isfinite(v)]
    return float(max(finite) - min(finite)) if len(finite) >= 2 else 0.0


def run_subgroups_report(settings: Settings) -> Path:
    """Fit the temporal binary model; audit per-service detection/false-alarm parity."""
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    result = fit_supervised(variant)

    # Raw attack score (a ranking), matching the operating points the evaluation report
    # uses; the threshold is chosen on validation, never on the test slice.
    s_val = positive_scores(result.proba_val, result.classes)
    s_test = positive_scores(result.proba_test, result.classes)

    test = load_split(settings, "temporal", "test")
    y_test = test[BINARY_TARGET].to_numpy()
    ports = test[DESTINATION_PORT].to_numpy()

    operating_fpr = settings.thresholds.fpr_targets[-1]  # looser budget -> more detection
    threshold = threshold_at_fpr(result.y_val.astype(int), s_val, operating_fpr)

    min_support = settings.subgroups.min_support
    slices = service_slices(y_test, s_test, ports, threshold, min_support)
    overall = rates_at_threshold(y_test, s_test, threshold)

    fpr_gap = parity_gap([s.fpr for s in slices if s.n_benign >= min_support])
    tpr_gap = parity_gap([s.detection for s in slices if s.n_attack >= min_support])

    fig = _plot(slices, operating_fpr, variant.paths.figures_dir / "subgroups.png")
    report = _render(slices, overall, threshold, operating_fpr, fpr_gap, tpr_gap, min_support, fig)
    out_path = variant.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote subgroups report", extra={"path": str(out_path), "services": len(slices)})

    with track_run(settings, "subgroups") as run:
        run.log_metrics(
            {
                "overall_fpr": overall["fpr"],
                "overall_tpr": overall["tpr"],
                "fpr_parity_gap": fpr_gap,
                "tpr_parity_gap": tpr_gap,
            }
        )
        run.log_metrics({f"fpr_{s.service.replace('/', '_')}": s.fpr for s in slices if s.n_benign})
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _plot(slices: list[ServiceSlice], operating_fpr: float, out_path: Path) -> Path:
    """Per-service FPR bars with the global budget drawn as a reference line."""
    scored = [s for s in slices if np.isfinite(s.fpr)]
    if not scored:  # degenerate split — draw an empty-but-valid figure
        scored = slices
    labels = [s.service for s in scored]
    values = [s.fpr if np.isfinite(s.fpr) else 0.0 for s in scored]
    headroom = max([*values, operating_fpr]) * 1.25 or 1.0
    return plots.plot_barh(
        labels,
        values,
        xlabel="False-positive rate within the service",
        title="Per-service false-alarm parity (temporal split)",
        out_path=out_path,
        xmax=headroom,
        vline=(f"global {operating_fpr * 100:g}% budget", operating_fpr),
    )


def _render(
    slices: list[ServiceSlice],
    overall: dict[str, float],
    threshold: float,
    operating_fpr: float,
    fpr_gap: float,
    tpr_gap: float,
    min_support: int,
    fig: Path,
) -> str:
    rows = [
        "| service (dst port) | benign | attack | detection [95% CI] | FPR [95% CI] "
        "| alert share |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(slices, key=lambda s: (s.fpr if np.isfinite(s.fpr) else -1.0), reverse=True):
        if np.isfinite(s.detection):
            d_lo, d_hi = wilson_interval(s.true_positives, s.n_attack)
            det = f"{s.detection * 100:.1f}% [{d_lo * 100:.1f}, {d_hi * 100:.1f}]"
        else:
            det = "— (no attacks)"
        if np.isfinite(s.fpr):
            f_lo, f_hi = wilson_interval(s.false_positives, s.n_benign)
            fpr = f"{s.fpr * 100:.2f}% [{f_lo * 100:.2f}, {f_hi * 100:.2f}]"
        else:
            fpr = "—"
        rows.append(
            f"| {s.service} | {s.n_benign:,} | {s.n_attack:,} | {det} | {fpr} "
            f"| {s.alert_share * 100:.0f}% |"
        )

    parts: list[str] = []
    scored = [s for s in slices if np.isfinite(s.fpr) and s.n_benign >= min_support]
    if scored:
        hot = max(scored, key=lambda s: s.fpr)
        cool = min(scored, key=lambda s: s.fpr)
        hot_lo, _hot_hi = wilson_interval(hot.false_positives, hot.n_benign)
        # Only claim the hottest service "genuinely exceeds" the budget if its whole
        # interval sits above it; otherwise say so — per-service supports are small
        # enough that part of any spread is binomial noise, and pretending otherwise
        # is exactly the over-reading this project exists to avoid.
        if hot_lo > operating_fpr:
            noise_note = (
                "and its interval sits wholly above the budget, so the excess is real, not "
                "sampling noise"
            )
        else:
            noise_note = (
                f"though its interval still straddles the budget — at ~{hot.n_benign:,} benign "
                f"flows per service much of this spread is binomial noise, which is why the table "
                f"carries Wilson intervals instead of bare rates"
            )
        parts.append(
            f"At the single global {operating_fpr * 100:g}%-FPR threshold the overall test "
            f"false-positive rate is {overall['fpr'] * 100:.2f}%, but nothing pins any *single* "
            f"service to that budget — the threshold only constrains the aggregate. "
            f"**{hot.service}** runs hottest at {hot.fpr * 100:.2f}% FPR and alone accounts for "
            f"{hot.alert_share * 100:.0f}% of every false positive raised ({noise_note}), while "
            f"**{cool.service}** sits at {cool.fpr * 100:.2f}%. A SOC routing alerts per service "
            f"watches the {hot.service} queue fill fastest either way; per-service thresholds "
            "would pin each queue to its own budget, which one global cut structurally cannot."
        )
    attacked = [s for s in slices if s.n_attack >= min_support and np.isfinite(s.detection)]
    if len(attacked) >= 2:
        best = max(attacked, key=lambda s: s.detection)
        worst = min(attacked, key=lambda s: s.detection)
        best_lo, _ = wilson_interval(best.true_positives, best.n_attack)
        _, worst_hi = wilson_interval(worst.true_positives, worst.n_attack)
        solid = ", and their intervals do not overlap — this gap is signal, not sampling noise"
        parts.append(
            f"Detection is uneven too: **{best.service}** catches {best.detection * 100:.0f}% of "
            f"its attacks against **{worst.service}** at {worst.detection * 100:.0f}% "
            f"(a {tpr_gap * 100:.0f}-point gap{solid if best_lo > worst_hi else ''}). The "
            "under-detected service is where the benign-only anomaly detector earns its place — "
            "the supervised model cannot recall an attack type concentrated on a service whose "
            "later-day traffic it never trained on."
        )
    elif attacked:
        only = attacked[0]
        parts.append(
            f"Only **{only.service}** carries enough later-day attacks ({only.n_attack:,}) to "
            f"score detection here ({only.detection * 100:.0f}%); the temporal split leaves the "
            "other services benign-only in the test window, so their parity story is about false "
            "alarms, not misses — which is the more common operational failure anyway."
        )
    if not parts:
        parts.append(
            "No service cleared the support floor on this split — rerun on the full dataset or "
            "lower `subgroups.min_support`."
        )
    read = "\n\n".join(parts)

    return f"""# NetSentry — Per-Service Detection Parity

_Synthetic stand-in. Honest **temporal** test flows grouped by the service implied by
`Destination Port`, scored at one global {operating_fpr * 100:g}%-FPR threshold (raw
attack score {threshold:.3f}, chosen on validation). Overall at this cut: detection
{overall['tpr'] * 100:.1f}%, FPR {overall['fpr'] * 100:.2f}%. Services with fewer than
{min_support:,} flows are omitted. FPR parity gap **{fpr_gap * 100:.2f} pts**; detection
parity gap **{tpr_gap * 100:.0f} pts**._

## Why slice by service, and why the port is safe to use here

A SOC routes and triages alerts by **service/asset**, not by attack class (which is
unknown when the alert fires). So the operational question is not only "which attacks
do we catch" (see the per-class slices) but "does one global threshold treat each
service fairly" — equal detection where attacks exist, and comparable false-alarm
pressure everywhere else. This is an equalized-odds audit in security clothing.

Crucially the grouping key, `Destination Port`, is a field the **model never sees**:
it is dropped from the feature set precisely so the model cannot memorise
"attack X always hits port Y" (`.claude/rules/ml.md`). Here the port only *labels the
slice*; it never enters a prediction. So every gap below is the model's behavioural
generalisation across services, not a port lookup leaking back in.

{chr(10).join(rows)}

![Per-service FPR](../figures/{fig.name})

## Read

{read}

The lesson mirrors the project's spine: just as the aggregate PR-AUC hides which
attacks are caught, a single global threshold hides *where* its false positives
concentrate. The honest move is to report the per-service spread and let the operator
set per-service operating points — the same "one number lies" discipline the temporal
split applies to the headline metric.
"""
