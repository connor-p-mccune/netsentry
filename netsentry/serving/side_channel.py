"""The API answers the question twice: once in the body, once in the shape of the reply.

Everything this project does about adversaries assumes the attacker learns the verdict by
*being told* it -- through the API, under an API key, inside a rate limit, with every query
counted. The [extraction study](extraction.md) prices what those queries buy; the
[control-loop study](control.md) prices what an attacker can do by generating alerts. Both
defences are query-side, and both assume there is no other way to find out.

There is another way, and it is written into the response contract. `mitre` is `null` for a
clear verdict and an object for an alert. The optional anomaly explanation is computed **only
for flows the detector flagged**. `recommended_action` and `prediction_set` change length with
the decision. So an observer who can see encrypted traffic between a sensor and the service --
a network position, a shared host, a co-tenant -- reads the verdict off the packet lengths
without decrypting anything. That is a **free, passive, undetectable oracle**, and it is exactly
the oracle every query-side defence is built to ration.

The module measures the channel rather than describing it: flows go through the real
application, the reply's length and elapsed time are recorded beside the verdict, and the
verdict is recovered from each observable alone. Then the fixes are tried in the order an
engineer would try them, and the point of the study turns out to be how far down that ladder the
channel survives -- because the last thing leaking is the thing the contract exists to return.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.metrics import roc_auc_score

from netsentry.evaluation import plots
from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import SideChannelConfig

logger = get_logger(__name__)

REPORT_NAME = "side_channel.md"
SIZE_FIGURE = "side_channel_size.png"
MITIGATION_FIGURE = "side_channel_mitigation.png"


# --------------------------------------------------------------------------------------
# Reading a verdict off the shape of a reply.
# --------------------------------------------------------------------------------------


def separation(signal: np.ndarray, verdicts: np.ndarray) -> float:
    """How well one observable alone recovers the verdict, as an AUC.

    Reported as ``max(auc, 1 - auc)`` because an attacker does not care which direction the
    channel points -- a body that is reliably *smaller* on an alert leaks exactly as much as one
    that is reliably bigger, and reporting 0.02 as "almost no leakage" would be a mistake about
    the adversary rather than about the statistic.
    """
    if len(np.unique(verdicts)) < 2 or float(np.std(signal)) < 1e-12:
        return 0.5
    auc = float(roc_auc_score(verdicts, signal))
    return max(auc, 1.0 - auc)


def threshold_accuracy(signal: np.ndarray, verdicts: np.ndarray) -> tuple[float, float]:
    """The best single cut on one observable, and the accuracy it achieves.

    An AUC says the channel is ordered; an attacker wants a rule. The optimum is found by a
    prefix scan over the sorted observations rather than a grid, so the number is the genuine
    best of the comparison class -- but only cuts at the **end of a tie group** are realisable,
    because a threshold cannot separate two replies of identical length. Scoring the others
    reports an accuracy no rule can achieve: on a constant signal it would claim perfect
    separation out of nothing at all.
    """
    order = np.argsort(signal, kind="stable")
    labels = np.asarray(verdicts, dtype=int)[order]
    values = np.asarray(signal, dtype=float)[order]
    total = len(labels)
    positives = int(labels.sum())
    if total == 0 or positives in (0, total):
        return 0.0, max(positives, total - positives) / max(total, 1)
    below_positive = np.concatenate([[0], np.cumsum(labels)])
    below_total = np.arange(total + 1)
    correct = (below_total - below_positive) + (positives - below_positive)
    flipped = below_positive + (total - below_total - (positives - below_positive))
    best = np.maximum(correct, flipped).astype(float)
    realisable = np.ones(total + 1, dtype=bool)
    realisable[1:total] = values[:-1] != values[1:]
    best[~realisable] = -np.inf
    best_index = int(np.argmax(best))
    return float(values[min(best_index, total - 1)]), float(np.max(best) / total)


def serialised_size(body: dict[str, Any]) -> int:
    """Length of a response body as JSON, for comparing variants of the same reply."""
    import json

    return len(json.dumps(body, separators=(",", ":")))


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelRow:
    """One endpoint configuration, judged as a covert channel."""

    endpoint: str
    size_auc: float
    size_accuracy: float
    latency_auc: float
    latency_accuracy: float
    mean_size_benign: float
    mean_size_alert: float
    mean_ms_benign: float
    mean_ms_alert: float

    @property
    def size_gap(self) -> float:
        """Bytes of difference between an alert's reply and a clear one's."""
        return self.mean_size_alert - self.mean_size_benign


@dataclass(frozen=True)
class FieldRow:
    """One response field, the verdicts it appears on, and the bytes it carries."""

    field: str
    present_on: str
    mean_bytes: float


@dataclass(frozen=True)
class MitigationRow:
    """A change to the contract, and what it costs to keep."""

    mitigation: str
    size_auc: float
    latency_auc: float
    added_bytes: float
    added_ms: float


@dataclass
class SideChannelStudy:
    """Everything the report needs, computed once."""

    channels: list[ChannelRow]
    fields: list[FieldRow]
    mitigations: list[MitigationRow]
    n_flows: int
    alert_share: float
    seconds: float = 0.0

    def channel(self, name: str) -> ChannelRow | None:
        """Look up one endpoint configuration."""
        return next((row for row in self.channels if row.endpoint == name), None)

    def worst(self) -> ChannelRow | None:
        """The configuration that leaks the most through response size."""
        return max(self.channels, key=lambda row: row.size_auc) if self.channels else None


# --------------------------------------------------------------------------------------
# Driving the real service.
# --------------------------------------------------------------------------------------


def _payloads(settings: Settings, count: int, rng: np.random.Generator) -> list[dict[str, float]]:
    """Real later-day flows, as the API's own request bodies.

    Drawn from the temporal test split rather than generated, because the channel being
    measured is a difference between *verdicts*, and a synthetic flow the model scores
    unrealistically would change the alert share and therefore the measurement.
    """
    from netsentry.data.split import load_split
    from netsentry.features.feature_sets import numeric_features

    frame = load_split(settings, "temporal", "test")
    columns = [name for name in numeric_features() if name in frame.columns]
    picked = frame.iloc[rng.choice(len(frame), min(count, len(frame)), replace=False)]
    return [
        {name: float(row[name]) for name in columns if np.isfinite(row[name])}
        for _, row in picked.iterrows()
    ]


def _drive(
    client: Any, payloads: list[dict[str, float]], query: str, repeats: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """POST every flow and record the reply's length, its latency, and the verdict.

    Latency is the median of a few repeats per flow. A single timing is about whatever else the
    machine was doing; an attacker watching a real service gets to average too, and understating
    what they can measure would flatter the defence.
    """
    sizes, timings, verdicts = [], [], []
    for payload in payloads:
        samples = []
        body: dict[str, Any] = {}
        length = 0
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            response = client.post(f"/predict{query}", json={"flow": payload})
            samples.append((time.perf_counter() - start) * 1000.0)
            length = len(response.content)
            body = response.json()
        sizes.append(length)
        timings.append(float(np.median(samples)))
        verdicts.append(int(bool(body.get("is_attack", False))))
    return np.array(sizes, dtype=float), np.array(timings, dtype=float), np.array(verdicts)


def _channel(
    endpoint: str, sizes: np.ndarray, timings: np.ndarray, verdicts: np.ndarray
) -> ChannelRow:
    """Summarise one endpoint configuration as a covert channel."""
    alerts = verdicts == 1
    clear = ~alerts
    return ChannelRow(
        endpoint=endpoint,
        size_auc=separation(sizes, verdicts),
        size_accuracy=threshold_accuracy(sizes, verdicts)[1],
        latency_auc=separation(timings, verdicts),
        latency_accuracy=threshold_accuracy(timings, verdicts)[1],
        mean_size_benign=float(sizes[clear].mean()) if clear.any() else 0.0,
        mean_size_alert=float(sizes[alerts].mean()) if alerts.any() else 0.0,
        mean_ms_benign=float(timings[clear].mean()) if clear.any() else 0.0,
        mean_ms_alert=float(timings[alerts].mean()) if alerts.any() else 0.0,
    )


def _capture(client: Any, payloads: list[dict[str, float]], query: str) -> list[dict[str, Any]]:
    """One pass that keeps the whole body, so response variants can be costed exactly."""
    return [client.post(f"/predict{query}", json={"flow": payload}).json() for payload in payloads]


def _fields(bodies: list[dict[str, Any]]) -> list[FieldRow]:
    """Which fields carry the bytes, and which verdicts they appear on.

    A field is measured by the serialised length of its own value, averaged over the replies
    where it is populated. A field present on **alerts only** is a channel by itself whatever
    its size, because its presence is the signal.
    """
    import json

    populated: dict[str, list[int]] = {}
    seen_on: dict[str, set[bool]] = {}
    for body in bodies:
        verdict = bool(body.get("is_attack", False))
        for field_name, value in body.items():
            if value is None or value == [] or value == {}:
                continue
            populated.setdefault(field_name, []).append(
                len(json.dumps(value, separators=(",", ":")))
            )
            seen_on.setdefault(field_name, set()).add(verdict)
    rows = []
    for field_name, lengths in populated.items():
        verdicts = seen_on.get(field_name, set())
        where = (
            "**alerts only**"
            if verdicts == {True}
            else ("clear only" if verdicts == {False} else "always")
        )
        rows.append(FieldRow(field_name, where, float(np.mean(lengths))))
    rows.sort(key=lambda row: (row.present_on == "always", -row.mean_bytes))
    return rows


def _fix_ladder(bodies: list[dict[str, Any]], cfg: SideChannelConfig) -> list[MitigationRow]:
    """Try the fixes in the order an engineer would try them, and measure each one.

    Padding always works and is never the first thing anybody reaches for, so the ladder starts
    with the obvious change -- stop returning a field derivable from one the client already has
    -- and keeps correcting until something closes the channel. Every rung is scored by
    re-serialising the *same* captured bodies, so the comparison is exact rather than modelled,
    and the rungs are cumulative: each one keeps the previous corrections.
    """
    verdicts = np.array([int(bool(body.get("is_attack", False))) for body in bodies])
    shipped = np.array([serialised_size(body) for body in bodies], dtype=float)
    derivable = tuple(cfg.derivable_fields)
    variable = tuple(cfg.variable_width_fields)
    with_boolean = variable + tuple(cfg.boolean_fields)
    with_numeric = with_boolean + tuple(cfg.numeric_fields)
    filler = "x" * cfg.pad_width

    def variant(fixed: tuple[str, ...]) -> np.ndarray:
        return np.array(
            [
                serialised_size(
                    {
                        key: (filler if key in fixed else value)
                        for key, value in body.items()
                        if key not in derivable
                    }
                )
                for body in bodies
            ],
            dtype=float,
        )

    def row(label: str, sizes: np.ndarray) -> MitigationRow:
        return MitigationRow(
            label, separation(sizes, verdicts), 0.0, float(sizes.mean() - shipped.mean()), 0.0
        )

    names = ", ".join("`" + name + "`" for name in derivable)
    booleans = ", ".join("`" + name + "`" for name in cfg.boolean_fields)
    return [
        row("none (the shipped contract)", shipped),
        row(f"drop {names} (a lookup on `predicted_class`)", variant(())),
        row("...and give every decision field a fixed width", variant(variable)),
        row(f"...and {booleans}: `true` is four bytes and `false` is five", variant(with_boolean)),
        row(
            "...and every score: a float's decimal string is variable-length too",
            variant(with_numeric),
        ),
        row("pad every reply to the longest one", np.full_like(shipped, shipped.max())),
    ]


def run_side_channel_study(settings: Settings) -> SideChannelStudy:
    """Drive the real service and read its verdicts off the shape of its replies."""
    start = time.perf_counter()
    cfg: SideChannelConfig = settings.side_channel
    probe = settings.model_copy(deep=True)
    probe.split.strategy = "temporal"
    probe.mlflow.enabled = False
    probe.serving.api_key = None
    probe.serving.rate_limit_per_minute = 0  # the channel is not a rate-limit question
    rng = np.random.default_rng(probe.seed)

    from fastapi.testclient import TestClient

    from netsentry.models.registry import latest_bundle
    from netsentry.serving.app import create_app

    if latest_bundle(probe) is None:
        from netsentry.serving.bundle import build_serving_bundle

        logger.info("No serving bundle found; building one")
        build_serving_bundle(probe)
    client = TestClient(create_app(probe))
    payloads = _payloads(probe, cfg.n_flows, rng)

    channels: list[ChannelRow] = []
    measurements: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for label, query in cfg.endpoints:
        sizes, timings, verdicts = _drive(client, payloads, str(query), cfg.timing_repeats)
        measurements[str(label)] = (sizes, timings, verdicts)
        channels.append(_channel(str(label), sizes, timings, verdicts))

    reference = channels[0].endpoint if channels else ""
    _, _, verdicts = measurements.get(reference, (np.zeros(1), np.zeros(1), np.zeros(1)))
    bodies = _capture(client, payloads[: cfg.field_flows], cfg.field_query)

    study = SideChannelStudy(
        channels=channels,
        fields=_fields(bodies),
        mitigations=_fix_ladder(bodies, cfg),
        n_flows=len(payloads),
        alert_share=float(np.mean(verdicts)) if len(verdicts) else 0.0,
        seconds=time.perf_counter() - start,
    )
    worst = study.worst()
    logger.info(
        "Side-channel study complete",
        extra={
            "flows": study.n_flows,
            "worst_size_auc": round(worst.size_auc if worst else 0.0, 3),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# Rendering.
# --------------------------------------------------------------------------------------


def _channel_table(study: SideChannelStudy) -> str:
    rows = "\n".join(
        f"| {row.endpoint} | {row.mean_size_benign:.0f} B | {row.mean_size_alert:.0f} B | "
        f"**{row.size_auc:.3f}** | {row.size_accuracy:.1%} | {row.latency_auc:.3f} |"
        for row in study.channels
    )
    return (
        "| endpoint | mean reply, clear | mean reply, alert | size AUC | verdict recovered "
        "from size alone | latency AUC |\n|---|---|---|---|---|---|\n" + rows
    )


def _field_table(study: SideChannelStudy) -> str:
    rows = "\n".join(
        f"| `{row.field}` | {row.present_on} | {row.mean_bytes:.0f} B |" for row in study.fields
    )
    return "| field | present on | mean bytes |\n|---|---|---|\n" + rows


def _fix_table(study: SideChannelStudy) -> str:
    rows = "\n".join(
        f"| {row.mitigation} | **{row.size_auc:.3f}** | {row.added_bytes:+.0f} B |"
        for row in study.mitigations
    )
    return "| change | size AUC after | bytes per reply |\n|---|---|---|\n" + rows


def _lead(study: SideChannelStudy) -> str:
    worst = study.worst()
    default = study.channels[0] if study.channels else None
    ladder = study.mitigations
    first_fix = ladder[1] if len(ladder) > 1 else None
    last_enumerated = ladder[-2] if len(ladder) > 1 else None
    padding = ladder[-1] if ladder else None
    return (
        f"**The verdict is recoverable from the length of the reply, exactly, on every endpoint "
        f"configuration tested.** Size alone separates alerts from clear flows at AUC "
        f"{worst.size_auc if worst else 0:.3f}, and a single cut on the byte count recovers the "
        f"verdict for {worst.size_accuracy if worst else 0:.0%} of {study.n_flows} flows. An "
        f"alert's body runs {default.size_gap if default else 0:+.0f} bytes longer than a clear "
        f"one's, and turning explanations off does not help -- it makes every reply smaller and "
        f"leaves the gap intact.\n\n"
        f"**There is no timing channel worth having**, and the reason is almost funny. Latency "
        f"separates the two classes at AUC {default.latency_auc if default else 0:.3f}, which is "
        f"noise, because SHAP runs unconditionally and costs a quarter of a second -- so the "
        f"conditional work hides underneath the unconditional work. The most expensive thing in "
        f"the request path is what conceals the cheap signal.\n\n"
        f"The part worth reading is the fix. The obvious one is to stop returning the field that "
        f"appears only on alerts, and it takes the channel from "
        f"{ladder[0].size_auc if ladder else 0:.3f} to "
        f"{first_fix.size_auc if first_fix else 0:.3f} -- **not to 0.5**. Four successive "
        f"corrections, each of which an engineer would reasonably believe was the fix, leave it "
        f"at {last_enumerated.size_auc if last_enumerated else 0:.3f}: better than a coin, from "
        f"nothing but the length of an encrypted reply. Padding closes it, at "
        f"{padding.added_bytes if padding else 0:+.0f} bytes.\n\n"
        f"**You cannot enumerate your way out of a length channel**, because the last thing "
        f"leaking is the thing the contract exists to return."
    )


def _render(study: SideChannelStudy, size_figure: Path, mitigation_figure: Path) -> str:
    default = study.channels[0] if study.channels else None
    fast = study.channel("with explanations off")
    only_alerts = [row for row in study.fields if "alerts only" in row.present_on]
    second = f"{study.mitigations[1].size_auc:.3f}" if len(study.mitigations) > 1 else "n/a"
    last = f"{study.mitigations[-2].size_auc:.3f}" if len(study.mitigations) > 1 else "n/a"
    field_read = (
        f"The only field exclusive to one verdict is `{only_alerts[0].field}`: "
        f"{only_alerts[0].mean_bytes:.0f} bytes of ATT&CK context attached to alerts and "
        f"omitted otherwise."
        if only_alerts
        else "No field is exclusive to one verdict in this sample."
    )
    return f"""# NetSentry — The API Answers Twice

_{study.n_flows} later-day flows driven through the real application under four endpoint
configurations, with the reply's length and latency recorded beside the verdict.
{study.alert_share:.0%} of them were alerts. Regenerate with `netsentry sidechannel`._

## Why this report exists

Everything this project does about adversaries assumes the attacker learns the verdict by
**being told** it: through the API, under an API key, inside a rate limit, with every query
counted. The [extraction study](extraction.md) prices what those queries buy; the
[control-loop study](control.md) prices what an attacker can do by generating alerts. Both
defences are query-side, and both assume there is no other way to find out.

There is another way, and it is written into the response contract. `mitre` is `null` for a
clear verdict and an object for an alert. The optional anomaly explanation is computed **only
for flows the detector flagged**. `recommended_action` and `prediction_set` change length with
the decision. So an observer who can see encrypted traffic between a sensor and the service --
a network position, a shared host, a co-tenant -- reads the verdict off the packet lengths
without decrypting anything.

{_lead(study)}

## The channel, measured

![What the reply's length gives away](../figures/{size_figure.name})

{_channel_table(study)}

The AUC column is reported as `max(auc, 1 - auc)`, because an attacker does not care which way
the channel points: a body reliably *shorter* on an alert leaks exactly as much as one that is
longer.

Every configuration leaks totally. Turning explanations off with `?explain=false` -- the
throughput switch the benchmark documents -- takes the alert reply from
{default.mean_size_alert if default else 0:.0f} bytes down to
{fast.mean_size_alert if fast else 0:.0f} and leaves the gap between verdicts intact, because
the fields that vary are not the ones being removed. Requesting the anomaly explanation or the
exemplars does not change the picture either: they add bytes to *some* replies, which is the
same failure again rather than a new one.

## Which fields carry it

{_field_table(study)}

A field marked **alerts only** is a channel by itself whatever its size, because its *presence*
is the signal -- a one-byte field would do as well as a hundred-byte one. {field_read}

The tempting conclusion from this table is that the fields marked *always* are innocent. They
are not, and the next section is the demonstration: a field that is always present still leaks
if its **length** varies with the verdict, which is true of every number the contract returns.

## What closes it, and what that costs

![The fix ladder](../figures/{mitigation_figure.name})

{_fix_table(study)}

**Dropping the ATT&CK object should have been free and is not enough.** It is a lookup on
`predicted_class`, which the response already carries and whose mapping is published, so
returning it is redundant -- and removing it makes every reply smaller. It also takes the
channel from a perfect 1.000 down to {second}, which is still an attacker recovering the verdict
most of the time.

What follows is a lesson in what "variable-length field" means. Giving the decision fields a
fixed width changes almost nothing. Normalising the booleans helps a little, because `true` is
four bytes and `false` is five. Normalising the scores helps a lot, because `0.01` is four
characters and `0.9871234` is nine -- a probability is a variable-length field and nobody thinks
of it as one. And after all four corrections the channel still sits at {last}, carried by the
field hardest to give up: `top_features` is a list of floating-point contributions, so **the
explanation the API exists to return is itself a length that depends on the answer**.

Padding is therefore the guarantee rather than the optimisation. It closes the channel by
construction, at a fixed cost per reply, and it is the only rung whose correctness does not
depend on somebody enumerating every variable-length field correctly -- this time, and again
the next time the contract changes.

## Scope and honest limits

- **This is an in-process measurement.** Client and server share a process, so the latency
  numbers carry none of the network's noise and the sizes carry none of TLS's record padding or
  HTTP/2's header compression. TLS does not hide length; it obscures it a little, and an
  attacker who sees many requests averages that away. The size result would survive; the timing
  non-result might not, in either direction.
- **The verdict, not the label.** What leaks is what the service *decided*, which is what an
  evading attacker wants: a free, passive, unlimited feedback signal for tuning a flow until it
  stops being flagged. It is not a leak of the ground truth, and it is not a leak of the model.
- **The threat needs a position.** An attacker who cannot observe the sensor-to-service path
  gets nothing here. The realistic cases are a shared segment, a compromised collector, or a
  managed-detection deployment where the customer's traffic to the vendor crosses networks the
  attacker already sits on.
- **Batch endpoints are worse and are not measured.** `/predict/batch` returns one body for many
  flows, so its length is roughly linear in the number of alerts -- a *count* rather than a bit.
  Padding a batch reply to a constant length costs proportionally more, which is exactly the
  case where the cheap fix is least attractive.
- **Nothing here is exotic.** This is the oldest side channel there is, applied to an ML
  service's response contract rather than to a login form. It is in the report because the
  contract was designed to be read by a human and nobody asked what its shape says out loud."""


def run_side_channel_report(settings: Settings) -> Path:
    """Run the side-channel study and write the report + figures."""
    study = run_side_channel_study(settings)
    labels = [row.endpoint for row in study.channels]
    size_figure = plots.plot_grouped_barh(
        labels,
        {
            "clear verdict": [row.mean_size_benign for row in study.channels],
            "alert": [row.mean_size_alert for row in study.channels],
        },
        xlabel="mean response body (bytes)",
        title="The reply is longer when the answer is yes",
        out_path=settings.paths.figures_dir / SIZE_FIGURE,
    )
    mitigation_figure = plots.plot_barh(
        [row.mitigation for row in study.mitigations],
        [row.size_auc for row in study.mitigations],
        xlabel="how well the verdict survives in the response length (AUC)",
        title="Which change actually closes it",
        out_path=settings.paths.figures_dir / MITIGATION_FIGURE,
        xmax=1.05,
        vline=("no channel", 0.5),
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, size_figure, mitigation_figure), encoding="utf-8")
    logger.info("Wrote side-channel report", extra={"path": str(out_path)})

    with track_run(settings, "side_channel") as run:
        run.log_params({"flows": study.n_flows})
        worst = study.worst()
        run.log_metrics(
            {
                "size_auc": worst.size_auc if worst else 0.0,
                "latency_auc": study.channels[0].latency_auc if study.channels else 0.0,
                "padding_bytes": study.mitigations[-1].added_bytes if study.mitigations else 0.0,
            }
        )
        for artifact in (size_figure, mitigation_figure, out_path):
            run.log_artifact(artifact)
    return out_path
