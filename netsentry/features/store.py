"""A point-in-time-correct feature store: host context without borrowing from the future.

The per-flow model is deliberately identity-blind — IPs are dropped before anything is modelled,
which is what keeps it from memorising *which host* attacked instead of *what an attack looks
like*. That firewall costs something real: a single flow cannot express "this source has opened
four hundred connections in the last minute", which is the signal a human analyst would reach for
first. Host **context** features recover it without reintroducing identity, because the feature
is a behaviour count, not an address.

Computing them correctly is the hard part, and it is the part production ML infrastructure exists
to solve. The obvious implementation — group the whole capture by source and join the aggregates
back — is a **temporal leak**: a flow at 09:00 receives a count that includes flows from 17:00,
so the model is told about the future of the very host it is being asked to judge. It will score
beautifully offline and be unreproducible in production, where 17:00 has not happened yet. This
is precisely the failure mode a feature store's **as-of join** prevents (the point-in-time
correctness guarantee of Feast, Tecton and every serious feature platform), and it is the same
class of mistake as the identifier leakage this project was built to avoid — one axis over.

So the store implements both, and measures the difference:

- ``as_of_features`` computes each flow's context from **strictly earlier** events only, within a
  bounded lookback window, which is exactly what a serving path could compute at request time;
- ``leaky_features`` computes the same aggregates over the entire capture, which is what a
  notebook does in one line of pandas.

The window is enforced with a two-pointer sweep over time-sorted events per entity, so the cost
is linear rather than the quadratic of a naive per-row filter, and the same code path serves the
offline join and the online lookup — a store whose training and serving definitions can diverge
has reintroduced the skew it exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from netsentry.log import get_logger

logger = get_logger(__name__)

# The context features the store computes per (entity, time). Each is a behaviour count over a
# bounded window — never an identity — so the leakage firewall the project relies on holds.
FEATURE_NAMES = (
    "ctx_flows_in_window",
    "ctx_distinct_dest_ports",
    "ctx_distinct_dest_hosts",
    "ctx_mean_gap_seconds",
)


@dataclass(frozen=True)
class ContextEvent:
    """One prior connection by an entity: when it happened and what it touched."""

    time: float
    dest_host: str
    dest_port: int


def _aggregate(window: list[ContextEvent], now: float) -> list[float]:
    """The context vector implied by a window of prior events (empty window == a quiet host)."""
    if not window:
        return [0.0, 0.0, 0.0, 0.0]
    times = [e.time for e in window]
    gaps = np.diff(times) if len(times) > 1 else np.array([now - times[0]])
    return [
        float(len(window)),
        float(len({e.dest_port for e in window})),
        float(len({e.dest_host for e in window})),
        float(np.mean(gaps)) if len(gaps) else 0.0,
    ]


def as_of_features(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    time_column: str,
    dest_host_column: str,
    dest_port_column: str,
    lookback_seconds: float,
) -> pd.DataFrame:
    """Point-in-time-correct host context: each row sees only strictly earlier events.

    For every flow, the context is aggregated over that source's events in
    `[t - lookback, t)` — earlier, never simultaneous, never later. The exclusion of the row's
    own timestamp matters more than it looks: including it leaks the label-bearing flow into its
    own feature, and ties at one-second resolution would quietly do exactly that.

    Implemented as a two-pointer sweep over time-sorted events per entity, so it is linear in the
    number of flows rather than quadratic, and it is the same computation a serving path would
    run against a live window.
    """
    if lookback_seconds <= 0:
        raise ValueError("lookback_seconds must be positive")
    times = _epoch_seconds(frame[time_column])
    order = np.argsort(times, kind="stable")
    values = np.zeros((len(frame), len(FEATURE_NAMES)), dtype=float)

    entities = frame[entity_column].to_numpy()
    hosts = frame[dest_host_column].astype(str).to_numpy()
    ports = frame[dest_port_column].fillna(-1).astype(int).to_numpy()

    history: dict[object, list[ContextEvent]] = {}
    starts: dict[object, int] = {}
    for idx in order:
        entity = entities[idx]
        now = float(times[idx])
        events = history.setdefault(entity, [])
        start = starts.get(entity, 0)
        # Advance the window's left edge past anything older than the lookback.
        while start < len(events) and events[start].time < now - lookback_seconds:
            start += 1
        starts[entity] = start
        # Only events strictly before `now` are visible: a flow may not see its own instant.
        end = len(events)
        while end > start and events[end - 1].time >= now:
            end -= 1
        values[idx] = _aggregate(events[start:end], now)
        events.append(ContextEvent(now, str(hosts[idx]), int(ports[idx])))

    return pd.DataFrame(values, columns=list(FEATURE_NAMES), index=frame.index)


def leaky_features(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    dest_host_column: str,
    dest_port_column: str,
) -> pd.DataFrame:
    """The one-line pandas version: aggregate the whole capture per entity and join it back.

    Kept, named, and measured rather than merely warned about. Every flow receives its host's
    totals over the *entire* capture, including flows that had not happened yet when the flow
    being scored occurred. This is a temporal leak, and it is the default implementation almost
    everyone writes first.
    """
    grouped = frame.groupby(entity_column)
    totals = pd.DataFrame(
        {
            FEATURE_NAMES[0]: grouped[dest_host_column].size(),
            FEATURE_NAMES[1]: grouped[dest_port_column].nunique(),
            FEATURE_NAMES[2]: grouped[dest_host_column].nunique(),
        }
    )
    joined = frame[[entity_column]].join(totals, on=entity_column)
    joined[FEATURE_NAMES[3]] = 0.0
    return joined[list(FEATURE_NAMES)].astype(float)


def _epoch_seconds(column: pd.Series) -> np.ndarray:
    """Timestamps as float seconds, tolerant of the several formats CIC-IDS files ship with."""
    parsed = pd.to_datetime(column, errors="coerce", format="mixed")
    seconds: np.ndarray = parsed.astype("int64").to_numpy(dtype=float) / 1e9
    filled: np.ndarray = np.nan_to_num(seconds, nan=0.0)
    return filled


def leakage_gap(as_of: pd.DataFrame, leaky: pd.DataFrame) -> dict[str, float]:
    """Per-feature mean ratio between the leaky and point-in-time versions of the same column.

    A ratio far above 1 says the leaky join is handing each flow a count it could not possibly
    have had — the size of the lie, before any model is fit on it.
    """
    gaps = {}
    for name in FEATURE_NAMES:
        pit_mean = float(np.mean(as_of[name]))
        gaps[name] = float(np.mean(leaky[name])) / pit_mean if pit_mean > 0 else float("inf")
    return gaps
