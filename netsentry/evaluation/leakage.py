"""Leakage attribution — reproduce the field's inflated number and price each source.

This is the executable form of the project's whole thesis. Most public CIC-IDS2017
write-ups report ~99% and it is almost always manufactured by leakage: a shuffled
split that straddles attack bursts, a ``Destination Port`` the model memorises, and
identifier columns (Flow ID, Source IP) that co-vary with the attack campaign. The
rest of the project *avoids* those; this study deliberately *adds them back*, one at a
time, on top of the honest temporal model — and reports the PR-AUC each one buys. The
ladder turns "we don't leak" from an assertion into a decomposition: here is the
honest number, here is the field's number, and here is exactly what separates them.

The ladder (raw-score PR-AUC, the honest scale the headline evaluation uses):

1. **Honest** — temporal split, no port, no identifier (the README headline).
2. **+ shuffled split** — the single biggest leak: a stratified random split lets
   near-duplicate flows from one attack burst land on both sides.
3. **+ Destination Port** — the borderline feature the headline model drops, added
   back as a categorical so the model can memorise "attack X hit port Y."
4. **+ session identifier** — a synthetic, per-(day, class) identifier standing in for
   Flow ID / Source IP: a value that co-varies with the attack campaign, exactly the
   real CIC leak. It is injected *only* on the shuffled ladder, because that is the
   honest finding — on the temporal split the later-day campaigns carry identifiers the
   model never saw, so the identifier leak needs the shuffled split to work at all.

The injected identifier is a controlled demonstration of the mechanism the leakage
firewall exists to stop; the pipeline never lets such a column through (``remainder=
"drop"``). The point is to *price* the anti-pattern, not to adopt it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from netsentry.data.clean import BINARY_TARGET, MULTICLASS_TARGET
from netsentry.data.split import load_split
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "leakage.md"
_DAY = "Day"


@dataclass
class LadderRung:
    """One rung of the leakage ladder: PR-AUC after adding a source, and its delta."""

    name: str
    source_added: str
    pr_auc: float
    delta: float  # PR-AUC gained over the rung below (the source's contribution)


def session_identifier(df: pd.DataFrame) -> np.ndarray:
    """A stable per-(day, class) numeric id — a stand-in for Flow ID / Source IP.

    Constant within an attack campaign and distinct across campaigns, so on a shuffled
    split (where a campaign's rows straddle train/test) the model memorises id -> label
    and it transfers. Derived by a stable hash so train and test agree on the code — it
    is a *session* identifier, not the label itself, mirroring how a reused IP encodes
    the campaign on the real data.
    """
    keys = df[_DAY].astype(str) + "|" + df[MULTICLASS_TARGET].astype(str)
    codes = keys.map(lambda k: int(hashlib.sha1(k.encode()).hexdigest()[:8], 16) % 100_000)
    result: np.ndarray = codes.to_numpy(dtype=float)
    return result


def _dense(x: object) -> np.ndarray:
    """Coerce a (possibly sparse) transformed matrix to a dense array."""
    to_array = getattr(x, "toarray", None)
    return to_array() if callable(to_array) else np.asarray(x)


def _pr_auc(
    settings: Settings,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Fit the binary model and return raw-score PR-AUC on test (the honest scale)."""
    seed_everything(settings.seed)
    model = SupervisedClassifier(settings).fit(x_train, y_train, eval_set=(x_val, y_val))
    scores = attack_probability(
        model.predict_proba(x_test), model.classes_, settings.labels.benign_label
    )
    return float(average_precision_score(y_test, scores))


def _load(
    settings: Settings, strategy: str, cap: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and row-cap a split's train/val/test frames (seeded subsample)."""
    frames = []
    for part in ("train", "val", "test"):
        frame = load_split(settings, strategy, part)
        if len(frame) > cap:
            frame = frame.sample(n=cap, random_state=settings.seed)
        frames.append(frame)
    return frames[0], frames[1], frames[2]


def run_leakage_report(settings: Settings) -> Path:
    """Build the leakage ladder, price each source, and write the report."""
    cfg = settings.leakage
    base = settings.model_copy(deep=True)
    base.supervised.task = "binary"

    # --- Rung 1: the honest temporal model (no port, no identifier). -------------
    honest = base.model_copy(deep=True)
    honest.features.encode_destination_port = False
    tr, va, te = _load(honest, "temporal", cfg.max_rows)
    pipe = build_pipeline(honest)
    y = (tr[BINARY_TARGET].to_numpy(), va[BINARY_TARGET].to_numpy(), te[BINARY_TARGET].to_numpy())
    pr_honest = _pr_auc(
        honest, pipe.fit_transform(tr), y[0], pipe.transform(va), y[1], pipe.transform(te), y[2]
    )

    # --- Rung 2: same model, shuffled (stratified) split. ------------------------
    tr, va, te = _load(honest, "stratified", cfg.max_rows)
    pipe = build_pipeline(honest)
    y = (tr[BINARY_TARGET].to_numpy(), va[BINARY_TARGET].to_numpy(), te[BINARY_TARGET].to_numpy())
    pr_shuffle = _pr_auc(
        honest, pipe.fit_transform(tr), y[0], pipe.transform(va), y[1], pipe.transform(te), y[2]
    )

    # --- Rung 3: shuffled split + Destination Port as a categorical feature. -----
    ported = base.model_copy(deep=True)
    ported.features.encode_destination_port = True
    tr, va, te = _load(ported, "stratified", cfg.max_rows)
    pipe = build_pipeline(ported)
    y = (tr[BINARY_TARGET].to_numpy(), va[BINARY_TARGET].to_numpy(), te[BINARY_TARGET].to_numpy())
    xt, xv, xte = (
        _dense(pipe.fit_transform(tr)),
        _dense(pipe.transform(va)),
        _dense(pipe.transform(te)),
    )
    pr_port = _pr_auc(ported, xt, y[0], xv, y[1], xte, y[2])

    # --- Rung 4: + injected session identifier (Flow ID / Source IP stand-in). ---
    id_tr, id_va, id_te = session_identifier(tr), session_identifier(va), session_identifier(te)
    xt_id = np.hstack([xt, id_tr.reshape(-1, 1)])
    xv_id = np.hstack([xv, id_va.reshape(-1, 1)])
    xte_id = np.hstack([xte, id_te.reshape(-1, 1)])
    pr_id = _pr_auc(ported, xt_id, y[0], xv_id, y[1], xte_id, y[2])

    rungs = [
        LadderRung("honest (temporal, no port)", "—", pr_honest, 0.0),
        LadderRung(
            "+ shuffled split", "shuffled train/test split", pr_shuffle, pr_shuffle - pr_honest
        ),
        LadderRung("+ Destination Port", "port memorisation", pr_port, pr_port - pr_shuffle),
        LadderRung(
            "+ session identifier", "identifier leak (Flow ID / Source IP)", pr_id, pr_id - pr_port
        ),
    ]
    for r in rungs:
        logger.info(
            "Leakage rung",
            extra={"rung": r.name, "pr_auc": round(r.pr_auc, 4), "delta": round(r.delta, 4)},
        )

    fig = plots.plot_barh(
        [r.source_added for r in rungs[1:]],
        [r.delta for r in rungs[1:]],
        xlabel="PR-AUC added by each leakage source (temporal baseline = honest)",
        title="What manufactures the field's ~99%",
        out_path=settings.paths.figures_dir / "leakage.png",
    )

    report = _render(rungs, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote leakage report", extra={"path": str(out_path)})

    with track_run(settings, "leakage") as run:
        run.log_metrics(
            {"pr_honest": pr_honest, "pr_shuffle": pr_shuffle, "pr_port": pr_port, "pr_id": pr_id}
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _table(rungs: list[LadderRung]) -> str:
    rows = ["| rung | leakage source added | PR-AUC | Δ PR-AUC |", "|---|---|---|---|"]
    for r in rungs:
        delta = "—" if r.delta == 0.0 else f"**{r.delta:+.3f}**"
        rows.append(f"| {r.name} | {r.source_added} | {r.pr_auc:.3f} | {delta} |")
    return "\n".join(rows)


def _read(rungs: list[LadderRung]) -> str:
    """Sign-aware summary so the prose cannot diverge from the measured ladder."""
    honest, top = rungs[0].pr_auc, rungs[-1].pr_auc
    biggest = max(rungs[1:], key=lambda r: r.delta)
    id_delta = rungs[3].delta
    id_note = (
        f"The identifier leak is the finisher: adding a per-campaign session id takes "
        f"PR-AUC to **{top:.3f}** (+{id_delta:.3f}), the near-perfect number the field "
        "reports — and it only works *because* the split is already shuffled, so the "
        "campaign's ids straddle train and test. On the temporal split the later-day "
        "campaigns carry ids the model never saw, and the same column is worthless: the "
        "identifier leak is a *consequence* of the split leak, not an independent one."
        if id_delta > 0.02
        else (
            f"The injected identifier adds little here ({id_delta:+.3f}); on the real "
            "CIC-IDS2017 captures, where a handful of IPs are reused within each attack "
            "burst, it is the dominant leak."
        )
    )
    return (
        f"The honest temporal model scores **{honest:.3f}**; stacking the three leakage "
        f"sources manufactures **{top:.3f}** — a gap of {top - honest:+.3f} PR-AUC that is "
        "entirely method, not skill. The largest single contributor is "
        f"**{biggest.source_added}** ({biggest.delta:+.3f}). {id_note}\n\n"
        "Read the other way: the ~99% that saturates this corner of the literature is not a "
        "strong model, it is a leaky protocol. Every rung here is something the rest of "
        "NetSentry deliberately refuses — the temporal split is the headline, `Destination "
        'Port` is dropped, and the `remainder="drop"` firewall discards any identifier that '
        "reaches the pipeline. This study is the price tag on each refusal."
    )


def _render(rungs: list[LadderRung], fig: Path) -> str:
    return f"""# NetSentry — Leakage Attribution

_Synthetic stand-in. Raw-score PR-AUC (attack vs benign), the same scale as the headline
evaluation. Each rung adds one leakage source on top of the honest temporal model and
reports the PR-AUC it buys — the field's ~99% reproduced and decomposed._

Most public CIC-IDS2017 projects report ~99% and it is almost always **manufactured**:
a shuffled split that straddles attack bursts, a `Destination Port` the model memorises,
and identifier columns that co-vary with the campaign. NetSentry avoids all three; this
study adds them back, one at a time, so the inflation is a decomposition rather than a
claim.

{_table(rungs)}

![What manufactures the field's ~99%](../figures/{fig.name})

## Read

{_read(rungs)}

## Scope

The session identifier is a **controlled injection** — a per-(day, class) code standing in
for the Flow ID / Source IP that leaks on the real captures — added only to demonstrate and
price the mechanism the leakage firewall exists to stop; the pipeline never lets such a
column through. On the real CIC-IDS2017 data the identifier leak is larger and the shuffled
split's advantage compounds it, so the honest-vs-leaky gap is wider still. The lesson is the
project's founding one: on this dataset a near-perfect score is overwhelmingly more likely to
be leakage than skill — which is why `netsentry gate` **fails** a PR-AUC above 0.999.
"""
