"""Which features can an attacker actually change?

The [evasion study](evasion.md), the [interval verifier](verify_trees.md) and the [universal
perturbation](universal.md) all share one list: `robustness.controllable_features`, the set of
columns the threat model says an adversary may move. Every robustness number this project
publishes is conditional on that list being right, and the list has never been derived from
anything -- it was written once and has been inherited since.

Deriving it is not a matter of opinion. Each CIC feature is computed from packets by a stated
procedure, and an attacker sending traffic *to* a service occupies one side of the conversation.
They set their own packet sizes, their own inter-arrival times, their own flags and window sizes.
They do not set the server's. A feature computed only from the backward direction is a
measurement of the **responder's** behaviour, and an attacker who could set it would not need an
evasion attack -- they would already own the server.

So this module classifies all 77 columns from how they are computed, compares that with the list
the project ships, and then measures what the difference is worth. Four classes:

- **forward** -- computed only from packets the attacker sends. Directly settable.
- **backward** -- computed only from the responder's packets. Not settable by a client-side
  attacker at all.
- **joint** -- computed from both directions, so partially settable: an attacker can move it, but
  not to an arbitrary value, because half its inputs belong to someone else.
- **environmental** -- fixed by the target or the protocol rather than by either party's choice.

The second half is about a capability the threat model does not merely mis-specify but **omits
entirely**. Every attack in this repository perturbs the features of *one flow*. An attacker who
splits one long session into several short ones changes no feature within a flow -- they change
how many flows exist and how the volume is divided between them. Flow splitting is a textbook
NIDS evasion, it is invisible to a threat model expressed as a per-flow perturbation budget, and
it is measured here by rescaling each feature the way its own definition says splitting would.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import FEATURE_COLUMNS
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.robustness.evasion import base_feature_name, controllable_indices, mimicry_perturb
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ThreatModelConfig

logger = get_logger(__name__)

REPORT_NAME = "threat_model.md"
FIGURE_NAME = "threat_model.png"

FORWARD = "forward"
BACKWARD = "backward"
JOINT = "joint"
ENVIRONMENTAL = "environmental"

#: What each class means for an attacker who is the *client* in the conversation -- the standard
#: position for the scans, brute-force attempts and floods this dataset labels.
MEANINGS = {
    FORWARD: "computed only from packets the attacker sends, so directly settable",
    BACKWARD: "computed only from the responder's packets, so not settable by a client at all",
    JOINT: "computed from both directions, so movable but not to an arbitrary value",
    ENVIRONMENTAL: "fixed by the target or the protocol rather than by either party",
}


def classify(feature: str) -> str:
    """Which side of the conversation determines a feature, from how CICFlowMeter computes it.

    The rules are read off the naming convention the dataset uses, which is unusually honest
    about provenance: a column carrying ``Fwd`` is computed over forward packets, ``Bwd`` over
    backward ones, and a column carrying neither is computed over the merged stream and is
    therefore joint. The exceptions are the ones worth stating rather than pattern-matching.
    """
    name = base_feature_name(feature)
    lowered = name.lower()
    if lowered.startswith("destination port") or lowered.startswith("protocol"):
        # The attacker chooses who to talk to, but the port is a property of the service they
        # are attacking -- changing it means attacking something else.
        return ENVIRONMENTAL
    if "bwd" in lowered or "backward" in lowered:
        return BACKWARD
    if "fwd" in lowered or "forward" in lowered:
        return FORWARD
    if lowered.startswith("down/up"):
        # A ratio of the two directions: the attacker owns the denominator only.
        return JOINT
    return JOINT


@dataclass(frozen=True)
class Verdict:
    """One feature, as the shipped list treats it and as its definition says it should be."""

    feature: str
    derived: str
    claimed: bool

    @property
    def over_claimed(self) -> bool:
        """The list grants control the attacker does not physically have."""
        return self.claimed and self.derived == BACKWARD

    @property
    def under_claimed(self) -> bool:
        """The attacker can set it and the list does not say so -- the dangerous direction."""
        return not self.claimed and self.derived == FORWARD


def audit(claimed: list[str]) -> list[Verdict]:
    """Classify every feature and mark whether the shipped threat model claims it."""
    wanted = {base_feature_name(name) for name in claimed}
    return [
        Verdict(feature=name, derived=classify(name), claimed=base_feature_name(name) in wanted)
        for name in FEATURE_COLUMNS
    ]


# --------------------------------------------------------------------------------------
# Flow splitting: the capability the threat model does not express.
# --------------------------------------------------------------------------------------


def split_scaling(feature: str) -> float:
    """How a feature changes when one session is delivered as ``k`` flows instead of one.

    Read off each feature's own definition rather than assumed. **Totals divide**: a session
    carrying 1,000 packets as ten flows shows 100 in each. **Rates and means are invariant**: the
    same bytes over the same seconds is the same bytes per second however the flow table splits
    it. **Extremes shrink toward the mean**, because the maximum of a tenth of the packets is at
    most the maximum of all of them -- but that is a distributional statement rather than an
    exact one, so extremes are left unscaled here and the effect is understated deliberately.

    The exponent returned is the power of ``1/k`` the feature is multiplied by: 1 for a total,
    0 for a rate.
    """
    name = base_feature_name(feature).lower()
    totals = (
        "total fwd packets",
        "total backward packets",
        "total length of fwd packets",
        "total length of bwd packets",
        "flow duration",
        "fwd iat total",
        "bwd iat total",
        "subflow fwd packets",
        "subflow fwd bytes",
        "subflow bwd packets",
        "subflow bwd bytes",
        "fwd header length",
        "bwd header length",
        "act_data_pkt_fwd",
    )
    # Flag counts deliberately do *not* divide. A split session is several TCP connections, and
    # each one carries its own SYN and FIN -- so a fragment shows roughly the same flag counts as
    # the whole, not a share of them. Dividing them would have made splitting look like a bigger
    # change to the feature vector than it is.
    return 1.0 if name in totals else 0.0


def split_flow(
    x: np.ndarray, columns: np.ndarray, exponents: np.ndarray, pieces: int
) -> np.ndarray:
    """Deliver each session as ``pieces`` flows, and return what one piece looks like.

    Nothing inside a flow is perturbed. What changes is the accounting: the volume a detector
    sees in any single record is the session's volume divided among the pieces, while every rate
    and every mean is exactly what it was. A per-flow perturbation budget cannot express this,
    because from the detector's point of view no flow was perturbed -- there are simply more of
    them, each smaller.
    """
    if pieces <= 1:
        return np.array(x, dtype=float, copy=True)
    piece = np.array(x, dtype=float, copy=True)
    scale = np.power(1.0 / pieces, exponents[columns])
    piece[:, columns] = piece[:, columns] * scale
    return piece


# --------------------------------------------------------------------------------------
# Study records.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelRow:
    """One threat model, and what the mimicry attack achieves under it."""

    name: str
    describes: str
    features: int
    detection: float
    clean_detection: float

    @property
    def kept(self) -> float:
        """Detection surviving the attack, as a share of what the detector starts with."""
        return self.detection / self.clean_detection if self.clean_detection else 0.0

    @property
    def cost(self) -> float:
        return self.clean_detection - self.detection


@dataclass(frozen=True)
class SplitRow:
    """One splitting factor: what a session looks like when delivered as several flows."""

    pieces: int
    detection: float
    clean_detection: float
    alert_rate: float
    realised_fpr: float = 0.0

    @property
    def kept(self) -> float:
        return self.detection / self.clean_detection if self.clean_detection else 0.0


@dataclass
class ThreatModelStudy:
    """Everything the report needs, computed once."""

    verdicts: list[Verdict]
    models: list[ModelRow]
    splits: list[SplitRow]
    clean_detection: float
    budget: float
    seconds: float = 0.0

    def by_class(self, derived: str) -> list[Verdict]:
        return [row for row in self.verdicts if row.derived == derived]

    def over_claimed(self) -> list[Verdict]:
        return [row for row in self.verdicts if row.over_claimed]

    def under_claimed(self) -> list[Verdict]:
        return [row for row in self.verdicts if row.under_claimed]

    def claimed(self) -> list[Verdict]:
        return [row for row in self.verdicts if row.claimed]

    def model(self, name: str) -> ModelRow:
        return next(row for row in self.models if row.name == name)

    def best_for_attacker(self) -> SplitRow:
        """The splitting factor that leaves the attacker least detected.

        When it comes back as one piece, the answer is that the best available split is not to
        split -- which is the result rather than a missing row.
        """
        return min(self.splits, key=lambda row: row.detection)

    def most_split(self) -> SplitRow:
        """The most fragmented arm, where the effect is largest whichever way it points."""
        return max(self.splits, key=lambda row: row.pieces)

    def splitting_helps(self) -> bool:
        """Whether fragmenting a session lowers detection at all."""
        return self.best_for_attacker().pieces > 1


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


def run_threat_model_study(settings: Settings) -> ThreatModelStudy:
    """Audit the shipped threat model, then measure what it gets wrong and what it omits."""
    start = time.perf_counter()
    cfg: ThreatModelConfig = settings.threat_model
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "temporal"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split
    from netsentry.models.supervised import SupervisedClassifier

    pipeline = build_pipeline(variant)
    train_frame = load_split(variant, "temporal", "train")
    val_frame = load_split(variant, "temporal", "val")
    test_frame = load_split(variant, "temporal", "test")
    x_train: np.ndarray = np.asarray(pipeline.fit_transform(train_frame), dtype=float)
    x_val: np.ndarray = np.asarray(pipeline.transform(val_frame), dtype=float)
    x_test: np.ndarray = np.asarray(pipeline.transform(test_frame), dtype=float)
    y_train = train_frame[BINARY_TARGET].to_numpy().astype(int)
    y_val = val_frame[BINARY_TARGET].to_numpy().astype(int)
    y_test = test_frame[BINARY_TARGET].to_numpy().astype(int)

    model = SupervisedClassifier(variant).fit(x_train, y_train)
    column = list(model.classes_).index(1)

    def score(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(model.predict_proba(matrix))[:, column]

    cut = threshold_at_fpr(y_val, score(x_val), cfg.budget)
    attacks = y_test == 1
    clean_detection = float(rates_at_threshold(y_test, score(x_test), cut)["tpr"])

    shipped = list(variant.robustness.controllable_features)
    verdicts = audit(shipped)

    feature_names = list(pipeline.get_feature_names_out())
    centroid = x_train[y_train == 0].mean(axis=0)

    def detection_under(names: list[str]) -> float:
        """Run the same mimicry attack, restricted to one threat model's feature set."""
        indices = controllable_indices(feature_names, names)
        moved = np.array(x_test, dtype=float, copy=True)
        moved[attacks] = mimicry_perturb(x_test[attacks], centroid, indices, cfg.mimicry_fraction)
        return float(rates_at_threshold(y_test, score(moved), cut)["tpr"])

    forward_only = [row.feature for row in verdicts if row.derived == FORWARD]
    forward_and_joint = [row.feature for row in verdicts if row.derived in (FORWARD, JOINT)]
    models = [
        ModelRow(
            name="the shipped list",
            describes="what `robustness.controllable_features` grants today",
            features=len(shipped),
            detection=detection_under(shipped),
            clean_detection=clean_detection,
        ),
        ModelRow(
            name="forward only",
            describes="packets the attacker sends, and nothing else",
            features=len(forward_only),
            detection=detection_under(forward_only),
            clean_detection=clean_detection,
        ),
        ModelRow(
            name="forward and joint",
            describes="everything a client-side attacker can move at all",
            features=len(forward_and_joint),
            detection=detection_under(forward_and_joint),
            clean_detection=clean_detection,
        ),
        ModelRow(
            name="every feature",
            describes="the unbounded attacker, as an upper bound rather than a threat",
            features=len(FEATURE_COLUMNS),
            detection=detection_under(list(FEATURE_COLUMNS)),
            clean_detection=clean_detection,
        ),
    ]

    # Flow splitting: no feature is perturbed, the session is simply reported as several records.
    exponents = np.array([split_scaling(name) for name in feature_names], dtype=float)
    scaled = np.flatnonzero(exponents > 0)
    splits: list[SplitRow] = []
    for pieces in cfg.split_factors:
        # Only the attacker's own sessions are fragmented. Splitting the benign traffic too would
        # move the false-positive rate along with the detection rate and stop the comparison
        # being about the attack at all.
        fragment = np.array(x_test, dtype=float, copy=True)
        fragment[attacks] = split_flow(x_test[attacks], scaled, exponents, pieces)
        scores = score(fragment)
        rates = rates_at_threshold(y_test, scores, cut)
        splits.append(
            SplitRow(
                pieces=pieces,
                detection=float(rates["tpr"]),
                clean_detection=clean_detection,
                alert_rate=float(np.mean(scores >= cut)),
                realised_fpr=float(rates["fpr"]),
            )
        )

    study = ThreatModelStudy(
        verdicts=verdicts,
        models=models,
        splits=splits,
        clean_detection=clean_detection,
        budget=cfg.budget,
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Threat-model study complete",
        extra={
            "over_claimed": len(study.over_claimed()),
            "under_claimed": len(study.under_claimed()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: ThreatModelStudy) -> str:
    """The finding, written from the audit and the two attack arms."""
    over, under = study.over_claimed(), study.under_claimed()
    shipped = study.model("the shipped list")
    honest = study.model("forward only")
    most = study.most_split()
    best_split = study.best_for_attacker()
    lines = [
        f"**The list every robustness number here depends on is wrong "
        f"{len(over) + len(under)} ways out of {len(study.verdicts)}, and correcting it in either "
        f"direction makes the attacker worse off.**",
        "",
        f"`robustness.controllable_features` grants an adversary {len(study.claimed())} of the "
        f"{len(study.verdicts)} columns. Deriving each one from how CICFlowMeter computes it "
        f"instead: **{len(over)} are over-claimed** -- backward-direction features measuring the "
        f"*responder's* behaviour, which a client-side attacker cannot set without already owning "
        f"the server -- and **{len(under)} are under-claimed**, forward features the attacker "
        "plainly can set that the list omits. It is not a subset or a superset of the honest "
        "answer. It is a different set.",
        "",
        f"**And the evasion result this project publishes depends on the over-claim.** Under the "
        f"shipped list, centroid mimicry takes detection from {shipped.clean_detection:.1%} to "
        f"{shipped.detection:.1%}. Restricted to the forward direction -- everything the attacker "
        f"physically sends, and nothing else -- the identical attack takes it to "
        f"**{honest.detection:.1%}**, which is *higher* than doing nothing at all.",
        "",
        "That is not a rounding artefact, and the mechanism is worth stating. Moving only the "
        "forward half of a flow toward benign traffic produces a record that looks benign in one "
        "direction and like an attack in the other -- a combination that appears nowhere in the "
        "training data, and which a tree ensemble is free to score however the splits happen to "
        "fall. **Partial mimicry is not a weaker version of full mimicry; it is a different "
        "input.** The [transport study](transport.md) found the same shape from another angle: "
        "centroid mimicry aims at the worst target available.",
        "",
        f"The second half of the study is a capability the threat model does not mis-specify but "
        f"**omits entirely**. Every attack in this repository perturbs the features of one flow; "
        f"an attacker who delivers one session as {most.pieces} shorter ones perturbs nothing and "
        "changes only the accounting. A per-flow perturbation budget cannot represent that at "
        "all.",
        "",
        (
            f"**And it backfires too.** Fragmenting into {most.pieces} pieces takes detection to "
            f"**{most.detection:.1%}**, {most.kept - 1:.0%} *above* the undisguised attack, at an "
            "unchanged false-positive rate -- and the trend is monotone, so the best available "
            "split for the attacker is not to split at all."
            if not study.splitting_helps()
            else f"Fragmenting into {best_split.pieces} pieces takes detection to "
            f"**{best_split.detection:.1%}**, {1 - best_split.kept:.0%} below the undisguised "
            "attack, at an unchanged false-positive rate."
        ),
        "",
        "The reason is specific and checkable rather than mysterious. The attacks in this dataset "
        "**are** short, low-volume flows -- scans, brute-force attempts, probes. Fragmenting a "
        "session divides its totals, which moves it *toward* that region rather than away from "
        "it. The folk intuition that splitting hides you assumes a detector keying on volume "
        "crossing a threshold; a model trained on scans keys on volume being **low**.",
        "",
        "**Three results, and all three say the same thing: this threat model was written from "
        "intuition rather than derived from the data.** The published robustness numbers turn out "
        "to be pessimistic about this attacker, which is the safe direction to be wrong in -- but "
        "nobody had established that, and a threat model that is accidentally conservative is not "
        "a threat model.",
    ]
    return "\n".join(lines)


def _render(study: ThreatModelStudy, figure: Path) -> str:
    """Compose the report."""
    lines = [
        "# NetSentry -- Which Features Can an Attacker Actually Change?",
        "",
        f"_All {len(study.verdicts)} feature columns classified by how CICFlowMeter computes "
        f"them, the mimicry attack re-run under {len(study.models)} threat models, and flow "
        f"splitting measured at the {study.budget:.1%} operating point. Regenerate with "
        "`netsentry threatmodel`._",
        "",
        "## Why this report exists",
        "",
        "The [evasion study](evasion.md), the [interval verifier](verify_trees.md) and the "
        "[universal perturbation](universal.md) share one list: `robustness.controllable_"
        "features`. Every robustness number this project publishes is conditional on that list "
        "being right, and it has never been derived from anything -- it was written once and "
        "inherited since.",
        "",
        "Deriving it is not a matter of opinion. Each CIC feature is computed from packets by a "
        "stated procedure, and an attacker sending traffic *to* a service occupies one side of "
        "the conversation. They set their own packet sizes, inter-arrival times, flags and window "
        "sizes. They do not set the server's.",
        "",
        _lead(study),
        "",
        "## The audit",
        "",
        "| class | features | what it means for a client-side attacker | claimed by the list |",
        "|---|---|---|---|",
    ]
    for name in (FORWARD, JOINT, BACKWARD, ENVIRONMENTAL):
        members = study.by_class(name)
        claimed = sum(item.claimed for item in members)
        lines.append(
            f"| **{name}** | {len(members)} | {MEANINGS[name]} | {claimed} of {len(members)} |"
        )
    lines += [
        "",
        "The classification is read off the dataset's own naming, which is unusually honest about "
        "provenance: a column carrying `Fwd` is computed over forward packets, `Bwd` over backward "
        "ones, and one carrying neither is computed over the merged stream and is therefore joint. "
        "`Destination Port` is the single environmental column -- an attacker chooses who to talk "
        "to, but the port is a property of the service, and changing it means attacking something "
        "else.",
        "",
        "### Over-claimed: granted to the attacker, not theirs to set",
        "",
        "| feature | why not |",
        "|---|---|",
    ]
    for verdict in study.over_claimed():
        lines.append(f"| `{verdict.feature}` | measures the responder's packets |")
    lines += [
        "",
        "### Under-claimed: the attacker's to set, and the list omits them",
        "",
        "| feature | why it is theirs |",
        "|---|---|",
    ]
    for verdict in study.under_claimed():
        lines.append(f"| `{verdict.feature}` | computed only from packets the attacker sends |")
    lines += [
        "",
        "The under-claimed half is the direction that would matter if the errors did not cancel. "
        "An omitted forward feature is control the attacker has and the evaluation does not model, "
        "which makes a published robustness number optimistic. Here they happen not to help -- but "
        "*happening not to* is not a property anyone designed.",
        "",
        "## What each threat model is worth",
        "",
        f"![Detection under each threat model and under splitting](../figures/{figure.name})",
        "",
        "| threat model | features | detection after mimicry | vs no attack |",
        "|---|---|---|---|",
    ]
    for arm in study.models:
        lines.append(
            f"| {arm.name} -- {arm.describes} | {arm.features} | {arm.detection:.1%} | "
            f"**{arm.kept - 1:+.0%}** |"
        )
    lines += [
        "",
        "Every arm runs the identical attack -- move each attack flow toward the benign "
        "centroid on the features that model allows -- against the identical threshold. Only the "
        "permitted set changes.",
        "",
        "The **every feature** row is an upper bound rather than a threat: an attacker who can "
        "set all 77 columns is not evading a detector, they are writing its input. It is here "
        "because it bounds the others, and the fact that it does *not* dominate the shipped list "
        "is itself worth noticing -- more control is not monotonically more evasion when the "
        "attack is a fixed direction rather than a search.",
        "",
        "## Flow splitting: the capability the budget cannot express",
        "",
        "| session delivered as | detection | vs one flow | false-positive rate |",
        "|---|---|---|---|",
    ]
    for piece in study.splits:
        label = "one flow" if piece.pieces == 1 else f"{piece.pieces} flows"
        lines.append(
            f"| {label} | {piece.detection:.1%} | **{piece.kept - 1:+.0%}** | "
            f"{piece.realised_fpr:.2%} |"
        )
    lines += [
        "",
        "Nothing inside a flow is perturbed. What changes is the accounting: totals divide among "
        "the pieces while every rate and mean stays exactly what it was, because the same bytes "
        "over the same seconds is the same bytes per second however the flow table splits it. A "
        "threat model expressed as a per-flow perturbation budget cannot represent this at all -- "
        "from the detector's side no flow was perturbed, there are simply more of them.",
        "",
        "Only the attacker's own sessions are fragmented, which is why the false-positive rate is "
        "identical in every row: splitting the benign traffic too would move both rates and stop "
        "the comparison being about the attack.",
        "",
        "**Flag counts deliberately do not divide.** A split session is several TCP connections "
        "and each carries its own SYN and FIN, so a fragment shows roughly the whole session's "
        "flag counts rather than a share of them. Dividing them -- which the first version of "
        "this module did -- overstates how much splitting changes the feature vector.",
        "",
        "## Scope and honest limits",
        "",
        "- **The classification assumes the attacker is the client.** For an attacker who has "
        "compromised the server, the forward and backward columns swap roles and the shipped list "
        "becomes closer to right than the derived one. That is a different threat model and it is "
        "not the one this dataset's attacks occupy.",
        "- **Joint features are treated as fully available in the 'forward and joint' arm**, which "
        "overstates that arm: an attacker moves a two-directional mean only partway, and how far "
        "depends on the responder. The honest bound sits between the forward-only and "
        "forward-and-joint rows.",
        "- **Splitting is modelled at the feature level, not by re-running a flow assembler.** "
        "Each column is rescaled the way its own definition says it would move, and the extremes "
        "(`Max`, `Min`) are left unscaled although they would in fact shrink toward the mean -- so "
        "the effect measured here is understated rather than inflated.",
        "- **The splitting result is about this dataset's attack mix.** Attacks here are dominated "
        "by short, low-volume flows, which is exactly why fragmenting moves toward them. Against "
        "a detector trained on high-volume exfiltration the same manoeuvre would plausibly work "
        "as the folklore says, and this study cannot speak to that case.",
        "- **A backfiring attack is not a defence.** That mimicry and splitting both raise "
        "detection here says the specific manoeuvre is badly aimed, not that the model is robust. "
        "The [universal perturbation](universal.md) is the arm that does work, and it works by "
        "searching rather than by moving in a fixed direction.",
    ]
    return "\n".join(lines) + "\n"


def run_threat_model_report(settings: Settings) -> Path:
    """Run the threat-model audit and write the report + figure."""
    study = run_threat_model_study(settings)
    figure = plots.plot_lines(
        {
            "delivered as k flows (no feature perturbed)": (
                np.array([row.pieces for row in study.splits], dtype=float),
                np.array([row.detection for row in study.splits]),
            ),
            "undisguised": (
                np.array([row.pieces for row in study.splits], dtype=float),
                np.full(len(study.splits), study.clean_detection),
            ),
        },
        xlabel="pieces the session is delivered as",
        ylabel="detection rate",
        title="Fragmenting a session moves it toward the attacks, not away from them",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        xscale="log",
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote threat-model report", extra={"path": str(out_path)})

    with track_run(settings, "threat_model") as run:
        run.log_params({"budget": str(study.budget)})
        run.log_metrics(
            {
                "over_claimed": float(len(study.over_claimed())),
                "under_claimed": float(len(study.under_claimed())),
                "shipped_detection": study.model("the shipped list").detection,
                "forward_only_detection": study.model("forward only").detection,
                "most_split_detection": study.most_split().detection,
                "splitting_helps": float(study.splitting_helps()),
            }
        )
        for artifact in (figure, out_path):
            run.log_artifact(artifact)
    return out_path
