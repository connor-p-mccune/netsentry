"""Would the attack taxonomy have been discovered without anyone labelling it?

The anomaly detector's job ends with a pile: flows that do not look like normal traffic,
handed to an analyst one at a time with no structure. That is the actual shape of novel-attack
detection in production — you find out *that* something is wrong long before anyone tells you
*what*. The pile is also where analyst time goes to die, because a hundred flows from one
scanning campaign arrive as a hundred separate tickets.

Clustering is the obvious response and the obvious trap. Grouping the anomalies is easy;
knowing whether the groups mean anything is not, and the usual failure is to pick the number
of clusters by checking which value best reproduces the labels — which is exactly the leakage
this project exists to avoid, dressed up as unsupervised learning. So the protocol here is
strict about the boundary: **`k` is chosen by silhouette score on the unlabelled features
alone**, the clustering never sees a label, and the labels appear only afterwards, to grade a
decision that was already made.

Three questions, in order of how much they matter:

1. **Do the clusters correspond to real attack families?** Adjusted Rand Index and normalized
   mutual information against the true classes, with a random-assignment control so "0.31"
   has a reference point. Purity per cluster says the same thing per-group.
2. **How many known families would have been rediscovered?** A family counts as discovered if
   some cluster is predominantly made of it — the concrete version of "would we have found
   this without being told".
3. **What does it save?** The triage ratio: flagged flows divided by clusters, which is how
   many tickets an analyst avoids opening if campaigns are triaged as groups.

Each cluster is also named by its nearest known-class centroid, so a cluster that sits far
from every known family is flagged as a candidate *new* family rather than silently
mislabelled as the closest thing — the distinction between recognising an attack and merely
matching it to the nearest thing in the catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

from netsentry.data.clean import BINARY_TARGET
from netsentry.data.schema import LABEL_COLUMN
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import attack_probability, threshold_at_fpr
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.supervised import SupervisedClassifier
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import DiscoveryConfig

logger = get_logger(__name__)

REPORT_NAME = "discovery.md"
FIGURE_NAME = "discovery.png"


def choose_k(
    x: np.ndarray, candidates: list[int], seed: int, sample: int
) -> tuple[int, dict[int, float]]:
    """Pick the cluster count by silhouette score — using no labels whatsoever.

    This is the methodological crux. Selecting ``k`` by whichever value best reproduces the
    known classes would make the whole study circular, so the choice is made on geometry
    alone: silhouette measures how much better each point fits its own cluster than the next
    nearest, and needs only the features. The scores are reported so the choice can be
    second-guessed.
    """
    scores: dict[int, float] = {}
    rng = np.random.default_rng(seed)
    idx = (
        rng.choice(len(x), size=sample, replace=False)
        if sample and len(x) > sample
        else np.arange(len(x))
    )
    for k in candidates:
        if k < 2 or k >= len(x):
            continue
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(x)
        if len(np.unique(labels[idx])) < 2:
            continue
        scores[k] = float(silhouette_score(x[idx], labels[idx]))
    if not scores:
        return max(2, min(candidates)), scores
    return max(scores, key=lambda k: scores[k]), scores


def cluster_purity(cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
    """Share of flows sitting in a cluster whose dominant true class is their own.

    The standard purity measure: sum the majority count of each cluster, divide by the total.
    It rewards fine-grained clusterings (one cluster per flow is perfectly pure), which is why
    it is reported next to ARI, which does not.
    """
    cl = np.asarray(cluster_labels)
    tl = np.asarray(true_labels)
    if len(cl) == 0:
        return 0.0
    total = 0
    for c in np.unique(cl):
        members = tl[cl == c]
        if len(members):
            _, counts = np.unique(members, return_counts=True)
            total += int(counts.max())
    return total / len(cl)


def discovered_families(
    cluster_labels: np.ndarray, true_labels: np.ndarray, min_purity: float, min_size: int
) -> list[str]:
    """Attack families that a cluster would have surfaced on its own.

    A family counts as discovered when some cluster is at least ``min_purity`` made of it and
    holds at least ``min_size`` flows — the concrete form of "an analyst opening this cluster
    would have found a coherent campaign", rather than a statistical coincidence in a group
    of three.
    """
    cl = np.asarray(cluster_labels)
    tl = np.asarray(true_labels)
    found: set[str] = set()
    for c in np.unique(cl):
        members = tl[cl == c]
        if len(members) < min_size:
            continue
        values, counts = np.unique(members, return_counts=True)
        dominant, share = values[counts.argmax()], counts.max() / len(members)
        if share >= min_purity:
            found.add(str(dominant))
    return sorted(found)


def random_baseline_ari(true_labels: np.ndarray, k: int, seed: int, trials: int) -> float:
    """Mean ARI of a random assignment into ``k`` groups — the reference any score needs.

    ARI is chance-corrected by construction, so this should land near zero; measuring it
    anyway is what turns "we got 0.31" into a claim rather than a number.
    """
    rng = np.random.default_rng(seed)
    tl = np.asarray(true_labels)
    return float(
        np.mean([adjusted_rand_score(tl, rng.integers(0, k, size=len(tl))) for _ in range(trials)])
    )


def nearest_known_class(
    centroid: np.ndarray, class_centroids: dict[str, np.ndarray]
) -> tuple[str, float]:
    """Name a cluster by the closest known-class centroid, and return that distance.

    The distance is the interesting half. A cluster that sits far from every known family is a
    candidate *new* family; naming it after the nearest catalogue entry regardless would be
    exactly the mistake that makes a novel-attack detector useless.
    """
    if not class_centroids:
        return "unknown", float("inf")
    best, best_d = "unknown", float("inf")
    for name, c in class_centroids.items():
        d = float(np.linalg.norm(centroid - c))
        if d < best_d:
            best, best_d = name, d
    return best, best_d


@dataclass
class ClusterSummary:
    """One discovered cluster: its size, what it actually contained, and what it looks like."""

    cluster_id: int
    size: int
    dominant_class: str
    purity: float
    nearest_known: str
    distance: float
    novel: bool


@dataclass
class DiscoveryStudy:
    """Everything the report renders."""

    n_flagged: int
    fpr_budget: float
    k: int
    silhouettes: dict[int, float]
    ari: float
    nmi: float
    purity: float
    random_ari: float
    families_present: list[str]
    families_discovered: list[str]
    triage_ratio: float
    clusters: list[ClusterSummary]
    novel_distance_threshold: float
    min_purity: float
    oracle_k: int
    oracle_ari: float
    oracle_purity: float
    oracle_discovered: int

    @property
    def discovery_rate(self) -> float:
        if not self.families_present:
            return 0.0
        return len(self.families_discovered) / len(self.families_present)


def run_discovery(settings: Settings) -> DiscoveryStudy:
    """Cluster the flagged flows without labels, then grade the clustering against them."""
    cfg: DiscoveryConfig = settings.discovery
    variant = settings.model_copy(deep=True)
    variant.split.strategy = "stratified"
    variant.supervised.task = "binary"
    variant.mlflow.enabled = False
    seed_everything(variant.seed)

    from netsentry.data.split import load_split

    train = load_split(variant, "stratified", "train")
    val = load_split(variant, "stratified", "val")
    test = load_split(variant, "stratified", "test")
    y_train = train[BINARY_TARGET].to_numpy().astype(int)
    y_val = val[BINARY_TARGET].to_numpy().astype(int)
    benign = variant.labels.benign_label

    pipeline = build_pipeline(variant)
    x_train = np.asarray(pipeline.fit_transform(train))
    x_val = np.asarray(pipeline.transform(val))
    x_test = np.asarray(pipeline.transform(test))
    model = SupervisedClassifier(variant).fit(x_train, y_train, eval_set=(x_val, y_val))

    def _scores(x: np.ndarray) -> np.ndarray:
        return attack_probability(np.asarray(model.predict_proba(x)), model.classes_, benign)

    threshold = threshold_at_fpr(y_val, _scores(x_val), cfg.flag_fpr)
    flagged = _scores(x_test) >= threshold
    x_flag = x_test[flagged]
    labels_flag = test[LABEL_COLUMN].to_numpy()[flagged]
    if cfg.max_flows and len(x_flag) > cfg.max_flows:
        keep = np.random.default_rng(variant.seed).choice(
            len(x_flag), size=cfg.max_flows, replace=False
        )
        x_flag, labels_flag = x_flag[keep], labels_flag[keep]

    k, silhouettes = choose_k(x_flag, cfg.k_candidates, variant.seed, cfg.silhouette_sample)
    km = KMeans(n_clusters=k, random_state=variant.seed, n_init=10).fit(x_flag)
    assignments = km.labels_

    # Known-class centroids come from the *training* split — the catalogue a deployed system
    # would have — so naming a cluster never consults the data being clustered.
    train_labels = train[LABEL_COLUMN].to_numpy()
    class_centroids = {
        str(name): x_train[train_labels == name].mean(axis=0)
        for name in np.unique(train_labels)
        if str(name) != benign and int((train_labels == name).sum()) >= cfg.min_class_rows
    }
    distances = []
    summaries: list[ClusterSummary] = []
    for c in range(k):
        members = labels_flag[assignments == c]
        if len(members) == 0:
            continue
        values, counts = np.unique(members, return_counts=True)
        dominant = str(values[counts.argmax()])
        nearest, dist = nearest_known_class(km.cluster_centers_[c], class_centroids)
        distances.append(dist)
        summaries.append(
            ClusterSummary(
                cluster_id=c,
                size=len(members),
                dominant_class=dominant,
                purity=float(counts.max() / len(members)),
                nearest_known=nearest,
                distance=dist,
                novel=False,
            )
        )
    # "Novel" is relative: a cluster is a candidate new family if it sits further from every
    # known centroid than most clusters do. An absolute distance would be unit-dependent.
    finite = [d for d in distances if np.isfinite(d)]
    novel_cut = float(np.quantile(finite, cfg.novel_quantile)) if finite else float("inf")
    for s in summaries:
        s.novel = bool(np.isfinite(s.distance) and s.distance >= novel_cut)

    attack_mask = labels_flag != benign
    families_present = sorted({str(x) for x in labels_flag[attack_mask]})
    discovered = discovered_families(
        assignments[attack_mask], labels_flag[attack_mask], cfg.min_purity, cfg.min_cluster_size
    )

    # Diagnostic arm: refit at k = the true number of families. This **uses the labels** and
    # is therefore not a result — it exists only to separate two very different failures. If
    # the oracle-k clustering also scores near zero, the feature geometry does not encode the
    # taxonomy and no selector could have found it; if it scores well, the geometry is fine
    # and silhouette simply chose badly.
    oracle_k = max(2, min(len(families_present), len(x_flag) - 1))
    oracle_assign = KMeans(n_clusters=oracle_k, random_state=variant.seed, n_init=10).fit_predict(
        x_flag
    )
    oracle_found = discovered_families(
        oracle_assign[attack_mask], labels_flag[attack_mask], cfg.min_purity, cfg.min_cluster_size
    )
    logger.info("Discovery clustering complete", extra={"k": k, "n_flagged": len(labels_flag)})

    return DiscoveryStudy(
        n_flagged=len(labels_flag),
        fpr_budget=cfg.flag_fpr,
        k=k,
        silhouettes=silhouettes,
        ari=float(adjusted_rand_score(labels_flag, assignments)),
        nmi=float(normalized_mutual_info_score(labels_flag, assignments)),
        purity=cluster_purity(assignments, labels_flag),
        random_ari=random_baseline_ari(labels_flag, k, variant.seed, cfg.baseline_trials),
        families_present=families_present,
        families_discovered=discovered,
        triage_ratio=len(labels_flag) / max(k, 1),
        clusters=sorted(summaries, key=lambda s: -s.size),
        novel_distance_threshold=novel_cut,
        min_purity=cfg.min_purity,
        oracle_k=oracle_k,
        oracle_ari=float(adjusted_rand_score(labels_flag, oracle_assign)),
        oracle_purity=cluster_purity(oracle_assign, labels_flag),
        oracle_discovered=len(oracle_found),
    )


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def run_discovery_report(settings: Settings) -> Path:
    """Run the discovery study and write the report + figure."""
    study = run_discovery(settings)

    ks = np.array(sorted(study.silhouettes))
    fig = plots.plot_lines(
        {
            "silhouette (chosen without labels)": (
                ks,
                np.array([study.silhouettes[int(k)] for k in ks]),
            )
        },
        xlabel="number of clusters k",
        ylabel="mean silhouette score",
        title=f"k chosen on geometry alone; the labels never voted (k = {study.k})",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
        vlines={f"chosen k = {study.k}": float(study.k)},
    )

    report = _render(study, fig)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote discovery report", extra={"path": str(out_path)})

    with track_run(settings, "discovery") as run:
        run.log_params({"k": study.k, "flag_fpr": study.fpr_budget})
        run.log_metrics(
            {
                "ari": study.ari,
                "nmi": study.nmi,
                "purity": study.purity,
                "random_ari": study.random_ari,
                "discovery_rate": study.discovery_rate,
                "triage_ratio": study.triage_ratio,
            }
        )
        run.log_artifact(fig)
        run.log_artifact(out_path)
    return out_path


def _cluster_table(study: DiscoveryStudy) -> str:
    rows = [
        "| cluster | flows | what it actually contained | purity | nearest known family "
        "| distance | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in study.clusters:
        dist = "n/a" if not np.isfinite(s.distance) else f"{s.distance:.1f}"
        verdict = "**candidate new family**" if s.novel else "matches the catalogue"
        rows.append(
            f"| {s.cluster_id} | {s.size:,} | {s.dominant_class} | {s.purity:.0%} "
            f"| {s.nearest_known} | {dist} | {verdict} |"
        )
    return "\n".join(rows)


def _quality_read(study: DiscoveryStudy) -> str:
    lift = study.ari - study.random_ari
    verdict = (
        "the structure is real"
        if study.ari > 0.15
        else (
            "the structure is weak but present"
            if study.ari > 0.05
            else "there is essentially no correspondence"
        )
    )
    return (
        f"Clustering the {study.n_flagged:,} flagged flows into k = {study.k} groups — a value "
        "chosen by silhouette score on the features alone, with the labels sealed — recovers an "
        f"Adjusted Rand Index of **{study.ari:.3f}** against the true attack classes, "
        f"normalized mutual information {study.nmi:.3f}, and cluster purity {study.purity:.1%}. "
        f"Random assignment into the same number of groups scores {study.random_ari:+.4f}, so "
        f"the lift is {lift:+.3f} and {verdict}. Purity and ARI disagree in the usual direction "
        "and both are reported for that reason: purity rewards splitting a family across several "
        "clusters (an analyst still sees coherent groups), while ARI penalises it (the taxonomy "
        "was not recovered exactly). Which one matters depends on whether the goal is triage or "
        "taxonomy."
    ) + _oracle_read(study)


def _oracle_read(study: DiscoveryStudy) -> str:
    """Separate 'the selector chose badly' from 'the geometry has nothing to find'."""
    lead = (
        "\n\nA negative result is only useful if it says *why*, so one diagnostic arm refits at "
        f"k = {study.oracle_k}, the true number of families. It **uses the labels** and is "
        "therefore not a result — it exists to separate two very different failures. It scores "
        f"ARI {study.oracle_ari:.3f}, purity {study.oracle_purity:.1%}, and would have surfaced "
        f"{study.oracle_discovered} families. "
    )
    if study.oracle_ari > max(study.ari, 0.0) + 0.05:
        return lead + (
            "So the geometry does carry the taxonomy and **the selector is what failed**: "
            "silhouette rewards a small number of compact, well-separated blobs, and the attack "
            "families here are neither equally sized nor spherical, so it collapses them. That "
            "is an actionable finding — the fix is a selector suited to unbalanced clusters "
            "(gap statistic, or a density-based method that does not need k at all), not a "
            "different feature space."
        )
    return lead + (
        "So the ceiling is barely above the floor, and **the selector is not the problem**: even "
        "told the right number of groups, k-means on these features does not recover the "
        "families. The taxonomy is not encoded in this feature space in a way spherical distance "
        "can see — which is a coherent finding rather than a shrug, and consistent with what the "
        "rest of this project already measures. The features are CICFlowMeter aggregate "
        "statistics chosen to separate *attack from benign*, and the "
        "[per-class](slices.md) and [novelty](novelty.md) studies both show the classes "
        "overlapping heavily in exactly this space. Attack-family discovery needs representation "
        "work (a supervised metric learned on the known families, or sequence-level features "
        "the per-flow view discards) before it needs a better clustering algorithm — which is "
        "the useful thing to learn from a null result, and the reason to run the diagnostic "
        "rather than reporting the ARI alone."
    )


def _discovery_read(study: DiscoveryStudy) -> str:
    missed = sorted(set(study.families_present) - set(study.families_discovered))
    missed_clause = (
        f" The families no cluster surfaced were {', '.join(missed)} — either too rare to form "
        "a cluster of their own at this flag budget, or behaviourally close enough to another "
        "family that the geometry merges them."
        if missed
        else " Every attack family present in the flagged set was surfaced by some cluster."
    )
    return (
        f"Of the {len(study.families_present)} attack families present among the flagged flows, "
        f"**{len(study.families_discovered)}** would have been surfaced by a cluster on their own "
        f"terms — a cluster at least {study.min_purity:.0%} made of that family and large enough "
        f"to be worth opening ({study.discovery_rate:.0%} discovery rate)." + missed_clause
    )


def _triage_read(study: DiscoveryStudy) -> str:
    novel = [s for s in study.clusters if s.novel]
    novel_clause = (
        f" {len(novel)} cluster(s) sit further than the "
        f"{study.novel_distance_threshold:.1f} distance cut from every known-class centroid and "
        "are flagged as candidate *new* families rather than named after the nearest catalogue "
        "entry — which is the distinction between recognising an attack and merely matching it "
        "to the closest thing already known."
        if novel
        else " No cluster sits far enough from the known centroids to be called a new family."
    )
    coherent = study.purity >= 0.6
    payoff = (
        f"The arithmetic looks spectacular and is not: {study.n_flagged:,} flagged flows become "
        f"{study.k} groups, a {study.triage_ratio:.0f}x reduction in tickets — but the saving is "
        "only real if judging one member of a group judges the rest, and at "
        f"{study.purity:.0%} purity it does not. An analyst who closed a whole cluster on the "
        "strength of one sample would be wrong most of the time. The triage ratio is reported "
        "because it is the number people quote, and qualified because on this data it does not "
        "survive contact with the purity column."
        if not coherent
        else f"The operational payoff is simpler than the metrics. {study.n_flagged:,} flagged "
        f"flows become {study.k} groups, a **{study.triage_ratio:.0f}x** reduction in the number "
        "of things an analyst must look at, and at "
        f"{study.purity:.0%} purity the groups are coherent enough that judging one member "
        "largely judges the rest."
    )
    return payoff + novel_clause


def _render(study: DiscoveryStudy, fig: Path) -> str:
    return f"""# NetSentry — Discovering the Attack Taxonomy Without Labels

_Synthetic stand-in. Stratified/binary split (every family appears in the test set, so
"would it have been discovered" is well posed). The {study.n_flagged:,} flows flagged at a
{study.fpr_budget:.0%} false-positive budget are clustered; **k is chosen by silhouette score
on the unlabelled features**, and the labels are opened only afterwards to grade a decision
already made._

## Why this report exists

The anomaly detector's job ends with a pile: flows that do not look like normal traffic,
handed over one at a time with no structure. That is what novel-attack detection actually
looks like — you learn *that* something is wrong long before anyone can say *what* — and it
is where analyst time goes, because a hundred flows from one campaign arrive as a hundred
tickets. Clustering is the obvious response and the obvious trap: grouping anomalies is easy,
and the usual failure is to choose the number of groups by seeing which value best reproduces
the labels, which is leakage wearing an unsupervised costume. Here the labels never vote.

## Does the geometry know about the taxonomy?

{_quality_read(study)}

![silhouette by k](../figures/{fig.name})

## Which families would have been rediscovered?

{_discovery_read(study)}

## The clusters

{_cluster_table(study)}

{_triage_read(study)}

## Scope

Clustering runs on the flows the *supervised* model flags, not on the anomaly detector's
output, so the population is "what the deployed system escalates" rather than "everything
unusual" — the [anomaly](anomaly.md) and [novelty](novelty.md) studies cover the latter, and
running this over the anomaly detector's flags is the natural extension. `k` is chosen by
silhouette, which prefers compact spherical clusters and therefore suits k-means and suits
some attack families better than others; a density-based method would find differently-shaped
groups and is the obvious alternative. Purity, ARI and NMI are computed *after* the fact and
never influence the clustering, but they are computed on labels this project treats as ground
truth, which the [label-audit](label_audit.md) study shows is itself an approximation. And
"candidate new family" is a relative judgement — a cluster far from the *known* centroids in
this feature space — not a claim that the traffic is novel in any absolute sense."""
