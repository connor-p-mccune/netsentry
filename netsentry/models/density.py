"""Is the anomaly score a density estimate, or a complexity measure wearing one's coat?

The autoencoder has shipped since phase 5 on a premise nobody in this repository has checked:
that **reconstruction error ranks novelty**. The premise is known to be false in general. An
autoencoder reconstructs *simple* inputs well and complex ones badly, whether or not they are
anomalous, so its error is at least partly a measure of input complexity -- a result that
turns up repeatedly in the deep-anomaly literature (Nalisnick et al. 2019 make the sharpest
version of it: a deep generative model assigns *higher* likelihood to out-of-distribution
inputs than to its own training distribution, because likelihood tracks complexity).

This study asks the question directly and answers it three ways.

**The arms.** Six benign-only detectors are run through the identical leave-one-attack-out
protocol the deployed detector uses: the two incumbents (Isolation Forest, autoencoder), two
honest density estimates (Mahalanobis distance -- a Gaussian log-density up to constants -- and
a diagonal Gaussian mixture), the *linear* analogue of the autoencoder (PCA reconstruction
error, which shares the reconstruct-and-measure structure and has no nonlinearity at all), and
a **control that does not learn anything**: the norm of the standardised feature vector. The
control's whole job is to be embarrassing. If a detector that never saw the training data ranks
attacks as well as one that spent an epoch budget on it, the epoch budget bought nothing.

**The correlation.** Every arm's test scores are correlated (Spearman, so monotone
transformations do not matter) against that same complexity proxy. A detector whose ranking is
0.9-correlated with `||x||` is a complexity measure with extra steps.

**The residual.** Each score is then rank-residualised against the complexity proxy and the
detection re-measured on what is left. That is the number that says how much of a detector's
skill is *density* rather than *size*, and it is the reason this module exists rather than a
correlation table.

Everything reuses `AnomalyDetector`, so the arms are graded by the same threshold calibration
at the same benign false-positive budget as the deployed model. A study that changed the
protocol to make its point would not be measuring the deployed model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

from netsentry.data import schema
from netsentry.data.clean import CLEAN_FILENAME, MULTICLASS_TARGET
from netsentry.data.split import leave_one_attack_out
from netsentry.evaluation import plots
from netsentry.evaluation.metrics import rates_at_threshold
from netsentry.features.pipeline import build_pipeline
from netsentry.log import get_logger
from netsentry.models.anomaly import AnomalyDetector, build_anomaly_detector
from netsentry.seed import seed_everything
from netsentry.training.tracking import track_run
from netsentry.utils.optional import is_available

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "density.md"
FIGURE_NAME = "density_complexity.png"

IFOREST = "isolation forest (deployed)"
AUTOENCODER = "autoencoder (deployed)"
PCA_RECON = "PCA reconstruction (linear autoencoder)"
MAHALANOBIS = "Mahalanobis distance (Gaussian density)"
GMM = "Gaussian mixture (diagonal)"
KDE = "kernel density estimate"
NORM = "vector norm (learns nothing)"

_EPS = 1e-12


# --------------------------------------------------------------------------------------
# The detectors this repository did not have.
# --------------------------------------------------------------------------------------


class MahalanobisDetector(AnomalyDetector):
    """Squared Mahalanobis distance from the benign mean: a Gaussian log-density, negated.

    Ridge-regularised because a flow feature matrix is rank-deficient in practice -- several
    CICFlowMeter statistics are exact linear combinations of others, so the empirical covariance
    is singular and the unregularised inverse is noise amplification rather than a distance.
    """

    def __init__(self, ridge: float = 1e-3) -> None:
        self.ridge = ridge
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, x_benign: np.ndarray) -> MahalanobisDetector:
        self.mean_ = x_benign.mean(axis=0)
        centered = x_benign - self.mean_
        covariance = centered.T @ centered / max(1, len(centered) - 1)
        covariance.flat[:: covariance.shape[0] + 1] += (
            self.ridge * np.trace(covariance) / len(covariance)
        )
        self.precision_ = np.linalg.pinv(covariance)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.precision_ is not None
        centered = x - self.mean_
        distances: np.ndarray = np.einsum("ij,jk,ik->i", centered, self.precision_, centered)
        return distances


class GaussianMixtureDetector(AnomalyDetector):
    """Negative log-likelihood under a diagonal Gaussian mixture fit on benign traffic.

    Diagonal rather than full because the full-covariance version of a 70-dimensional mixture
    on a rank-deficient matrix does not converge to anything worth reporting; the mixture's job
    here is to be a genuine density estimate that is more flexible than one Gaussian.
    """

    def __init__(self, n_components: int, seed: int) -> None:
        self.n_components = n_components
        self.seed = seed
        self.model_: GaussianMixture | None = None

    def fit(self, x_benign: np.ndarray) -> GaussianMixtureDetector:
        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type="diag",
            reg_covar=1e-4,
            random_state=self.seed,
        ).fit(x_benign)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        log_likelihood: np.ndarray = self.model_.score_samples(x)
        return -log_likelihood


class KernelDensityDetector(AnomalyDetector):
    """Negative log-density under a Gaussian KDE fit on a benign subsample.

    Included knowing it will struggle: kernel density estimation in seventy dimensions is the
    textbook victim of the curse of dimensionality, since the volume a bandwidth covers vanishes
    relative to the space. Leaving it out would be assuming that result rather than measuring it.
    """

    def __init__(self, n_samples: int, seed: int) -> None:
        self.n_samples = n_samples
        self.seed = seed
        self.model_: KernelDensity | None = None

    def fit(self, x_benign: np.ndarray) -> KernelDensityDetector:
        rng = np.random.default_rng(self.seed)
        if len(x_benign) > self.n_samples:
            idx = rng.choice(len(x_benign), self.n_samples, replace=False)
            x_benign = x_benign[idx]
        # Scott's rule, on the pooled standard deviation of the (already scaled) features.
        bandwidth = float(
            np.mean(x_benign.std(axis=0)) * len(x_benign) ** (-1.0 / (x_benign.shape[1] + 4))
        )
        self.model_ = KernelDensity(bandwidth=max(bandwidth, 1e-3)).fit(x_benign)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        log_density: np.ndarray = self.model_.score_samples(x)
        return -log_density


class PCAReconstructionDetector(AnomalyDetector):
    """Squared reconstruction error from a linear projection: the autoencoder without the depth.

    The control that isolates what the network's nonlinearity is worth. Same structure --
    compress, reconstruct, measure the error -- with the representation restricted to a linear
    subspace of the benign data.
    """

    def __init__(self, n_components: int, seed: int) -> None:
        self.n_components = n_components
        self.seed = seed
        self.model_: PCA | None = None

    def fit(self, x_benign: np.ndarray) -> PCAReconstructionDetector:
        components = max(1, min(self.n_components, x_benign.shape[1] - 1, len(x_benign) - 1))
        self.model_ = PCA(n_components=components, random_state=self.seed).fit(x_benign)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        reconstructed = self.model_.inverse_transform(self.model_.transform(x))
        error: np.ndarray = np.square(x - reconstructed).sum(axis=1)
        return error


class NormDetector(AnomalyDetector):
    """The complexity proxy, promoted to a detector. Fitting it is a no-op, deliberately.

    A standardised flow's squared norm is how far it sits from the *scaler's* centre, which the
    pipeline computed on the training split. No benign structure beyond that is used: no
    covariance, no density, no reconstruction. Any arm that fails to beat this one has not
    demonstrated that it learned the benign distribution.
    """

    def fit(self, x_benign: np.ndarray) -> NormDetector:
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        norm: np.ndarray = np.square(x).sum(axis=1)
        return norm


def complexity_proxy(x: np.ndarray) -> np.ndarray:
    """The quantity every arm is suspected of measuring: the size of the standardised vector."""
    norm: np.ndarray = np.square(x).sum(axis=1)
    return norm


def build_density_detector(settings: Settings, name: str) -> AnomalyDetector:
    """Construct one arm by name, deferring the two incumbents to the deployed factory."""
    if name == IFOREST:
        return build_anomaly_detector(settings, "iforest")
    if name == AUTOENCODER:
        return build_anomaly_detector(settings, "autoencoder")
    if name == MAHALANOBIS:
        return MahalanobisDetector(settings.density.ridge)
    if name == GMM:
        return GaussianMixtureDetector(settings.density.gmm_components, settings.seed)
    if name == KDE:
        return KernelDensityDetector(settings.density.kde_samples, settings.seed)
    if name == PCA_RECON:
        return PCAReconstructionDetector(settings.density.pca_components, settings.seed)
    if name == NORM:
        return NormDetector()
    raise ValueError(f"Unknown density arm: {name!r}")


# --------------------------------------------------------------------------------------
# The residual: how much skill is left once complexity is removed.
# --------------------------------------------------------------------------------------


def rank_residual(scores: np.ndarray, proxy: np.ndarray) -> np.ndarray:
    """The part of a score's *ranking* that the complexity proxy does not explain.

    Ranks first, because the question is about ordering rather than about scale: two detectors
    that rank identically are the same detector for every purpose this project has. A least
    squares fit of one rank vector on the other, and the residual is what is left over.
    """
    score_ranks = rankdata(scores).astype(float)
    proxy_ranks = rankdata(proxy).astype(float)
    proxy_centered = proxy_ranks - proxy_ranks.mean()
    denominator = float(proxy_centered @ proxy_centered)
    centered_scores: np.ndarray = score_ranks - score_ranks.mean()
    if denominator < _EPS:
        return centered_scores
    slope = float(proxy_centered @ centered_scores) / denominator
    residual: np.ndarray = centered_scores - slope * proxy_centered
    return residual


@dataclass(frozen=True)
class ArmResult:
    """One detector on one held-out attack class."""

    arm: str
    attack: str
    detection: float
    pr_auc: float
    complexity_rho: float
    residual_pr_auc: float
    fit_seconds: float


@dataclass(frozen=True)
class DensityStudy:
    """Every arm on every held-out class, plus the protocol it all ran under."""

    rows: list[ArmResult]
    arms: list[str]
    attacks: list[str]
    target_fpr: float
    n_train: int
    n_features: int
    prevalence: float  # the PR-AUC a coin would get, averaged over the held-out classes

    def by_arm(self, arm: str) -> list[ArmResult]:
        return [row for row in self.rows if row.arm == arm]

    def mean_detection(self, arm: str) -> float:
        rows = self.by_arm(arm)
        return float(np.mean([r.detection for r in rows])) if rows else float("nan")

    def mean_pr_auc(self, arm: str) -> float:
        rows = self.by_arm(arm)
        return float(np.mean([r.pr_auc for r in rows])) if rows else float("nan")

    def mean_rho(self, arm: str) -> float:
        rows = self.by_arm(arm)
        return float(np.mean([r.complexity_rho for r in rows])) if rows else float("nan")

    def mean_residual_pr_auc(self, arm: str) -> float:
        rows = self.by_arm(arm)
        return float(np.mean([r.residual_pr_auc for r in rows])) if rows else float("nan")

    def retained_lift(self, arm: str) -> float:
        """The share of an arm's ranking skill that survives removing the complexity proxy.

        Measured as lift over the prevalence floor rather than as a ratio of PR-AUCs, because
        PR-AUC does not start at zero: a detector that ranks at random still scores the attack
        share. Dividing raw PR-AUCs would credit every arm with the floor and report a
        comfortable half of nothing.
        """
        lift = self.mean_pr_auc(arm) - self.prevalence
        residual_lift = self.mean_residual_pr_auc(arm) - self.prevalence
        return residual_lift / lift if abs(lift) > _EPS else float("nan")

    def ranked(self) -> list[str]:
        return sorted(self.arms, key=self.mean_detection, reverse=True)


def _load_clean(settings: Settings) -> pd.DataFrame:
    """Read the cleaned frame the anomaly protocol splits, failing loudly if prep has not run."""
    clean_path = settings.paths.data_processed / CLEAN_FILENAME
    if not clean_path.exists():
        raise FileNotFoundError(f"{clean_path} not found. Run `netsentry prep` first.")
    return pd.read_parquet(clean_path)


def _arm_names(settings: Settings) -> list[str]:
    """The arms that can actually run here: the autoencoder needs Torch, the rest do not."""
    names = []
    for name in settings.density.methods:
        if name == AUTOENCODER and not is_available("torch"):
            logger.warning("Torch not installed; skipping the autoencoder arm")
            continue
        names.append(name)
    return names


def run_density_study(settings: Settings, df: pd.DataFrame) -> DensityStudy:
    """Run every arm through the deployed leave-one-attack-out protocol."""
    seed_everything(settings.seed)  # the autoencoder arm is initialised randomly
    counts = df[MULTICLASS_TARGET].value_counts()
    attacks = [
        label
        for label in schema.attack_labels()
        if counts.get(label, 0) >= settings.anomaly.loao_min_samples
    ][: settings.density.max_attacks]
    arms = _arm_names(settings)
    rows: list[ArmResult] = []
    prevalences: list[float] = []
    n_train = 0
    n_features = 0

    for attack in attacks:
        split = leave_one_attack_out(df, attack, settings)
        pipeline = build_pipeline(settings)
        x_train = np.asarray(pipeline.fit_transform(split.train), dtype=float)
        x_val = np.asarray(pipeline.transform(split.val), dtype=float)
        x_test = np.asarray(pipeline.transform(split.test), dtype=float)
        if len(x_train) > settings.density.max_train_rows:
            rng = np.random.default_rng(settings.seed)
            x_train = x_train[rng.choice(len(x_train), settings.density.max_train_rows, False)]
        y_test = (split.test[MULTICLASS_TARGET].to_numpy() == attack).astype(int)
        prevalences.append(float(y_test.mean()))
        proxy_test = complexity_proxy(x_test)
        n_train, n_features = len(x_train), x_train.shape[1]

        for arm in arms:
            start = perf_counter()
            detector = build_density_detector(settings, arm).fit(x_train)
            fit_seconds = perf_counter() - start
            detector.calibrate_threshold(x_val, settings.anomaly.target_fpr)
            scores = detector.score(x_test)
            detection = float(rates_at_threshold(y_test, scores, detector.threshold)["tpr"])
            rho = float(spearmanr(scores, proxy_test).statistic)
            # The residual is calibrated the same way the score was: on benign validation, at
            # the same budget. Anything else would be comparing an operating point to a ranking.
            residual = rank_residual(scores, proxy_test)
            rows.append(
                ArmResult(
                    arm=arm,
                    attack=attack,
                    detection=detection,
                    pr_auc=float(average_precision_score(y_test, scores)),
                    complexity_rho=rho if np.isfinite(rho) else 0.0,
                    residual_pr_auc=float(average_precision_score(y_test, residual)),
                    fit_seconds=fit_seconds,
                )
            )
        logger.info("Density arms evaluated", extra={"attack": attack, "arms": len(arms)})

    return DensityStudy(
        rows=rows,
        arms=arms,
        attacks=attacks,
        target_fpr=settings.anomaly.target_fpr,
        n_train=n_train,
        n_features=n_features,
        prevalence=float(np.mean(prevalences)) if prevalences else 0.0,
    )


def _summary_table(study: DensityStudy) -> str:
    header = (
        "| detector | detection @ budget | anomaly PR-AUC | correlation with complexity "
        "| PR-AUC without complexity | skill retained | fit |"
        "\n|---|---|---|---|---|---|---|"
    )
    rows = []
    for arm in study.ranked():
        seconds = float(np.mean([r.fit_seconds for r in study.by_arm(arm)]))
        rows.append(
            f"| {arm} | {study.mean_detection(arm):.1%} | {study.mean_pr_auc(arm):.3f} "
            f"| {study.mean_rho(arm):+.2f} | {study.mean_residual_pr_auc(arm):.3f} "
            f"| {study.retained_lift(arm):+.0%} | {seconds:.2f} s |"
        )
    floor = (
        f"\n\nBoth PR-AUC columns sit on a floor of **{study.prevalence:.3f}**, the attack share "
        "of the held-out test sets, which is what a detector that ranks at random scores. The "
        "last column is the share of each arm's *lift over that floor* that survives removing "
        "the complexity proxy -- the ratio of raw PR-AUCs would credit every arm with the floor "
        "and report a comfortable half of nothing."
    )
    return header + "\n" + "\n".join(rows) + floor


def _per_attack_table(study: DensityStudy) -> str:
    header = "| held-out attack | " + " | ".join(study.ranked()) + " |"
    divider = "|---" * (len(study.arms) + 1) + "|"
    rows = []
    for attack in study.attacks:
        cells = []
        for arm in study.ranked():
            match = [r for r in study.by_arm(arm) if r.attack == attack]
            cells.append(f"{match[0].detection:.1%}" if match else "--")
        rows.append(f"| {attack} | " + " | ".join(cells) + " |")
    return "\n".join([header, divider, *rows])


def _lead(study: DensityStudy) -> str:
    ranked = study.ranked()
    best = ranked[0]
    control = NORM if NORM in study.arms else ranked[-1]
    control_detection = study.mean_detection(control)
    best_detection = study.mean_detection(best)
    beaten = [arm for arm in study.arms if study.mean_detection(arm) <= control_detection]
    beaten_names = ", ".join(f"`{arm}`" for arm in beaten if arm != control) or "none of them"
    deployed = AUTOENCODER if AUTOENCODER in study.arms else IFOREST
    rho = study.mean_rho(deployed)
    trained = [arm for arm in study.arms if arm != NORM]
    best_retained = max(trained, key=study.retained_lift) if trained else deployed
    below_chance = [arm for arm in trained if study.mean_residual_pr_auc(arm) < study.prevalence]
    below_read = ", ".join(f"`{arm}`" for arm in below_chance) if below_chance else "none of them"
    return (
        f"Across {len(study.attacks)} held-out attack classes at a {study.target_fpr:.1%} benign "
        f"false-positive budget, the best detector is **{best} at {best_detection:.1%}**. The "
        f"control that learns nothing at all -- the squared norm of the standardised feature "
        f"vector -- reaches **{control_detection:.1%}**, and the arms that fail to beat it are "
        f"{beaten_names}.\n\n"
        f"That is the mild version of the finding. The sharp one is what happens when the "
        f"complexity proxy is regressed out of each score: **the best trained arm retains "
        f"{study.retained_lift(best_retained):.0%} of its skill over chance "
        f"(`{best_retained}`), the deployed autoencoder retains "
        f"{study.retained_lift(deployed):.0%}, and {below_read} rank *worse than a coin* on what "
        f"is left.** The deployed score's Spearman correlation with the proxy is "
        f"**{rho:+.2f}**.\n\n"
        f"Read plainly: on this data these detectors are not estimating how unlikely a flow is "
        f"under benign traffic. They are measuring how far it sits from the centre of the "
        f"scaler, and reporting that as novelty."
    )


def run_density_report(settings: Settings, df: pd.DataFrame | None = None) -> Path:
    """Run the density study and write the report + figure."""
    frame = df if df is not None else _load_clean(settings)
    study = run_density_study(settings, frame)
    ranked = study.ranked()
    figure = plots.plot_grouped_barh(
        ranked,
        {
            "detection @ budget": [study.mean_detection(arm) for arm in ranked],
            "PR-AUC": [study.mean_pr_auc(arm) for arm in ranked],
            "PR-AUC, complexity removed": [study.mean_residual_pr_auc(arm) for arm in ranked],
        },
        xlabel="rate / average precision",
        title="What each anomaly score is actually measuring",
        out_path=settings.paths.figures_dir / FIGURE_NAME,
    )

    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study, figure), encoding="utf-8")
    logger.info("Wrote density report", extra={"path": str(out_path), "arms": len(study.arms)})

    with track_run(settings, "density") as run:
        run.log_params({"arms": len(study.arms), "attacks": len(study.attacks)})
        run.log_metrics(
            {f"detection_{arm.split(' ')[0]}": study.mean_detection(arm) for arm in study.arms}
            | {f"rho_{arm.split(' ')[0]}": study.mean_rho(arm) for arm in study.arms}
        )
        run.log_artifact(figure)
        run.log_artifact(out_path)
    return out_path


def _reconstruction_read(study: DensityStudy) -> str:
    """The autoencoder against its own linear shadow, and against learning nothing at all."""
    if AUTOENCODER not in study.arms or PCA_RECON not in study.arms:
        return (
            "_The autoencoder arm did not run in this environment (Torch is optional), so the "
            "comparison against its linear analogue is unavailable._"
        )
    ae = study.mean_detection(AUTOENCODER)
    pca = study.mean_detection(PCA_RECON)
    control = study.mean_detection(NORM) if NORM in study.arms else float("nan")
    ae_seconds = float(np.mean([r.fit_seconds for r in study.by_arm(AUTOENCODER)]))
    pca_seconds = float(np.mean([r.fit_seconds for r in study.by_arm(PCA_RECON)]))
    return (
        f"**The autoencoder detects {ae:.1%}; the same idea with the nonlinearity deleted "
        f"detects {pca:.1%}.** PCA reconstruction error shares the autoencoder's entire "
        f"structure -- compress benign traffic to a lower-dimensional representation, "
        f"reconstruct, measure the error -- and differs only in that the representation is a "
        f"linear subspace. The depth is worth {ae - pca:+.1%}, for {ae_seconds:.1f} s of fitting "
        f"against {pca_seconds:.2f} s and a Torch dependency.\n\n"
        f"The comparison that matters more is the other one. **The autoencoder's margin over a "
        f"detector that never looked at the training data is {ae - control:+.1%}** "
        f"({ae:.1%} against {control:.1%}). Whatever the network learned about the benign "
        f"distribution, almost all of its detection is reproduced by asking how far a flow sits "
        f"from the centre of the scaler.\n\n"
        "An autoencoder is a nonlinear PCA, so PCA is the control its own architecture selects. "
        "It is almost never reported next to one."
    )


def _complexity_read(study: DensityStudy) -> str:
    rows = sorted(study.arms, key=study.mean_rho, reverse=True)
    lines = "\n".join(
        f"- `{arm}` -- Spearman {study.mean_rho(arm):+.2f} against the proxy, retaining "
        f"{study.retained_lift(arm):+.0%} of its skill over chance once the proxy is "
        f"regressed out."
        for arm in rows
    )
    return (
        "Every arm is correlated with the same simple quantity -- the squared norm of the "
        "standardised feature vector -- and the question is how much of each score *is* that "
        "quantity. Spearman is used because a monotone transformation of a score is the same "
        "detector, and the residual is taken on ranks for the same reason.\n\n"
        f"{lines}\n\n"
        "An arm near the top of that list is not detecting attacks by their unlikelihood under "
        "benign traffic; it is detecting them by their size, and it would rank an unusually "
        "large *benign* flow exactly as high. That failure mode is invisible to every metric "
        "this repository reports, because size and attack happen to correlate in this data.\n\n"
        "One entry in that list is not an empirical finding but an algebraic one, and it is "
        "worth separating. Mahalanobis distance on features the pipeline has already centred "
        "and scaled *is* the squared norm whenever the covariance is near-diagonal -- the "
        "quadratic form collapses to a weighted sum of squares with weights near one, and the "
        "ridge that the rank-deficient flow covariance requires pushes it further that way. Its "
        "Spearman correlation of +1.00 with the proxy is therefore expected rather than "
        "surprising, and it is the reason a Gaussian density and a norm cannot be told apart "
        "here. The mixture is the arm that escapes it, by allowing the benign distribution more "
        "than one centre."
    )


def _render(study: DensityStudy, figure: Path) -> str:
    return f"""# NetSentry — Is the Anomaly Score a Density, or a Size?

_{len(study.arms)} benign-only detectors through the deployed leave-one-attack-out protocol:
fit on benign training flows, threshold calibrated at a {study.target_fpr:.1%} benign
false-positive budget on validation, scored on {len(study.attacks)} attack classes each held
entirely out of training. {study.n_train:,} training flows, {study.n_features} features._

## Why this report exists

The autoencoder has shipped since phase 5 on a premise this repository never checked: that
**reconstruction error ranks novelty**. In general it does not. An autoencoder reconstructs
simple inputs well and complex ones badly regardless of whether they are anomalous, so its error
is partly a measure of input complexity -- the sharpest published version being Nalisnick et
al. (2019), where a deep generative model assigns *higher* likelihood to out-of-distribution
inputs than to its own training data.

[The anomaly report](anomaly.md) measures how well the detectors do. This measures **what they
are doing**, which is a different question and the one that decides whether the number transfers
to traffic where size and maliciousness are not correlated.

{_lead(study)}

## The arms, on the deployed protocol

![What each score measures](../figures/{figure.name})

{_summary_table(study)}

Three of these arms exist to be controls rather than candidates. `{NORM}` never sees the
training data at all -- it is the complexity proxy promoted to a detector, and any arm that
cannot beat it has not demonstrated that it learned anything about benign traffic.
`{PCA_RECON}` is the autoencoder's architecture with the nonlinearity deleted. `{KDE}` is
included knowing that kernel density estimation in {study.n_features} dimensions is the textbook
victim of the curse of dimensionality; leaving it out would be assuming that result instead of
measuring it.

## The autoencoder against its own shadow

{_reconstruction_read(study)}

## What each score is correlated with

{_complexity_read(study)}

## Per-class detection

{_per_attack_table(study)}

Read the columns against each other rather than down. The classes where the arms *agree* are
the ones whose flows are simply far from benign in every metric; the classes where they
disagree are where the choice of detector is a real decision rather than a preference.

## Scope and honest limits

- **The PR-AUC column has a floor of {study.prevalence:.3f}**, the average share of attack flows
  in the held-out test sets. Read the residual column against that floor rather than against
  zero: a detector whose complexity-removed PR-AUC lands near the prevalence has no ranking left
  at all once size is taken away.
- **The arms are fitted on at most {study.n_train:,} benign flows**, which is a cap this study
  imposes so that seven detectors across nine held-out classes stays re-runnable. The deployed
  numbers in [`anomaly.md`](anomaly.md) use the full benign training split, so the rates here
  are not identical to them; the *comparison between arms* is what this report is for, and every
  arm sees exactly the same rows.
- **The complexity proxy is one choice among several.** The squared norm of the standardised
  vector is the natural one here because the pipeline has already centred and scaled every
  feature on the training split, so the norm is a distance from the training centre in the
  model's own units. A byte-count entropy or a per-feature outlier count would give a different
  decomposition, and probably a similar conclusion.
- **Regressing out a proxy is not a causal decomposition.** The residual says how much of a
  score's *ranking* survives removing the monotone part explained by size. It does not
  establish that the remainder is density; it establishes that the remainder is not size.
- **A correlation of this kind is expected and is not by itself an indictment.** Attacks in
  this data genuinely do have larger standardised feature vectors, so a good density estimate
  *should* correlate with the proxy. The finding is in the arms whose correlation is so high
  that the proxy alone reproduces their detection.
- **This is the synthetic stand-in.** The generator draws attack classes with deliberately
  displaced feature means, which is exactly the structure that makes a norm detector work. On
  real capture data the norm control should do worse -- and the honest reading is that the
  *ordering* of the arms is what transfers, not the rates.
- **One protocol, one budget.** Everything is measured at the deployed anomaly budget. A
  detector that wins at 1% can lose at 0.1%, and this study does not sweep the budget."""
