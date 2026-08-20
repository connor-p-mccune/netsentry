"""A specification of what the serving lifecycle must do, and a machine that checks it.

The serving layer has grown a lifecycle: load a bundle, replay its canaries, serve, authenticate,
rate-limit, hot-reload behind a canary gate, refuse a reload that fails it. Each of those parts
has a test. What none of those tests cover is the part that actually breaks in production --
the *sequences*. A reload that half-succeeds, an authentication check that stops applying after
a swap, a health endpoint that keeps reporting the version it used to serve: every one of those
is a two-step bug, and a suite of single-step tests cannot see any of them.

This module states the lifecycle as an explicit model -- a small state machine with the
transitions the API is allowed to make -- and then drives the *real* application through random
sequences of operations, checking after every step that the observed response and the model
agree. It is the same idea as a conformance test for a protocol implementation, and it finds
the class of bug that unit tests structurally cannot.

Three properties are worth naming, because they are the ones a single-request test cannot state:

1. **A refused reload changes nothing.** Not the version, not the health, not the next
   prediction. A 409 that leaves a half-swapped engine is indistinguishable from a 200 until
   the next request, which is exactly when it matters.
2. **The version a prediction reports is the version the model was in when it started.** The
   swap is a single reference reassignment for this reason, and the machine checks that no
   response ever names a version the app was never in.
3. **The guard applies to the guarded routes and to nothing else**, in every state. An
   authentication check that a reload silently drops is a hole nobody would find by reading.

As with the [static-analysis rules](mlint.md), a checker nobody has watched fail is a checker
nobody should trust, so this module carries mutants: deliberately broken versions of the
service, injected at the HTTP boundary, that the machine has to catch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings

logger = get_logger(__name__)

REPORT_NAME = "state_machine.md"

# Operation names. Deliberately close to what an operator or an attacker actually does.
PREDICT = "predict"
PREDICT_BATCH = "predict batch"
PREDICT_NO_KEY = "predict without the API key"
HEALTH = "health"
METRICS = "metrics"
RELOAD_VALID = "reload a good bundle"
RELOAD_CANARY_FAIL = "reload a bundle whose canaries do not reproduce"
RELOAD_MISSING = "reload a bundle that is not there"
RELOAD_ESCAPE = "reload a path outside the models dir"
MALFORMED = "predict with a malformed flow"

#: The operations that construct a whole inference engine, and so cost seconds rather than
#: milliseconds. They are drawn less often for that reason alone.
HEAVY_OPERATIONS: frozenset[str] = frozenset(
    {"reload a good bundle", "reload a bundle whose canaries do not reproduce"}
)

OPERATIONS: tuple[str, ...] = (
    PREDICT,
    PREDICT_BATCH,
    PREDICT_NO_KEY,
    HEALTH,
    METRICS,
    RELOAD_VALID,
    RELOAD_CANARY_FAIL,
    RELOAD_MISSING,
    RELOAD_ESCAPE,
    MALFORMED,
)


@dataclass(frozen=True)
class LifecycleState:
    """What the service should currently be, as far as an observer can tell.

    Deliberately tiny. A model that mirrors the implementation cannot disagree with it, and a
    model that cannot disagree cannot find anything: this holds only what the API contract
    promises an observer.
    """

    version: str
    canary_ok: bool
    api_key: str | None
    seen_versions: frozenset[str] = field(default_factory=frozenset)

    def after_reload(self, version: str) -> LifecycleState:
        return replace(self, version=version, seen_versions=self.seen_versions | {version})


@dataclass(frozen=True)
class Expectation:
    """What the model says an operation should do."""

    status: tuple[int, ...]
    version_changes: bool = False
    describes_version: bool = False


def expected(state: LifecycleState, operation: str, candidate_version: str | None) -> Expectation:
    """The contract, as a function. This is the specification the machine tests against."""
    if operation in {PREDICT, PREDICT_BATCH}:
        return Expectation(status=(200,), describes_version=True)
    if operation == PREDICT_NO_KEY:
        # With no key configured the guard is inert and the request is ordinary.
        return Expectation(status=(401,) if state.api_key else (200,))
    if operation == HEALTH:
        return Expectation(status=(200,), describes_version=True)
    if operation == METRICS:
        return Expectation(status=(200,))
    if operation == RELOAD_VALID:
        changes = candidate_version is not None and candidate_version != state.version
        return Expectation(status=(200,), version_changes=changes, describes_version=True)
    if operation == RELOAD_CANARY_FAIL:
        # The gate the whole hot-reload design exists for: refuse, and keep serving.
        return Expectation(status=(409, 422))
    if operation == RELOAD_MISSING:
        return Expectation(status=(404,))
    if operation == RELOAD_ESCAPE:
        return Expectation(status=(400,))
    if operation == MALFORMED:
        return Expectation(status=(422,))
    raise ValueError(f"unknown operation: {operation!r}")


@dataclass(frozen=True)
class Step:
    """One operation, what came back, and every way it disagreed with the model."""

    index: int
    operation: str
    status: int
    version_reported: str | None
    violations: tuple[str, ...]


def check_step(
    state: LifecycleState,
    operation: str,
    status: int,
    body: dict[str, Any] | None,
    candidate_version: str | None,
) -> tuple[LifecycleState, tuple[str, ...]]:
    """Compare one observed response against the model; return the next state and violations.

    Pure, so the same function grades a Hypothesis-driven run, a random exploration and a
    mutant. Anything that decided what to do next by looking at the response would be testing
    the implementation against itself.
    """
    spec = expected(state, operation, candidate_version)
    violations: list[str] = []
    if status not in spec.status:
        violations.append(
            f"{operation}: expected HTTP {' or '.join(map(str, spec.status))}, got {status}"
        )

    reported = None
    if isinstance(body, dict):
        reported = body.get("model_version")

    next_state = state
    if operation == RELOAD_VALID and status == 200 and candidate_version is not None:
        next_state = state.after_reload(candidate_version)
    elif (
        spec.version_changes is False
        and reported is not None
        and status < 400
        and reported != state.version
    ):
        # Property 1: nothing except a successful reload may change the served version.
        violations.append(
            f"{operation}: served version {reported!r} but the model is on {state.version!r}"
        )

    if operation in {RELOAD_CANARY_FAIL, RELOAD_MISSING, RELOAD_ESCAPE} and status < 400:
        violations.append(f"{operation}: a refused reload returned a success status ({status})")

    if operation == HEALTH and isinstance(body, dict):
        # Property: health never claims ok while the canary it replayed did not reproduce.
        canary = body.get("canary")
        if isinstance(canary, dict) and canary.get("ok") is False and body.get("status") == "ok":
            violations.append("health: reported ok while its canary was failing")

    return next_state, tuple(violations)


#: One observed operation: what was asked, and what the service said.
Observation = tuple[str, int, "dict[str, Any] | None", "str | None"]


@dataclass(frozen=True)
class Exploration:
    """One random walk over the operation space, and everything it saw."""

    steps: list[Step]
    n_operations: int
    label: str
    transcript: list[Observation] = field(default_factory=list)

    @property
    def violations(self) -> list[Step]:
        return [step for step in self.steps if step.violations]

    @property
    def clean(self) -> bool:
        return not self.violations

    def coverage(self) -> dict[str, int]:
        counts = {operation: 0 for operation in OPERATIONS}
        for step in self.steps:
            counts[step.operation] = counts.get(step.operation, 0) + 1
        return counts


def explore(
    execute: Callable[[str], tuple[int, dict[str, Any] | None, str | None]],
    initial: LifecycleState,
    steps: int,
    rng: np.random.Generator,
    operations: Iterable[str] = OPERATIONS,
    label: str = "the real service",
    plan: list[str] | None = None,
) -> Exploration:
    """Drive ``execute`` through a random sequence of operations, grading every response.

    ``execute`` performs one operation and returns ``(status, body, candidate_version)``. The
    indirection is what lets the same walk run against the real application, against a mutant,
    and against a stub in a unit test.
    """
    transcript: list[Observation] = []
    for operation in plan or list(rng.choice(list(operations), size=steps)):
        status, body, candidate_version = execute(str(operation))
        transcript.append((str(operation), status, body, candidate_version))
    return grade(transcript, initial, label)


def plan_operations(
    steps: int,
    rng: np.random.Generator,
    heavy: frozenset[str] = HEAVY_OPERATIONS,
    min_heavy: int = 4,
    min_light: int = 8,
    operations: Iterable[str] = OPERATIONS,
) -> list[str]:
    """Build the operation schedule before running it, so coverage is guaranteed rather than hoped.

    A weighted random draw is the obvious implementation and it produced a headline run in which
    the single most important transition -- a *successful* reload -- was never drawn at all. The
    cost structure forces the weighting (two operations build an entire inference engine and take
    seconds; the rest take milliseconds), so the fix is to allocate the expensive ones explicitly,
    fill the remainder with the cheap ones, and shuffle. Order stays random; coverage stops being
    a matter of luck.
    """
    names = list(operations)
    schedule: list[str] = []
    for name in names:
        schedule.extend([name] * (min_heavy if name in heavy else min_light))
    light = [name for name in names if name not in heavy]
    while len(schedule) < steps and light:
        schedule.append(str(rng.choice(light)))
    schedule = schedule[: max(steps, len(schedule))]
    rng.shuffle(schedule)
    return schedule


def grade(transcript: list[Observation], initial: LifecycleState, label: str) -> Exploration:
    """Walk the model along a recorded transcript, collecting every disagreement.

    Separating the walk from the grading is what lets a mutant re-run the *identical* sequence:
    the regressions are injected at the HTTP boundary, so replaying the transcript through a
    rewrite is exactly equivalent to driving a service that had that regression -- and it costs
    no reloads, each of which builds a whole inference engine.
    """
    state = initial
    records: list[Step] = []
    for index, (operation, status, body, candidate_version) in enumerate(transcript):
        state, violations = check_step(state, operation, status, body, candidate_version)
        records.append(
            Step(
                index=index,
                operation=operation,
                status=status,
                version_reported=(body or {}).get("model_version") if body else None,
                violations=violations,
            )
        )
    return Exploration(
        steps=records, n_operations=len(transcript), label=label, transcript=transcript
    )


def apply_mutation(
    transcript: list[Observation],
    rewrite: Callable[[str, int, dict[str, Any] | None], tuple[int, dict[str, Any] | None]],
) -> list[Observation]:
    """Rewrite a recorded transcript as the mutated service would have answered."""
    mutated: list[Observation] = []
    for operation, status, body, candidate_version in transcript:
        new_status, new_body = rewrite(operation, status, body)
        mutated.append((operation, new_status, new_body, candidate_version))
    return mutated


# --------------------------------------------------------------------------------------
# Driving the real service.
# --------------------------------------------------------------------------------------

SAMPLE_FLOW: dict[str, float] = {
    "Flow Duration": 1200.0,
    "Total Fwd Packets": 8.0,
    "Flow Packets/s": 50.0,
}

CANARY_FAIL_BUNDLE = "_lifecycle_canary_fail.joblib"


def write_canary_failing_bundle(settings: Settings, source: Path, destination: Path) -> Path:
    """Copy a bundle and perturb its embedded expectations so its canary cannot reproduce.

    This is the artifact the reload gate exists to refuse. Corrupting the *expected scores*
    rather than the model is deliberate: it produces exactly the symptom a genuine
    environment mismatch produces -- the runtime scores the embedded flows differently from
    whatever produced them -- without needing a second training run to manufacture one.
    """
    import joblib

    from netsentry.serving.canary import CANARY_KEY

    bundle = joblib.load(source)
    payload = dict(bundle.metadata.get(CANARY_KEY) or {})
    if not payload.get("expected_scores"):
        raise ValueError(f"{source} carries no canary to corrupt")
    payload["expected_scores"] = [float(score) + 0.5 for score in payload["expected_scores"]]
    bundle.metadata[CANARY_KEY] = payload
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, destination)
    return destination


def make_executor(
    client: Any, settings: Settings, valid_bundle: str, failing_bundle: str
) -> Callable[[str], tuple[int, dict[str, Any] | None, str | None]]:
    """Bind the operation vocabulary to real HTTP calls against ``client``."""
    key = settings.serving.api_key
    headers = {"x-api-key": key} if key else {}

    def _json(response: Any) -> dict[str, Any] | None:
        try:
            body = response.json()
        except Exception:  # a metrics response is text; that is not a failure
            return None
        return body if isinstance(body, dict) else None

    def execute(operation: str) -> tuple[int, dict[str, Any] | None, str | None]:
        if operation == PREDICT:
            response = client.post(
                "/predict", json={"flow": SAMPLE_FLOW}, params={"explain": "false"}, headers=headers
            )
        elif operation == PREDICT_BATCH:
            response = client.post(
                "/predict/batch",
                json={"flows": [SAMPLE_FLOW, SAMPLE_FLOW]},
                params={"explain": "false"},
                headers=headers,
            )
            body = _json(response)
            version = None
            if body and isinstance(body.get("predictions"), list) and body["predictions"]:
                version = body["predictions"][0].get("model_version")
            return response.status_code, ({"model_version": version} if version else None), None
        elif operation == PREDICT_NO_KEY:
            response = client.post(
                "/predict", json={"flow": SAMPLE_FLOW}, params={"explain": "false"}
            )
        elif operation == HEALTH:
            response = client.get("/health")
        elif operation == METRICS:
            response = client.get("/metrics")
        elif operation == RELOAD_VALID:
            response = client.post("/admin/reload", json={"bundle": valid_bundle}, headers=headers)
            body = _json(response)
            return response.status_code, body, (body or {}).get("model_version")
        elif operation == RELOAD_CANARY_FAIL:
            response = client.post(
                "/admin/reload", json={"bundle": failing_bundle}, headers=headers
            )
        elif operation == RELOAD_MISSING:
            response = client.post(
                "/admin/reload", json={"bundle": "no_such_bundle.joblib"}, headers=headers
            )
        elif operation == RELOAD_ESCAPE:
            response = client.post(
                "/admin/reload", json={"bundle": "../../etc/passwd"}, headers=headers
            )
        elif operation == MALFORMED:
            response = client.post(
                "/predict", json={"flow": {"Flow Duration": "not a number"}}, headers=headers
            )
        else:
            raise ValueError(f"unknown operation: {operation!r}")
        return response.status_code, _json(response), None

    return execute


# --------------------------------------------------------------------------------------
# The mutants: broken services the machine has to catch.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Mutant:
    """A regression injected at the HTTP boundary, and what it is meant to imitate."""

    name: str
    description: str
    rewrite: Callable[[str, int, dict[str, Any] | None], tuple[int, dict[str, Any] | None]]


def _swap_before_gate(
    operation: str, status: int, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any] | None]:
    """A reload that swaps the engine and *then* checks the canary."""
    if operation == RELOAD_CANARY_FAIL:
        return 200, {"model_version": "corrupted-candidate"}
    return status, body


def _stale_version(
    operation: str, status: int, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any] | None]:
    """A swap that updates the engine but leaves /health reporting the old version."""
    if operation == HEALTH and body is not None:
        return status, {**body, "model_version": "0.0.0-stale"}
    return status, body


def _guard_dropped(
    operation: str, status: int, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any] | None]:
    """Authentication that stops applying: the unkeyed request is served anyway."""
    if operation == PREDICT_NO_KEY and status == 401:
        return 200, {"model_version": "whatever"}
    return status, body


def _refusal_is_silent(
    operation: str, status: int, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any] | None]:
    """A missing bundle reported as success -- the failure mode of a swallowed exception."""
    if operation in {RELOAD_MISSING, RELOAD_ESCAPE}:
        return 200, body
    return status, body


def _health_lies(
    operation: str, status: int, body: dict[str, Any] | None
) -> tuple[int, dict[str, Any] | None]:
    """A health endpoint that reports ok while its own canary says otherwise."""
    if operation == HEALTH and body is not None:
        return status, {**body, "status": "ok", "canary": {"ok": False, "n": 8}}
    return status, body


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        "swap before the canary gate",
        "the candidate is installed and *then* validated, so a bundle this runtime cannot "
        "reproduce ends up serving traffic",
        _swap_before_gate,
    ),
    Mutant(
        "stale version after a swap",
        "the engine is replaced but /health keeps reporting the version it used to serve, so "
        "an operator cannot tell which model answered",
        _stale_version,
    ),
    Mutant(
        "the guard stops applying",
        "authentication passes on a route that must require it -- the hole a refactor leaves "
        "and no single-request test notices",
        _guard_dropped,
    ),
    Mutant(
        "a refusal reported as success",
        "a missing or out-of-tree bundle returns 200, which is what a swallowed exception "
        "looks like from outside",
        _refusal_is_silent,
    ),
    Mutant(
        "health that lies about its canary",
        "status ok while the embedded canary did not reproduce: the readiness probe every "
        "orchestrator trusts",
        _health_lies,
    ),
)


def mutate(
    execute: Callable[[str], tuple[int, dict[str, Any] | None, str | None]], mutant: Mutant
) -> Callable[[str], tuple[int, dict[str, Any] | None, str | None]]:
    """Wrap an executor so that one specific regression appears in its responses."""

    def execute_mutant(operation: str) -> tuple[int, dict[str, Any] | None, str | None]:
        status, body, candidate = execute(operation)
        status, body = mutant.rewrite(operation, status, body)
        return status, body, candidate

    return execute_mutant


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleStudy:
    """The real run, and every mutant's run beside it."""

    real: Exploration
    mutants: list[tuple[Mutant, Exploration]]
    initial_version: str
    steps: int

    @property
    def caught(self) -> int:
        return sum(1 for _, run in self.mutants if not run.clean)

    @property
    def detection_rate(self) -> float:
        return self.caught / len(self.mutants) if self.mutants else 0.0


def run_lifecycle_study(settings: Settings, client: Any | None = None) -> LifecycleStudy:
    """Explore the real service, then re-run the identical walk against each mutant."""
    from netsentry.models.registry import latest_bundle
    from netsentry.serving.app import create_app

    probe = settings.model_copy(deep=True)
    probe.serving.reload_enabled = True  # the lifecycle under test is the one with reload in it
    # The guard is only worth checking when it is switched on, and the rate limiter's window is
    # wall-clock, so a burst mid-walk would poison every later step with a 429 nobody asked for.
    probe.serving.api_key = probe.serving.api_key or "lifecycle-probe-key"
    probe.serving.rate_limit_per_minute = 0
    models_dir = probe.paths.models_dir

    # Order matters. The engine resolves "the newest bundle in the models directory", so the
    # deliberately-corrupted copy has to be written *after* the app has bound to the real one --
    # otherwise the service under test is the broken bundle and every result is about that.
    # A stale copy from an interrupted run would do the same, so it is removed first.
    (models_dir / CANARY_FAIL_BUNDLE).unlink(missing_ok=True)
    resolved = probe.serving.artifact_path or latest_bundle(probe)
    if resolved is None:
        raise FileNotFoundError("No model bundle to drive the lifecycle against.")
    source = Path(resolved)

    from fastapi.testclient import TestClient

    live = client if client is not None else TestClient(create_app(probe))
    failing = write_canary_failing_bundle(probe, source, models_dir / CANARY_FAIL_BUNDLE)
    try:
        execute = make_executor(live, probe, source.name, failing.name)
        version = str((live.get("/health").json() or {}).get("model_version", "unknown"))
        initial = LifecycleState(
            version=version,
            canary_ok=True,
            api_key=probe.serving.api_key,
            seen_versions=frozenset(),
        )
        rng = np.random.default_rng(settings.seed)
        plan = plan_operations(
            settings.lifecycle.steps,
            rng,
            min_heavy=settings.lifecycle.min_heavy,
            min_light=settings.lifecycle.min_light,
        )
        real = explore(execute, initial, len(plan), rng, label="the real service", plan=plan)
    finally:
        failing.unlink(missing_ok=True)

    mutants = [
        (mutant, grade(apply_mutation(real.transcript, mutant.rewrite), initial, mutant.name))
        for mutant in MUTANTS
    ]
    return LifecycleStudy(
        real=real, mutants=mutants, initial_version=version, steps=settings.lifecycle.steps
    )


def _coverage_table(exploration: Exploration) -> str:
    counts = exploration.coverage()
    rows = "\n".join(
        f"| {operation} | {counts[operation]} |" for operation in OPERATIONS if operation in counts
    )
    return "| operation | times exercised |\n|---|---|\n" + rows


def _mutant_table(study: LifecycleStudy) -> str:
    rows = []
    for mutant, run in study.mutants:
        first = run.violations[0] if run.violations else None
        detail = f"step {first.index}: {first.violations[0]}" if first else "**not caught**"
        rows.append(f"| {mutant.name} | {mutant.description} | {len(run.violations)} | {detail} |")
    header = (
        "| injected regression | what it imitates | steps that disagreed | first disagreement |"
    )
    return header + "\n|---|---|---|---|\n" + "\n".join(rows)


def _render(study: LifecycleStudy) -> str:
    real = study.real
    verdict = (
        "no disagreement between the model and the service"
        if real.clean
        else f"**{len(real.violations)} disagreements** (listed below)"
    )
    return f"""# NetSentry — The Serving Lifecycle, as a State Machine

_A model of what the API is allowed to do, driven against the real application for
{real.n_operations} random operations, then re-run against {len(study.mutants)} deliberately
broken versions of the same service. Regenerate with `netsentry statemachine`._

## Why this report exists

Every part of the serving lifecycle has a test: the bundle loads, the canary replays, the guard
rejects a missing key, a reload with a bad path is refused. What none of them covers is the
part that breaks in production -- the **sequences**. A reload that half-succeeds, an
authentication check that stops applying after a swap, a health endpoint still reporting the
version it used to serve: each is a two-step bug, and a suite of single-step tests is
structurally unable to see any of them.

So the contract is written down as a model -- a state machine holding only what an observer can
check -- and the real service is driven through random sequences, with model and service
compared after every single step.

## The properties

1. **A refused reload changes nothing.** Not the version, not the health, not the next
   prediction. A 409 that leaves a half-swapped engine looks identical to a clean refusal until
   the next request arrives.
2. **Only a successful reload may change the served version.** Every response that names a
   version is checked against the version the model believes is live.
3. **A refusal is a refusal.** A canary mismatch, a missing bundle and a path outside the
   models directory must each produce their own error status and never a success.
4. **Health never claims `ok` while its own canary is failing**, since that is the signal every
   orchestrator uses to decide whether to send traffic.
5. **The guard applies to the guarded routes in every state**, including after a swap.

## The run

Starting from model version `{study.initial_version}`, the machine performed
{real.n_operations} operations: {verdict}.

{_coverage_table(real)}

The mix is not realistic and is not meant to be. A production trace is 99% predictions, which is
the sequence single-request tests already cover; the interesting transitions are the rare ones,
so the schedule allocates them deliberately.

That allocation replaced a weighted random draw, and the reason is worth recording. Two of the
ten operations construct an entire inference engine and take seconds rather than milliseconds,
so they have to be drawn less often -- and under a weighted draw the headline run came back with
**zero** successful reloads. The most important positive transition in the lifecycle went
unexercised, and the report would have said nothing about it while looking complete. The
schedule now allocates the expensive operations explicitly, fills the remainder with cheap ones
and shuffles: the order stays random, the coverage stops being a matter of luck.

## Does the machine catch anything?

A conformance machine that has never failed is indistinguishable from one that cannot fail, so
each mutant below is a specific regression injected into the service's responses -- the
observable symptom of a real bug -- and the identical walk is re-run against it.

**{study.caught} of {len(study.mutants)} caught.**

{_mutant_table(study)}

The mutants are injected at the HTTP boundary rather than inside the application. That buys
determinism and costs realism: it proves the model notices the *symptom* of each regression,
not that the code path producing it is reachable. The symptom is what a monitoring system sees,
which is the same reason the properties are stated in terms of an observer.

## Scope and honest limits

- **The walk is random, not exhaustive.** {real.n_operations} operations over
  {len(OPERATIONS)} verbs cover the pairs and most triples that matter, and prove nothing about
  a rare interleaving nobody drew. The Hypothesis-driven version of this machine lives in the
  test suite, where a failure shrinks to a minimal sequence instead of a long one.
- **The model is deliberately weaker than the implementation.** It holds a version, a canary
  state and a key -- not thresholds, not the engine, not the shadow model. A model that
  mirrored the implementation could not disagree with it, and a model that cannot disagree
  cannot find anything.
- **Concurrency is not tested here.** Every operation is sequential. The claim that an in-flight
  request finishes on the model it started with rests on the swap being a single reference
  reassignment, which this checks by inspection rather than by racing it.
- **A canary-failing bundle is manufactured by perturbing the stored expectations**, which
  reproduces the symptom of an environment mismatch without needing a second training run to
  create one. The gate cannot tell the two apart, and that is the point of the gate."""


def run_lifecycle_report(settings: Settings, client: Any | None = None) -> Path:
    """Run the lifecycle study and write the report."""
    study = run_lifecycle_study(settings, client)
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study), encoding="utf-8")
    logger.info(
        "Wrote state-machine report",
        extra={
            "path": str(out_path),
            "violations": len(study.real.violations),
            "mutants_caught": study.caught,
        },
    )

    with track_run(settings, "statemachine") as run:
        run.log_params({"steps": study.steps, "mutants": len(study.mutants)})
        run.log_metrics(
            {
                "violations": float(len(study.real.violations)),
                "mutant_detection_rate": study.detection_rate,
            }
        )
        run.log_artifact(out_path)
    return out_path
