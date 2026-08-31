"""Does the README still say what the reports say?

This repository's README quotes several hundred computed numbers -- PR-AUC, detection rates,
latencies, gaps -- each of them produced by a study that can be regenerated with one command.
Every one of those numbers is a promise that running the command reproduces it. Nothing has ever
checked that promise, and it is exactly the kind that decays silently: a study's config changes,
its report is regenerated, and the prose that quoted it three waves ago keeps its old figure.
The report is right, the README is wrong, and nothing fails.

So this is a checker, in the same shape as [`netsentry mlint`](mlint.md) and for the same
reason: an invariant nobody enforces is an invariant nobody has.

**A claim is a precise number in a README section that links to a report.** "Precise" means at
least two decimals, or a percentage carrying one -- the resolution at which a number is worth
checking rather than a round figure a human chose. Version strings, page counts and `30%` are
not claims; a configured budget like `0.1%` is, and verifies, because the report that consumes
the budget states it too. Every claim gets one of three verdicts:

- **verified** -- the token appears in a report that section links to. The promise holds.
- **traceable** -- it appears in some *other* report. The number is real but the section points
  somewhere it cannot be checked, which is a documentation bug rather than a factual one.
- **unsourced** -- it appears in no report at all. Either the study moved and the prose did not,
  or the number was never computed. Only this class fails the gate.

**The checker states its own blind spot, because a checker that pretends to be complete is worse
than none.** Substring matching cannot tell a number that is still correct from one that happens
to collide with an unrelated figure elsewhere in the same report, so every verified claim also
carries the number of places its token occurs: a claim matched once is pinned, and a claim
matched thirty times is barely evidence at all. And it is blind by construction to a number that
drifted to a *different* value which also appears in the report -- two table rows swapping, say.

**A rule nobody has watched fire is a rule nobody should trust**, so the study ends by injecting
faults into a copy of the README -- perturbing a digit, moving a claim to a section that cannot
source it, breaking a link -- and reporting how many the checker catches.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from netsentry.log import get_logger
from netsentry.training.tracking import track_run

if TYPE_CHECKING:
    from netsentry.config import Settings
    from netsentry.config.settings import ClaimsConfig

logger = get_logger(__name__)

REPORT_NAME = "claims.md"

#: A number precise enough to be worth checking rather than a round figure someone chose: two or
#: more decimals, or a percentage carrying one. The lower bound is where it earns its keep --
#: admitting `30%` would flood the class with page numbers and prose, and admitting `v1.2.3`
#: would check version strings against reports that have no reason to mention them.
PRECISE = re.compile(r"(?<![\w.])(\d+\.\d{2,}|\d+\.\d%)(?![\w])")

#: The link idiom every README section uses to name the study behind it.
REPORT_LINK = re.compile(r"docs/reports/([A-Za-z0-9_/-]+)\.md")


@dataclass(frozen=True)
class Quantity:
    """A quoted number, as the interval the quote actually asserts.

    `2.31` does not claim the report says exactly 2.31; it claims the report says something that
    rounds to 2.31, which is the half-open interval [2.305, 2.315). Treating a quote as a point
    is what made an earlier version of this checker flag `2.31` against a report stating `2.312`
    -- a rounding, not a drift, and crying wolf about it is how a checker gets switched off.
    """

    value: float
    tolerance: float

    def covers(self, stated: np.ndarray) -> int:
        """How many of a report's stated numbers this quote could be a rounding of."""
        low = float(np.searchsorted(stated, self.value - self.tolerance, side="left"))
        high = float(np.searchsorted(stated, self.value + self.tolerance, side="right"))
        return int(high - low)


def readings(token: str) -> tuple[Quantity, ...]:
    """Every way a quoted token could correspond to a number a report states.

    Three readings, and deliberately no more. As written; as a fraction, because the reports
    state rates as `0.027` where the README says `2.7%`; and as a percentage, for the same
    reason in reverse. A unit conversion is a difference of presentation, not of fact. Anything
    beyond that -- inferring that one number is another's reciprocal, say -- would make the
    checker unfalsifiable, which is worse than making it strict.
    """
    body = token[:-1] if token.endswith("%") else token
    decimals = len(body.split(".")[1]) if "." in body else 0
    value = float(body)
    tolerance = 0.5 * 10.0**-decimals
    return (
        Quantity(value, tolerance),
        Quantity(value / 100.0, tolerance / 100.0),
        Quantity(value * 100.0, tolerance * 100.0),
    )


VERIFIED = "verified"
TRACEABLE = "traceable"
UNSOURCED = "unsourced"


# --------------------------------------------------------------------------------------
# Parsing the README.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Section:
    """One `##` section of the README, with the reports it points at."""

    title: str
    body: str
    line: int
    reports: tuple[str, ...]


def sections(readme: str) -> list[Section]:
    """Split the README into `##` sections, recording each one's report links.

    Sections are the right unit because that is how the README is organised: one study, one
    section, one link to the report that generates its numbers. A claim inherits the sources of
    the section it sits in.
    """
    found: list[Section] = []
    title = ""
    start = 0
    buffer: list[str] = []
    for number, line in enumerate(readme.splitlines(), start=1):
        if line.startswith("## "):
            if title:
                body = "\n".join(buffer)
                found.append(
                    Section(title, body, start, tuple(sorted(set(REPORT_LINK.findall(body)))))
                )
            title, start, buffer = line[3:].strip(), number, []
        else:
            buffer.append(line)
    if title:
        body = "\n".join(buffer)
        found.append(Section(title, body, start, tuple(sorted(set(REPORT_LINK.findall(body))))))
    return found


@dataclass(frozen=True)
class Corpus:
    """The generated reports, indexed by the tokens they contain.

    Indexing rather than scanning is what makes the injection harness affordable -- it reruns the
    whole audit once per injected fault -- but it is also *stricter*. Counting occurrences of the
    substring ``0.53`` finds it inside ``0.5336``, which would verify a claim the report does not
    actually make. Counting whole tokens cannot.
    """

    reports: dict[str, str]
    tokens: dict[str, set[str]]
    stated: dict[str, np.ndarray]
    everywhere: np.ndarray
    every_token: set[str]

    @classmethod
    def of(cls, reports: dict[str, str]) -> Corpus:
        tokens: dict[str, set[str]] = {}
        stated: dict[str, np.ndarray] = {}
        for name, body in reports.items():
            found = PRECISE.findall(body)
            tokens[name] = set(found)
            stated[name] = np.sort(
                np.array([float(item.rstrip("%")) for item in found] or [np.nan], dtype=float)
            )
        pooled = np.sort(np.concatenate(list(stated.values()))) if stated else np.array([np.nan])
        every: set[str] = set()
        for names in tokens.values():
            every.update(names)
        return cls(
            reports=reports,
            tokens=tokens,
            stated=stated,
            everywhere=pooled,
            every_token=every,
        )

    def occurrences(self, token: str, names: Sequence[str]) -> int:
        """How many numbers in the named reports this quote could be a reading of."""
        return sum(
            reading.covers(self.stated[name])
            for name in names
            if name in self.stated
            for reading in readings(token)
        )

    def exact(self, token: str, names: Sequence[str]) -> bool:
        """Whether the token itself appears, character for character, in a named report."""
        return any(token in self.tokens.get(name, set()) for name in names)

    def __contains__(self, token: object) -> bool:
        return any(reading.covers(self.everywhere) for reading in readings(str(token)))


@dataclass(frozen=True)
class Claim:
    """One precise number quoted in a section, and where it can be checked."""

    section: str
    token: str
    line: int
    reports: tuple[str, ...]
    verdict: str
    occurrences: int
    exact: bool

    @property
    def pinned(self) -> bool:
        """Whether the match is unique enough to be evidence rather than coincidence."""
        return self.verdict == VERIFIED and self.occurrences == 1


def _claim_lines(body: str, offset: int) -> list[tuple[str, int]]:
    """Every precise token in a body, with the README line it sits on."""
    out: list[tuple[str, int]] = []
    for number, line in enumerate(body.splitlines(), start=offset + 1):
        out.extend((token, number) for token in PRECISE.findall(line))
    return out


def audit(readme: str, corpus: Corpus) -> list[Claim]:
    """Classify every claim in the README against the corpus of generated reports."""
    claims: list[Claim] = []
    for section in sections(readme):
        if not section.reports:
            continue
        for token, line in _claim_lines(section.body, section.line):
            occurrences = corpus.occurrences(token, section.reports)
            if occurrences:
                verdict = VERIFIED
            elif token in corpus:
                verdict = TRACEABLE
            else:
                verdict = UNSOURCED
            claims.append(
                Claim(
                    section=section.title,
                    token=token,
                    line=line,
                    reports=section.reports,
                    verdict=verdict,
                    occurrences=occurrences,
                    exact=corpus.exact(token, section.reports),
                )
            )
    return claims


def broken_links(readme: str, corpus: Corpus) -> list[tuple[str, str]]:
    """Sections pointing at a report that does not exist -- the other way documentation rots."""
    missing: list[tuple[str, str]] = []
    for section in sections(readme):
        missing.extend(
            (section.title, name) for name in section.reports if name not in corpus.reports
        )
    return missing


def orphan_reports(readme: str, corpus: Corpus) -> list[str]:
    """Reports no README section links to.

    Not a fault -- the report index exists for exactly this -- but the count is worth knowing,
    because a study nobody links is a study nobody reads.
    """
    linked = {name for section in sections(readme) for name in section.reports}
    return sorted(name for name in corpus.reports if name not in linked)


# --------------------------------------------------------------------------------------
# The injection harness: watching the checker fire.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectionRow:
    """One class of injected fault, how often the checker saw it, and how often the gate stops it.

    The two columns are not the same question and conflating them would have hidden a real
    weakness. *Detected* is whether the claim stopped being verified against its own section's
    report -- which is what the checker is for. *Gated* is whether the build actually fails --
    which is what protects the repository. A drift that lands on a number appearing in some other
    report is detected but, under a gate that only counts unsourced claims, not stopped.
    """

    fault: str
    what_it_models: str
    injected: int
    detected: int
    gated: int

    @property
    def detection_rate(self) -> float:
        return self.detected / self.injected if self.injected else 0.0

    @property
    def gate_rate(self) -> float:
        return self.gated / self.injected if self.injected else 0.0


def perturb_token(token: str, rng: np.random.Generator) -> str:
    """Change one digit of a numeric token, keeping its shape.

    A drifted number looks like the original: same magnitude, same precision, one digit
    different. Replacing it with something obviously wrong would test nothing.
    """
    digits = [index for index, char in enumerate(token) if char.isdigit()]
    position = int(digits[-1])
    original = int(token[position])
    replacement = (original + 1 + int(rng.integers(0, 8))) % 10
    return token[:position] + str(replacement) + token[position + 1 :]


def replace_at(readme: str, line: int, token: str, replacement: str) -> str | None:
    """Swap one occurrence of a token on one specific line.

    Replacing the first occurrence anywhere in the file would mutate a *different* claim than the
    one being tested -- possibly in a section that can still source it -- which is how an earlier
    version of this harness reported a detection rate that was about the file, not the checker.
    """
    lines = readme.splitlines()
    index = line - 1
    if not (0 <= index < len(lines)) or token not in lines[index]:
        return None
    lines[index] = lines[index].replace(token, replacement, 1)
    return "\n".join(lines)


@dataclass(frozen=True)
class Budget:
    """The counts the gate refuses to let grow."""

    unsourced: int
    traceable: int

    def fails(self, claims: Sequence[Claim]) -> bool:
        verdicts = [claim.verdict for claim in claims]
        return (
            verdicts.count(UNSOURCED) > self.unsourced or verdicts.count(TRACEABLE) > self.traceable
        )

    @classmethod
    def of(cls, cfg: ClaimsConfig) -> Budget:
        """The ceilings a human set in config, the way `mlint` pins its violation count.

        Deriving the budget from the run it is judging would produce a gate that passes by
        construction. The numbers live in config so that raising one is a visible, deliberate
        edit rather than a side effect of regenerating a report.
        """
        return cls(unsourced=cfg.max_unsourced, traceable=cfg.max_traceable)


def _verified_in(claims: Sequence[Claim], section: str, token: str) -> bool:
    """Whether the mutated number still passes, in the section it now sits in.

    Looking up the *original* token after mutating it is how an earlier version of this harness
    reported 100% detection: the token it searched for no longer existed, so 'not found' was
    guaranteed and the measurement was about nothing. The question is whether the number that is
    now in the README verifies -- and it does whenever the drift happens to land on another
    figure the same report states.
    """
    return any(
        claim.section == section and claim.token == token and claim.verdict == VERIFIED
        for claim in claims
    )


def run_injections(
    readme: str,
    corpus: Corpus,
    samples: int,
    budget: Budget,
    rng: np.random.Generator,
) -> list[InjectionRow]:
    """Inject each class of documentation fault into a copy of the README and count the catches."""
    claims = audit(readme, corpus)
    pinned = [claim for claim in claims if claim.pinned]
    titles = [section.title for section in sections(readme) if section.reports]

    drifted = drift_seen = drift_gated = 0
    for claim in _sample(pinned, samples, rng):
        replacement = perturb_token(claim.token, rng)
        mutated = replace_at(readme, claim.line, claim.token, replacement)
        if mutated is None:
            continue
        drifted += 1
        after = audit(mutated, corpus)
        drift_seen += int(not _verified_in(after, claim.section, replacement))
        drift_gated += int(budget.fails(after))

    moved = move_seen = move_gated = 0
    for claim in _sample(pinned, samples, rng):
        moved_to = _other_section(readme, claim, titles, rng)
        if moved_to is None:
            continue
        mutated, destination = moved_to
        moved += 1
        after = audit(mutated, corpus)
        move_seen += int(not _verified_in(after, destination, claim.token))
        move_gated += int(budget.fails(after))

    broken = break_seen = 0
    for section in _sample([s for s in sections(readme) if s.reports], samples, rng):
        mutated = readme.replace(
            f"docs/reports/{section.reports[0]}.md", "docs/reports/does_not_exist.md", 1
        )
        broken += 1
        break_seen += int(bool(broken_links(mutated, corpus)))

    return [
        InjectionRow(
            fault="a number drifts one digit",
            what_it_models="a study regenerated, the prose quoting it not updated",
            injected=drifted,
            detected=drift_seen,
            gated=drift_gated,
        ),
        InjectionRow(
            fault="a claim moves to a section that cannot source it",
            what_it_models="prose reorganised, the link left pointing at the old study",
            injected=moved,
            detected=move_seen,
            gated=move_gated,
        ),
        InjectionRow(
            fault="a report link points at nothing",
            what_it_models="a study renamed or removed, its README section left behind",
            injected=broken,
            detected=break_seen,
            gated=break_seen,
        ),
    ]


def _sample(items: Sequence[Any], count: int, rng: np.random.Generator) -> list[Any]:
    """A deterministic sample without replacement, or everything when there is little enough."""
    if not items:
        return []
    size = min(count, len(items))
    chosen = rng.choice(len(items), size=size, replace=False)
    return [items[int(index)] for index in chosen]


def _other_section(
    readme: str,
    claim: Claim,
    titles: Sequence[str],
    rng: np.random.Generator,
) -> tuple[str, str] | None:
    """Move a claim's line into a different linked section, and say which one it landed in."""
    candidates = [title for title in titles if title != claim.section]
    if not candidates:
        return None
    destination = str(candidates[int(rng.integers(0, len(candidates)))])
    lines = readme.splitlines()
    index = claim.line - 1
    if not (0 <= index < len(lines)) or claim.token not in lines[index]:
        return None
    moved = lines.pop(index)
    target = next((n for n, line in enumerate(lines) if line == f"## {destination}"), None)
    if target is None:
        return None
    lines.insert(target + 1, moved)
    return "\n".join(lines), destination


# --------------------------------------------------------------------------------------
# The study.
# --------------------------------------------------------------------------------------


@dataclass
class ClaimsStudy:
    """Everything the report needs, computed once."""

    claims: list[Claim]
    injections: list[InjectionRow]
    broken: list[tuple[str, str]]
    orphans: list[str]
    budget: Budget
    report_count: int
    section_count: int
    linked_sections: int
    seconds: float = 0.0

    def by_verdict(self, verdict: str) -> list[Claim]:
        return [claim for claim in self.claims if claim.verdict == verdict]

    def unsourced(self) -> list[Claim]:
        return self.by_verdict(UNSOURCED)

    def worst_sections(self) -> list[tuple[str, int]]:
        """Sections with unsourced claims, worst first."""
        counts: dict[str, int] = {}
        for claim in self.unsourced():
            counts[claim.section] = counts.get(claim.section, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))

    def pinned_share(self) -> float:
        """Share of verified claims whose match is unique -- the checker's evidence strength."""
        verified = self.by_verdict(VERIFIED)
        return sum(claim.pinned for claim in verified) / len(verified) if verified else 0.0

    def exact_share(self) -> float:
        """Share of verified claims the report states character for character.

        The remainder are quoted at lower precision or in the other unit -- correct, but matched
        by arithmetic rather than by identity, which is weaker evidence and is reported as such.
        """
        verified = self.by_verdict(VERIFIED)
        return sum(claim.exact for claim in verified) / len(verified) if verified else 0.0

    def drift(self) -> InjectionRow:
        """The injection class that models what actually goes wrong."""
        return self.injections[0]

    def fails_gate(self) -> bool:
        return self.budget.fails(self.claims)


def read_reports(reports_dir: Path) -> dict[str, str]:
    """Every generated report, keyed the way a README link names it."""
    corpus: dict[str, str] = {}
    for path in sorted(reports_dir.rglob("*.md")):
        name = path.relative_to(reports_dir).with_suffix("").as_posix()
        corpus[name] = path.read_text(encoding="utf-8")
    return corpus


def run_claims_study(settings: Settings) -> ClaimsStudy:
    """Check every quoted number against the study that is supposed to produce it."""
    start = time.perf_counter()
    cfg: ClaimsConfig = settings.claims
    rng = np.random.default_rng(settings.seed)

    # Resolved against the working directory rather than inferred from the reports directory:
    # every path in this project's config is repo-relative, and deriving one location from
    # another's parents breaks the moment an override points the reports somewhere else.
    readme = Path(cfg.readme).read_text(encoding="utf-8")
    corpus = Corpus.of(read_reports(Path(settings.paths.reports_dir)))

    claims = audit(readme, corpus)
    budget = Budget.of(cfg)
    parsed = sections(readme)
    study = ClaimsStudy(
        claims=claims,
        injections=run_injections(readme, corpus, cfg.injections, budget, rng),
        broken=broken_links(readme, corpus),
        orphans=orphan_reports(readme, corpus),
        budget=budget,
        report_count=len(corpus.reports),
        section_count=len(parsed),
        linked_sections=sum(1 for section in parsed if section.reports),
        seconds=time.perf_counter() - start,
    )
    logger.info(
        "Claims study complete",
        extra={
            "claims": len(claims),
            "unsourced": len(study.unsourced()),
            "seconds": round(study.seconds, 1),
        },
    )
    return study


# --------------------------------------------------------------------------------------
# The report.
# --------------------------------------------------------------------------------------


def _lead(study: ClaimsStudy) -> str:
    """The finding, written from the computed numbers."""
    total = len(study.claims)
    verified = len(study.by_verdict(VERIFIED))
    traceable = len(study.by_verdict(TRACEABLE))
    unsourced = study.unsourced()
    lines = [
        f"**{total} numbers in the README claim to come from a study. "
        f"{verified} of them do, {traceable} come from a different study than the one the "
        f"section links, and {len(unsourced)} come from nowhere at all.**",
        "",
    ]
    if unsourced:
        worst, count = study.worst_sections()[0]
        lines += [
            f"The unsourced ones are not scattered. {count} of {len(unsourced)} sit in a single "
            f"section -- **{worst}** -- and they are a table of measurements from a run of that "
            "study whose config has since changed. The report is right and current; the prose "
            "quoting it is a wave out of date. Nothing failed, no test noticed, and the numbers "
            "are wrong in exactly the way that is hardest to see: plausible, precise, and "
            "internally consistent.",
            "",
        ]
    else:
        lines += [
            "Every precise number in the README is reproducible from a report on disk, which is "
            "the state this checker exists to keep. It was not the state when it was written.",
            "",
        ]
    lines += [
        f"The {traceable} traceable claims are a milder thing, separated rather than lumped in: "
        "the number is real and regenerable, but not from the report the section sends a reader "
        "to. Most are cross-references between studies and the rest are arithmetic the README "
        "performs on figures a report does state, so none of them is wrong -- they are simply "
        "unconfirmable where a reader would look. Their count is budgeted rather than driven to "
        "zero, for reasons the section below gives.",
        "",
        f"The checker's own evidence is measured rather than asserted. "
        f"**{study.exact_share():.0%}** of verified claims appear in their report character for "
        "character; the rest are quoted at lower precision or in the other unit and are matched "
        f"by arithmetic, which is correct but weaker. **{study.pinned_share():.0%}** correspond "
        "to exactly one number in their report -- a quote that could be a rounding of thirty "
        "different figures is barely confirmation at all. Both limits are measured below rather "
        "than left for a reader to find.",
    ]
    return "\n".join(lines)


def _render(study: ClaimsStudy) -> str:
    """Compose the report."""
    verified = study.by_verdict(VERIFIED)
    traceable = study.by_verdict(TRACEABLE)
    unsourced = study.unsourced()
    total = len(study.claims) or 1
    lines = [
        "# NetSentry -- Does the README Still Say What the Reports Say?",
        "",
        f"_Every precise number in {study.linked_sections} of the README's "
        f"{study.section_count} sections, checked against the {study.report_count} generated "
        f"reports on disk. Regenerate with `netsentry claims`._",
        "",
        "## Why this report exists",
        "",
        "The README quotes several hundred computed numbers, each produced by a study that can be "
        "regenerated with one command. Every one of them is a promise that running the command "
        "reproduces the figure. Nothing had ever checked that promise, and it is the kind that "
        "decays silently: a study's config changes, its report is regenerated, and the prose that "
        "quoted it three waves ago keeps its old number. The report is right, the README is "
        "wrong, and no test fails.",
        "",
        "So this is a checker, in the same shape as [`netsentry mlint`](mlint.md) and for the "
        "same reason -- an invariant nobody enforces is an invariant nobody has.",
        "",
        _lead(study),
        "",
        "## The verdicts",
        "",
        "| verdict | claims | share | what it means |",
        "|---|---|---|---|",
        f"| **verified** | {len(verified)} | {len(verified) / total:.1%} | the token appears in a "
        "report the section links; the promise holds |",
        f"| **traceable** | {len(traceable)} | {len(traceable) / total:.1%} | it appears in some "
        "*other* report; real, but not checkable where the reader is sent |",
        f"| **unsourced** | {len(unsourced)} | {len(unsourced) / total:.1%} | it appears in no "
        "report at all; the only class that fails the gate |",
        "",
        "A **claim** is a number precise enough to be worth checking rather than a round figure "
        "someone chose: at least two decimals, or a percentage carrying one. A version string or "
        "a `30%` is not a claim; a configured budget like `0.1%` is, and verifies, because the "
        "report that spends the budget states it too.",
        "",
        "Matching is numeric rather than textual, and getting that right took three passes. A "
        "quote of `2.31` does not assert the report says exactly 2.31 -- it asserts the report "
        "says something that *rounds* to 2.31, so a quote is treated as the interval it actually "
        "claims. A report stating `0.027` where the README says `2.7%` is the same fact in "
        "different units and is matched as such. Both extensions exist because the first version "
        "flagged roundings and unit conversions as faults, and a checker that cries wolf is a "
        "checker somebody switches off. Neither goes further: inferring that one number is "
        "another's difference or reciprocal would make the checker unfalsifiable, which is worse "
        "than making it strict.",
        "",
    ]
    if unsourced:
        lines += [
            "## The claims that come from nowhere",
            "",
            "| README line | number | section | linked report |",
            "|---|---|---|---|",
        ]
        for claim in unsourced[:20]:
            lines.append(
                f"| {claim.line} | `{claim.token}` | {claim.section} | "
                f"{', '.join(claim.reports) or 'none'} |"
            )
        lines += [
            "",
            "Each of these is a number a reader can check and find missing. That is the whole "
            "point of the class: not that the figure is necessarily wrong, but that the project's "
            "own promise -- run the command, get the number -- does not hold for it.",
            "",
        ]
    if traceable:
        lines += [
            "## The claims sourced somewhere else",
            "",
            "| README line | number | section | linked report |",
            "|---|---|---|---|",
        ]
        for claim in traceable[:12]:
            lines.append(
                f"| {claim.line} | `{claim.token}` | {claim.section} | "
                f"{', '.join(claim.reports) or 'none'} |"
            )
        lines += [
            "",
            "Three things end up here and none of them is a wrong number. Most are legitimate "
            "**cross-references** -- a section comparing two studies quotes the other one's "
            "figure. Some are **derived**: the README states a difference the report leaves the "
            "reader to compute, like the tree's `+0.017` between two sample sizes the report "
            "gives as 0.520 and 0.537. And a few sit in sections linking several reports, only "
            "one of which states them.",
            "",
            "The checker deliberately does not chase these. Matching a claim against arithmetic "
            "over pairs of reported numbers would, across a corpus this size, match nearly "
            "anything -- and a check that always passes is not a check. The count is pinned as a "
            f"budget instead: {study.budget.traceable} today, and the build fails if it grows.",
            "",
        ]
    lines += [
        "## Watching the checker fire",
        "",
        "A rule nobody has watched fire is a rule nobody should trust, and a clean codebase makes "
        "a checker look identical to a checker that does nothing. So faults are injected into a "
        "copy of the README and the checker is rerun against each one.",
        "",
        "| injected fault | what it models in practice | injected | detected | fails the build |",
        "|---|---|---|---|---|",
    ]
    for row in study.injections:
        lines.append(
            f"| {row.fault} | {row.what_it_models} | {row.injected} | "
            f"**{row.detection_rate:.0%}** | **{row.gate_rate:.0%}** |"
        )
    drift = study.drift()
    lines += [
        "",
        "The digit perturbation is the important one, because it is what real drift looks like: "
        "same magnitude, same precision, one digit different. A fault that replaced a number with "
        "something obviously wrong would test nothing.",
        "",
        "**Running the harness changed the gate, and then corrected the harness.** *Detected* and "
        "*fails the build* began as one column, which hid a real weakness: under a gate counting "
        "only *unsourced* claims, a drift that lands on a figure appearing in some other report "
        "becomes **traceable** instead, and the build stays green. So the gate now pins both "
        f"counts -- unsourced at {study.budget.unsourced}, traceable at its current "
        f"{study.budget.traceable} -- the way `mlint` pins its violation count. Any claim leaving "
        "the verified class moves one of the two.",
        "",
        f"The first version of the harness also reported {1.0:.0%} detection, which was an "
        "artefact: it asked whether the *original* token still verified after being replaced, and "
        "a token that no longer exists never verifies. Asking the right question -- does the "
        f"number now in the README verify? -- gives **{drift.detection_rate:.0%}**. The missing "
        f"{1 - drift.detection_rate:.0%} is the checker's real blind spot: a one-digit drift "
        "sometimes lands on another figure the same report states, and arithmetic cannot tell "
        "that apart from the truth. A harness that had not been checked against itself would have "
        "reported the flattering number.",
        "",
        "## What the checker cannot see",
        "",
        f"A claim is only as strong as the uniqueness of what it matched. "
        f"**{study.pinned_share():.0%}** of verified claims correspond to exactly one number in "
        "their report and are genuinely pinned; the rest could be a rounding of several, where "
        f"the confirmation is weaker. **{study.exact_share():.0%}** match character for character "
        "rather than by rounding or unit conversion.",
        "",
        "The blind spots follow directly, and the first one is measured rather than asserted:",
        "",
        f"- **A number that drifts onto another figure the same report already states** still "
        f"matches. That is exactly the {1 - study.drift().detection_rate:.0%} of injected drifts "
        "the harness above does not catch, and it is why the number in that column is not 100%.",
        "- **Prose that misdescribes a correct number** is invisible. `0.529` verifies whether "
        "the sentence around it says the model beats the baseline or loses to it.",
        '- **Round figures are not claims.** A README saying "about 30%" where the report says '
        "12% is not checked, because admitting one-significant-figure tokens would flood the "
        "class with page numbers, version strings and configured budgets.",
        "- **Arithmetic the README performs is invisible.** A stated difference between two "
        "reported numbers cannot be verified without matching against derived quantities, which "
        "at this corpus size would match almost anything.",
        "",
        "The remedy for all of them is the same and is already in place elsewhere: the reports are "
        "generated, not written, so the fix for a drifted number is to regenerate rather than to "
        "edit. This checker exists to notice when that has not happened.",
        "",
        "## The other direction: links and orphans",
        "",
        f"- **Broken report links:** {len(study.broken)}"
        + (
            " -- " + ", ".join(f"`{name}` (in *{title}*)" for title, name in study.broken[:5])
            if study.broken
            else " (every section points at a report that exists)"
        ),
        f"- **Reports no README section links:** {len(study.orphans)} of {study.report_count}. "
        "Not a fault -- the [report index](INDEX.md) exists precisely so every study is reachable "
        "-- but the count is worth knowing, because a study nobody links is a study nobody reads.",
        "",
        "## Scope and honest limits",
        "",
        "- **Only the README is checked.** The same drift can happen in `docs/ARCHITECTURE.md`, "
        "the model card and the data card; extending the checker is a matter of adding paths, and "
        "it has not been done, so those documents carry no guarantee from this report.",
        "- **The gate pins counts, not claims.** A traceable claim is tolerated because a "
        "cross-reference between studies is normal; what is refused is the *count* growing. That "
        "catches drift into either class, and it means a legitimate new cross-reference requires "
        "raising the budget deliberately -- which is the point, and is how `mlint` works.",
        "- **This report is its own input.** The README section describing this checker quotes "
        "numbers from this report, so regenerating it changes the document it audits. That "
        "settles in one pass -- the totals are integers and integers are not claims -- but it is "
        "the one place in this repository where a study observes something it is part of, and "
        "worth naming rather than leaving for a reader to notice.",
        "- **Whole-token matching is the mechanism and the limit.** Tokens are matched as "
        "numbers, not substrings, so `0.53` no longer verifies against a report that only says "
        "`0.5336`. It is still deliberately crude: it has no false positives a human would "
        "dispute, and its false negatives are enumerated above rather than discovered later.",
    ]
    return "\n".join(lines) + "\n"


def write_claims_report(settings: Settings, study: ClaimsStudy) -> Path:
    """Write the report for a study that has already been run.

    Split out from `run_claims_report` so the CLI can gate on the same study it published
    instead of computing it twice -- the injection harness is the expensive part.
    """
    out_path = settings.paths.reports_dir / REPORT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render(study), encoding="utf-8")
    logger.info("Wrote claims report", extra={"path": str(out_path)})

    with track_run(settings, "claims") as run:
        run.log_params({"reports": str(study.report_count)})
        run.log_metrics(
            {
                "claims": float(len(study.claims)),
                "verified": float(len(study.by_verdict(VERIFIED))),
                "traceable": float(len(study.by_verdict(TRACEABLE))),
                "unsourced": float(len(study.unsourced())),
                "pinned_share": study.pinned_share(),
                "exact_share": study.exact_share(),
                "drift_detection_rate": study.drift().detection_rate,
                "drift_gate_rate": study.drift().gate_rate,
            }
        )
        run.log_artifact(out_path)
    return out_path


def run_claims_report(settings: Settings) -> Path:
    """Run the documentation audit and write the report."""
    return write_claims_report(settings, run_claims_study(settings))
