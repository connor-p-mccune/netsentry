"""Hand-written signature rules — the baseline a SOC already has.

Before an ML detector earns its complexity it must beat the incumbent: a handful
of interpretable, port-scoped threshold rules of the kind analysts write for
Snort/Suricata. Each rule is a conjunction of clauses over raw flow features,
declared in config (``rules.definitions``) so it can be audited and tuned like a
real ruleset. The engine is deliberately fit-free: rules encode domain knowledge,
not training data, so there is nothing to leak and nothing to calibrate — which is
both their appeal (auditable, deployable today) and their ceiling (no
precision/recall dial, no coverage of patterns nobody wrote a signature for).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from netsentry.log import get_logger

if TYPE_CHECKING:
    from netsentry.config.settings import RuleDefinition

logger = get_logger(__name__)


class RuleEngine:
    """Evaluate configured signature rules over a raw (cleaned) flow frame."""

    def __init__(self, definitions: list[RuleDefinition]) -> None:
        self.definitions = list(definitions)

    def matches(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-rule boolean fire mask — one column per rule, aligned to ``df``."""
        fired = {rule.name: self._rule_mask(df, rule) for rule in self.definitions}
        return pd.DataFrame(fired, index=df.index)

    def decisions(self, df: pd.DataFrame) -> np.ndarray:
        """Union decision: a flow is flagged when *any* rule fires."""
        matches = self.matches(df)
        if matches.shape[1] == 0:
            return np.zeros(len(df), dtype=bool)
        return np.asarray(matches.to_numpy().any(axis=1))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Fraction of rules fired per flow — a coarse severity, not a probability.

        Exposed so the rule set can sit on a ranking plot next to the model, but the
        honest operating point of a ruleset is its binary union decision: real
        signature engines alert or stay silent, they do not emit scores.
        """
        matches = self.matches(df)
        if matches.shape[1] == 0:
            return np.zeros(len(df), dtype=float)
        return np.asarray(matches.to_numpy().mean(axis=1))

    def _rule_mask(self, df: pd.DataFrame, rule: RuleDefinition) -> np.ndarray:
        """One rule's fire mask. A rule referencing an absent feature never fires."""
        missing = [c.feature for c in rule.clauses if c.feature not in df.columns]
        if missing:
            logger.warning(
                "Rule %r disabled: feature(s) %s not in the input frame", rule.name, missing
            )
            return np.zeros(len(df), dtype=bool)
        mask = np.ones(len(df), dtype=bool)
        for clause in rule.clauses:
            col = pd.to_numeric(df[clause.feature], errors="coerce")
            if clause.op == "ge":
                hit = col >= clause.value
            elif clause.op == "le":
                hit = col <= clause.value
            else:  # eq — NaN compares False, so an unset feature never satisfies a clause
                hit = col == clause.value
            mask &= hit.to_numpy(dtype=bool)
        return mask
