"""The leakage firewall: one fitted sklearn pipeline applied at train and serve.

Drops identifier/leaky columns, imputes (median, train-fit), scales, and
optionally encodes ``Destination Port``. Fitting happens on the training split
ONLY. Implemented in Phase 3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from netsentry.config import Settings


def build_pipeline(settings: Settings) -> Pipeline:
    """Construct (unfitted) the leakage-safe preprocessing pipeline."""
    raise NotImplementedError("Implemented in Phase 3")
