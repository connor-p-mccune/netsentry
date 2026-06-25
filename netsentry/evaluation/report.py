"""Generate the Markdown evaluation report.

Leads with operational metrics and contrasts the honest temporal split against
the optimistic stratified split, calling out the gap. Implemented in Phase 5.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from netsentry.config import Settings


def build_report(settings: Settings, results: dict[str, object], out_path: Path) -> Path:
    """Write the evaluation report and return its path."""
    raise NotImplementedError("Implemented in Phase 5")
