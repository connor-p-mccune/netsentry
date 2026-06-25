"""Named feature groups, including the with/without ``Destination Port`` variants.

Implemented in Phase 3.
"""

from __future__ import annotations


def get_feature_set(name: str, *, include_destination_port: bool = False) -> list[str]:
    """Return the ordered feature columns for a named feature set."""
    raise NotImplementedError("Implemented in Phase 3")
