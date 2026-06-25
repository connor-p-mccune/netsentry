"""Global deterministic seeding.

Reproducibility is a project invariant: every run is recreatable from its logged
config + seed. This sets every RNG we touch (Python, NumPy, and — if installed —
PyTorch) from a single seed so results do not drift between runs.
"""

from __future__ import annotations

import os
import random

import numpy as np

from netsentry.log import get_logger

logger = get_logger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all relevant RNGs from one value.

    scikit-learn / LightGBM estimators are seeded separately via their
    ``random_state`` constructor argument (driven from the same config seed);
    this covers the global generators and PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is an optional dependency (autoencoder only)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    logger.debug("Seeded all RNGs", extra={"seed": seed})
