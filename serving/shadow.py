"""Shadow deployment logic — run both Production and Staging models, track divergence."""

import logging
import sys
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent.parent))

from serving.metrics import SHADOW_DIVERGENCE_RATE

logger = logging.getLogger(__name__)


class ShadowTracker:
    """Thread-safe tracker for shadow vs production prediction divergence."""

    def __init__(self) -> None:
        self._total: int = 0
        self._diverged: int = 0
        self._lock = Lock()

    def record(self, prod_prediction: int, shadow_prediction: int) -> None:
        """Record one shadow comparison and update the divergence gauge.

        Args:
            prod_prediction: Prediction from the Production model.
            shadow_prediction: Prediction from the Staging/shadow model.
        """
        with self._lock:
            self._total += 1
            if prod_prediction != shadow_prediction:
                self._diverged += 1
            rate = self._diverged / self._total if self._total > 0 else 0.0
            SHADOW_DIVERGENCE_RATE.set(rate)
            logger.debug(
                f"Shadow divergence: {self._diverged}/{self._total} = {rate:.2%}"
            )

    @property
    def divergence_rate(self) -> float:
        """Current divergence rate as a fraction between 0 and 1."""
        with self._lock:
            return self._diverged / self._total if self._total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """Total number of shadow comparisons recorded."""
        with self._lock:
            return self._total
