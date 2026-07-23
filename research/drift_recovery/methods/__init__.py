"""Pure-NumPy method implementations for the D2 crossover kill-screen."""

from research.drift_recovery.methods.affine import AffineRidgeAdapter
from research.drift_recovery.methods.bounded_chelation import DetectorGatedChelationAdapter
from research.drift_recovery.methods.hubness import HubnessScoreScaling
from research.drift_recovery.methods.isotropy import AllButTopAdapter, CBIEAdapter, ZCAAdapter
from research.drift_recovery.methods.local_adapters import SoftRoutedLocalAdapter

__all__ = [
    "AffineRidgeAdapter",
    "AllButTopAdapter",
    "CBIEAdapter",
    "DetectorGatedChelationAdapter",
    "HubnessScoreScaling",
    "SoftRoutedLocalAdapter",
    "ZCAAdapter",
]
