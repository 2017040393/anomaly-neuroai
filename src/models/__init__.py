"""Model building utilities."""

from .build_model import build_model
from .mlp import MLPClassifier

__all__ = ["MLPClassifier", "build_model"]

try:
    from .snn import SNNClassifier
except ImportError:
    SNNClassifier = None
else:
    __all__.append("SNNClassifier")
