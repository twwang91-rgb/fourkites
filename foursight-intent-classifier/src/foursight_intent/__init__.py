"""FourSight intent classification package.

Single-turn intent classifier for the FourKites FourSight assistant plus the
multi-turn contextual resolver, calibration utilities and submission tooling.
"""

from foursight_intent.labels import LABELS, LABEL2ID, ID2LABEL

__all__ = ["LABELS", "LABEL2ID", "ID2LABEL"]
__version__ = "0.1.0"
