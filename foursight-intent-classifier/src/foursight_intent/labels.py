"""The closed seven-label taxonomy. Order matches labels.json / score.py."""

from __future__ import annotations

LABELS: list[str] = [
    "shipment_information",
    "realtime_query",
    "analytics",
    "disruptions",
    "chitchat",
    "knowledge_base",
    "other",
]
LABEL_SET: frozenset[str] = frozenset(LABELS)
LABEL2ID: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

# The three operationally adjacent classes the challenge calls out as the hard boundary.
OPERATIONAL_TRIAD: tuple[str, str, str] = ("shipment_information", "realtime_query", "analytics")
RARE_CLASSES: tuple[str, str, str] = ("disruptions", "chitchat", "other")

# Taxonomy version: bump when the label set changes (see docs/new_intents.md).
TAXONOMY_VERSION = "v1"


def validate_label(label: str) -> str:
    """Return the canonical label or raise ValueError if it is outside the closed set."""
    norm = (label or "").strip().lower()
    if norm not in LABEL_SET:
        raise ValueError(f"label {label!r} is not one of {LABELS}")
    return norm
