"""Light, deterministic text normalisation shared by every model and the inference path.

We deliberately keep normalisation minimal: casing, punctuation and reference numbers
carry signal (a bare 10-digit number *is* the question). We only remove control
characters and collapse whitespace so that models see what users typed.
"""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Reference-number / shipment-identifier pattern used by EDA, rules and the conversation layer.
# Matches things like 8006907462, SOU1001357772, 00332009115438339, JBHU315929, BS20260723-1235.
REFERENCE_ID = re.compile(r"\b(?=[A-Za-z0-9#\-]*\d)[A-Z]{0,5}#?\d[\dA-Za-z\-]{4,}\b")


def clean_text(text: object) -> str:
    """Normalise a raw question string.

    - coerces None / NaN / non-str to ""
    - NFKC-normalises unicode (full-width digits, NBSP -> space)
    - strips control characters
    - collapses all whitespace runs (incl. newlines from pasted tables) to one space
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        try:
            if text != text:  # NaN
                return ""
        except Exception:  # pragma: no cover - defensive
            pass
        text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = _CTRL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text


def dedup_key(text: object) -> str:
    """Key used to detect duplicate questions: cleaned, lower-cased, punctuation-light."""
    t = clean_text(text).lower()
    t = re.sub(r"[^\w\s#]", " ", t)
    t = _WS.sub(" ", t).strip()
    return t


def extract_reference_ids(text: str) -> list[str]:
    """Return candidate shipment / order / container identifiers found in the text."""
    return [m.group(0) for m in REFERENCE_ID.finditer(clean_text(text))]


def head_tail_truncate(ids: list[int], max_len: int, tail: int = 0) -> list[int]:
    """Keep the first ``max_len - tail`` and last ``tail`` token ids of an over-long sequence.

    Pasted spreadsheets often put the actual question *after* the table, so keeping a
    short tail preserves it. With ``tail=0`` this is ordinary head truncation.
    """
    if len(ids) <= max_len:
        return ids
    if tail <= 0:
        return ids[:max_len]
    head = max_len - tail
    return ids[:head] + ids[-tail:]
