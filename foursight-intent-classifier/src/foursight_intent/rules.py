"""Minimal, auditable domain rules evaluated as a *post-processing* layer.

Rules only act when the model is uncertain (max calibrated probability below
``max_conf``) and only move probability mass between explicitly named classes, so a
confident model prediction is never overridden. Each rule is a dataclass with a name,
a regex, a target label and a bounded boost; ``apply_rules`` returns adjusted
probabilities plus a per-row audit trail. Whether the layer is kept is decided by
out-of-fold validation evidence (scripts/analyze_results.py), never by assumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from foursight_intent.labels import LABEL2ID, LABELS
from foursight_intent.text import clean_text, extract_reference_ids


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    target: str
    boost: float
    only_if_top_in: tuple[str, ...] = ()   # fire only when the current top class is one of these
    description: str = ""


RULES: list[Rule] = [
    Rule(
        "greeting_or_thanks",
        re.compile(r"^(hi|hello|hey|hola|good (morning|afternoon|evening)|thanks?|thank you|that('s| is) (perfect|great|awesome)|what can you do|what (questions )?can i ask( you)?)\b[\s!.?]*$", re.I),
        "chitchat", 0.6, (), "pure greetings / thanks with nothing else",
    ),
    Rule(
        "disruption_vocabulary",
        re.compile(r"\b(tropical (storm|depression)|hurricane|typhoon|wildfire|flood\w*|port congestion|congestion at|strike|border (closure|wait)|canal|disruption\w*|impacted by|weather(-| )related)\b", re.I),
        "disruptions", 0.35, ("shipment_information", "realtime_query", "analytics", "other"), "explicit disruption vocabulary",
    ),
    Rule(
        "past_aggregate",
        re.compile(r"\b(how many|count of|number of|percent\w*|%|average|avg|trend\w*|breakdown|on[- ]time (performance|percentage|rate)|otd|otp)\b.*\b(last|past|previous)\s+(\d+\s+)?(day|week|month|quarter|year)s?\b|\b(last|past|previous)\s+(\d+\s+)?(day|week|month|quarter|year)s?\b.*\b(how many|count|percent\w*|average|avg|trend\w*|breakdown|on[- ]time)\b", re.I),
        "analytics", 0.3, ("shipment_information", "realtime_query"), "aggregate measure over a past period",
    ),
    Rule(
        "how_to_product",
        re.compile(r"^(how (do|can|should|would) (i|we|you)|how to|where (do|can) (i|we)|is there a way to|can (i|we|you) (set ?up|configure|create|add|remove|delete|change|edit|export|download|filter))\b", re.I),
        "knowledge_base", 0.25, ("shipment_information", "realtime_query", "analytics", "other"), "procedural product question",
    ),
]


def _bare_reference_rule(text: str) -> tuple[str, float] | None:
    """A message consisting only of identifiers is a live lookup unless the model already
    prefers shipment_information with margin; returns (target, boost) or None."""
    ids = extract_reference_ids(text)
    if not ids:
        return None
    residue = text
    for r in ids:
        residue = residue.replace(r, " ")
    if len(re.sub(r"[^\w]", "", residue)) <= 8:
        return "realtime_query", 0.2
    return None


def apply_rules(texts: list[str], probs: np.ndarray, max_conf: float = 0.6, rules: list[Rule] | None = None, use_bare_reference: bool = True) -> tuple[np.ndarray, list[list[str]]]:
    """Return (adjusted_probs, fired_rule_names_per_row).

    Only rows whose max probability is below ``max_conf`` are eligible. A fired rule adds
    ``boost`` to its target then renormalises; several rules may fire on one row."""
    rules = RULES if rules is None else rules
    out = probs.copy()
    fired: list[list[str]] = [[] for _ in texts]
    for i, raw in enumerate(texts):
        if out[i].max() >= max_conf:
            continue
        text = clean_text(raw)
        top = LABELS[int(out[i].argmax())]
        for rule in rules:
            if rule.only_if_top_in and top not in rule.only_if_top_in:
                continue
            if rule.pattern.search(text):
                out[i, LABEL2ID[rule.target]] += rule.boost
                fired[i].append(rule.name)
        if use_bare_reference:
            hit = _bare_reference_rule(text)
            if hit and top in ("shipment_information", "realtime_query", "analytics", "other"):
                out[i, LABEL2ID[hit[0]]] += hit[1]
                fired[i].append("bare_reference")
        if fired[i]:
            out[i] = out[i] / out[i].sum()
    return out, fired
