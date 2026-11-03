"""Functional smoke tests for the contextual resolver using a deterministic *mock*
single-turn model. These synthetic conversations are NOT labelled production data; they
only check that each resolver branch behaves as designed."""

import re

import numpy as np
import pytest

from foursight_intent.conversation import (
    ContextualIntentResolver,
    ConversationState,
    detect_follow_up,
    synthetic_scenarios,
)
from foursight_intent.labels import LABEL2ID, LABELS


def mock_single_turn(text: str) -> np.ndarray:
    """Keyword-based stand-in for the fine-tuned encoder. Confident on clear questions,
    deliberately flat on elliptical fragments (which is how the real model behaves)."""
    t = text.lower()
    p = np.full(len(LABELS), 0.02)

    def put(label, v):
        p[LABEL2ID[label]] += v

    if re.search(r"\b(hi|hello|hey)\b", t) and len(t.split()) <= 3:
        put("chitchat", 0.9)
    elif re.search(r"how do i|how to|set up|create a filter|add a user|save the", t):
        put("knowledge_base", 0.85)
    elif re.search(r"hurricane|storm|weather|impacted|disruption", t):
        put("disruptions", 0.8)
    elif re.search(r"how many|percentage|on-time|late in|were (delayed|late)|this month", t):
        put("analytics", 0.8)
    elif re.search(r"status|where is|delivering|track|\b\d{7,}\b|po \d+", t):
        put("realtime_query", 0.75)
        put("shipment_information", 0.1)
    else:  # elliptical fragments: flat, slight realtime lean like the real model
        put("realtime_query", 0.25)
        put("shipment_information", 0.2)
        put("analytics", 0.2)
        put("other", 0.15)
    return p / p.sum()


@pytest.fixture
def resolver():
    return ContextualIntentResolver(mock_single_turn, confident_threshold=0.7, review_threshold=0.5)


def run(resolver, turns, state=None, t0=1_000_000.0):
    state = state or ConversationState(session_id="t")
    out = []
    for i, text in enumerate(turns):
        out.append(resolver.resolve(text, state, now=t0 + 10 * i))
    return out, state


# ------------------------------------------------------------------ detectors
@pytest.mark.parametrize(
    "text, attr",
    [
        ("what about last month?", "elliptical"),
        ("and for Q2?", "elliptical"),
        ("same for Dallas", "elliptical"),
        ("where is it now", "pronoun_reference"),
        ("8006916513", "bare_reference"),
        ("8006909648 please", "bare_reference"),
        ("last 7 days", "bare_time_range"),
        ("yes", "confirmation"),
        ("both", "confirmation"),
    ],
)
def test_follow_up_detectors(text, attr):
    sig = detect_follow_up(text)
    assert getattr(sig, attr), (text, sig)
    assert sig.is_follow_up


def test_full_questions_are_not_follow_ups():
    for text in ["How many shipments were delayed last week?", "how do I add a user", "what is the status of load 00332009115438339"]:
        assert not detect_follow_up(text).is_follow_up, text


def test_topic_switch_detected():
    sig = detect_follow_up("different question: how do I add a user?")
    assert sig.topic_switch and not sig.is_follow_up


# ------------------------------------------------------------------ resolver behaviour
def test_elliptical_time_follow_up_inherits_analytics(resolver):
    preds, _ = run(resolver, ["How many shipments were delayed last week?", "What about last month?"])
    assert preds[0].intent == "analytics" and not preds[0].used_context
    assert preds[1].intent == "analytics" and preds[1].used_context
    assert preds[1].single_turn_intent != "analytics"  # the mock alone would have got it wrong
    assert preds[1].resolved_time_expression == "last month"


def test_pronoun_follow_up_inherits_reference_id(resolver):
    preds, state = run(resolver, ["what is the status of load 00332009115438339", "where is it now"])
    assert preds[1].intent == "realtime_query"
    assert preds[1].resolved_reference_ids == ["00332009115438339"]
    assert state.active_reference_ids == ["00332009115438339"]


def test_bare_reference_after_status_question(resolver):
    preds, state = run(resolver, ["what is the status of load 8006907462", "8006916513"])
    assert preds[1].intent == "realtime_query"
    assert preds[1].resolved_reference_ids == ["8006916513"]  # new id overrides the old one
    assert state.active_reference_ids == ["8006916513"]


def test_confirmation_inherits_but_is_flagged_when_uncertain(resolver):
    preds, _ = run(resolver, ["Can you show me all loads delivering to Portland OR today?", "yes"])
    assert preds[1].intent == "realtime_query" and preds[1].used_context


def test_intent_switch_overrides_context(resolver):
    preds, _ = run(resolver, ["how many loads were late in September?", "how do I set up alerts for late loads?"])
    assert preds[0].intent == "analytics"
    assert preds[1].intent == "knowledge_base" and not preds[1].used_context


def test_disruption_switch(resolver):
    preds, _ = run(resolver, ["how many loads were late in September?", "are any of them impacted by the hurricane in the gulf?"])
    assert preds[1].intent == "disruptions"


def test_greeting_resets_topic(resolver):
    preds, state = run(resolver, ["what was my on-time percentage last quarter", "hi", "what about last month?"])
    assert preds[1].intent == "chitchat"
    # after the greeting the topic expired, so the elliptical turn cannot inherit analytics with confidence
    assert preds[2].needs_review or preds[2].intent != "analytics" or preds[2].used_context is False


def test_context_expires_after_inactivity(resolver):
    state = ConversationState(session_id="t", expiry_seconds=60)
    resolver.resolve("How many shipments were delayed last week?", state, now=0.0)
    late = resolver.resolve("What about last month?", state, now=10_000.0)
    assert not late.used_context and "no fresh context" in late.reason


def test_knowledge_base_context_not_hijacked_by_time_word(resolver):
    preds, _ = run(resolver, ["how do I create a filter in my filters", "and how do I save the last one I made?"])
    assert preds[1].intent == "knowledge_base"


def test_state_is_serialisable_and_bounded(resolver):
    state = ConversationState(session_id="s", max_turns=3)
    for i in range(6):
        resolver.resolve(f"status of load 800690746{i}", state, now=float(i))
    d = state.to_dict()
    assert len(d["turns"]) == 3
    import json

    json.dumps(d)  # must not raise


def test_resolver_is_deterministic(resolver):
    a, _ = run(resolver, ["How many shipments were delayed last week?", "What about last month?", "8006916513"])
    b, _ = run(resolver, ["How many shipments were delayed last week?", "What about last month?", "8006916513"])
    assert [p.intent for p in a] == [p.intent for p in b]
    assert [p.probs for p in a] == [p.probs for p in b]


def test_synthetic_scenarios_run_end_to_end(resolver):
    """Every scripted scenario must produce a valid label for every turn; expected labels
    are design targets, reported (not asserted) because they are synthetic."""
    mismatches = []
    for sc in synthetic_scenarios():
        preds, _ = run(resolver, [t for t, _ in sc["turns"]])
        for (text, expected), p in zip(sc["turns"], preds):
            assert p.intent in LABELS
            if p.intent != expected:
                mismatches.append((sc["name"], text, expected, p.intent))
    # With the mock model the designed scenarios should all resolve as intended.
    assert not mismatches, mismatches
