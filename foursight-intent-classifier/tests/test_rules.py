import numpy as np

from foursight_intent.labels import LABEL2ID, LABELS
from foursight_intent.rules import RULES, apply_rules


def uncertain(top: str, p_top: float = 0.4) -> np.ndarray:
    p = np.full(len(LABELS), (1 - p_top) / (len(LABELS) - 1))
    p[LABEL2ID[top]] = p_top
    return p


def test_rule_boost_is_bounded_so_near_ties_are_not_flipped_blindly():
    """A rule adds at most its boost; a clear (if sub-threshold) preference survives."""
    adj, fired = apply_rules(["8006916513"], np.stack([uncertain("shipment_information", 0.55)]))
    assert "bare_reference" in fired[0]
    assert LABELS[int(adj[0].argmax())] == "shipment_information"


def test_rules_never_override_confident_predictions():
    probs = np.stack([uncertain("realtime_query", 0.95)])
    adj, fired = apply_rules(["hi"], probs, max_conf=0.6)
    assert fired == [[]]
    assert np.array_equal(adj, probs)


def test_greeting_rule_moves_uncertain_row_to_chitchat():
    adj, fired = apply_rules(["hello"], np.stack([uncertain("other")]))
    assert "greeting_or_thanks" in fired[0]
    assert LABELS[int(adj[0].argmax())] == "chitchat"


def test_disruption_rule():
    adj, fired = apply_rules(["what loads could be impacted by tropical storm bertha"], np.stack([uncertain("realtime_query")]))
    assert "disruption_vocabulary" in fired[0]
    assert LABELS[int(adj[0].argmax())] == "disruptions"


def test_past_aggregate_rule_only_from_triad():
    text = "how many loads were late in the last 3 months"
    adj, fired = apply_rules([text], np.stack([uncertain("realtime_query", 0.35)]))
    assert "past_aggregate" in fired[0] and LABELS[int(adj[0].argmax())] == "analytics"
    adj2, fired2 = apply_rules([text], np.stack([uncertain("knowledge_base")]))
    assert "past_aggregate" not in fired2[0]


def test_bare_reference_rule():
    adj, fired = apply_rules(["8006916513"], np.stack([uncertain("shipment_information", 0.3)]))
    assert "bare_reference" in fired[0]
    assert LABELS[int(adj[0].argmax())] == "realtime_query"


def test_probabilities_stay_normalised():
    texts = ["hi", "how many loads were late last month", "8006916513", "how do I add a user"]
    probs = np.stack([uncertain("other"), uncertain("realtime_query"), uncertain("analytics"), uncertain("realtime_query")])
    adj, _ = apply_rules(texts, probs)
    assert np.allclose(adj.sum(1), 1.0)


def test_rules_are_deterministic():
    texts = ["hi", "how many loads were late last month"]
    probs = np.stack([uncertain("other"), uncertain("realtime_query")])
    a, fa = apply_rules(texts, probs)
    b, fb = apply_rules(texts, probs)
    assert np.array_equal(a, b) and fa == fb


def test_rule_registry_is_well_formed():
    for r in RULES:
        assert r.target in LABELS and 0 < r.boost <= 1
