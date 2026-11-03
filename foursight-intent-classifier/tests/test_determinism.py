"""Determinism and robustness of the saved artefact. Skipped until models/final exists."""

import numpy as np
import pytest

from foursight_intent.labels import LABELS

SAMPLES = [
    "where is my shipment 8006907462",
    "how many loads were late last month?",
    "how do I add a user to my account",
    "hi",
    "what loads could be impacted by the tropical storm in the gulf?",
    "can you draft an email to the carrier for me",
    "what is the status of load 00332009115438339",
]


def test_model_loads_with_label_order(classifier):
    assert classifier.label_order() == LABELS


def test_repeated_predictions_identical(classifier):
    p1 = classifier.probabilities(SAMPLES)
    p2 = classifier.probabilities(SAMPLES)
    assert np.array_equal(p1, p2), "same input must give bit-identical probabilities"
    assert [p.intent for p in classifier.predict(SAMPLES)] == [p.intent for p in classifier.predict(SAMPLES)]


def test_prediction_independent_of_batch_composition(classifier):
    single = [classifier.predict_one(s).intent for s in SAMPLES]
    batched = [p.intent for p in classifier.predict(SAMPLES)]
    shuffled_in = list(reversed(SAMPLES))
    shuffled = [p.intent for p in classifier.predict(shuffled_in)][::-1]
    assert single == batched == shuffled


def test_fresh_load_gives_same_result(model_dir, classifier):
    from foursight_intent.inference import load_classifier

    other = load_classifier(model_dir, device="cpu", num_threads=2)
    assert np.array_equal(other.probabilities(SAMPLES), classifier.probabilities(SAMPLES))


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t", None, float("nan"), "a" * 20_000, "where is my load " * 600, "いつ倉庫に納品される？", "¿dónde está mi envío?", "🚚📦 where is it", "\x00\x01status", 123456789],
)
def test_malformed_and_edge_inputs_return_valid_label(classifier, text):
    pred = classifier.predict_one(text)
    assert pred.intent in LABELS
    assert 0.0 <= pred.confidence <= 1.0
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-5


def test_every_training_class_is_reachable(classifier):
    """Sanity: the model can emit each of the seven labels on obvious prompts."""
    prompts = {
        "realtime_query": "what is the status of load 00332009115438339",
        "knowledge_base": "how do I add a user to the platform",
        "chitchat": "hello",
        "analytics": "what was my on time delivery percentage last month",
        "disruptions": "which loads are impacted by tropical storm bertha",
    }
    for label, text in prompts.items():
        assert classifier.predict_one(text).intent == label, (label, text)


def test_needs_review_flag_is_metadata_only(classifier):
    preds = classifier.predict(["yes", "both", "2"])
    for p in preds:
        assert p.intent in LABELS  # a label is always emitted even when flagged
        assert isinstance(p.needs_review, bool)
