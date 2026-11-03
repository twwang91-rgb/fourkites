# Multi-turn intent classification architecture

The training data is single-turn. In production FourSight is a chat, and the intent of a
turn such as *"what about last month?"* exists only relative to the previous turn. This
document specifies how the single-turn classifier is embedded in a contextual system,
what is implemented in this repository, and what is proposed for production.

## 1. Components

```
                ┌──────────────────────────────────────────────────────────────┐
 user turn ───► │ 1. Single-turn encoder (DistilBERT/DeBERTa, fine-tuned)      │──► p_current (7 probs, calibrated)
                └──────────────────────────────────────────────────────────────┘
                ┌──────────────────────────────────────────────────────────────┐
 user turn ───► │ 2. Follow-up detectors (deterministic regex / length rules)  │──► signals: elliptical, pronoun_reference,
                └──────────────────────────────────────────────────────────────┘    bare_reference, bare_time_range,
                                                                                    confirmation, topic_switch, greeting
 session ─────► ┌──────────────────────────────────────────────────────────────┐
 store          │ 3. ConversationState (compact, serialisable)                 │
                │    • last ≤6 user turns, their intents, confidences, probs   │
                │    • active_reference_ids   (shipment / PO / container ids)  │
                │    • active_topic           (last confidently resolved intent)│
                │    • active_time_expression (e.g. "last week")               │
                │    • turns_since_topic, timestamps  → freshness / expiry     │
                └──────────────────────────────────────────────────────────────┘
                ┌──────────────────────────────────────────────────────────────┐
                │ 4. ContextualIntentResolver                                  │
                │    decides *whether* and *how much* to use context           │
                └──────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
                intent, confidence, needs_review, resolved ids / time range, reason
```

Implemented in `src/foursight_intent/conversation.py`; exercised by
`tests/test_conversation.py` and `python scripts/predict.py --conversation`.

## 2. Resolver decision procedure (implemented)

For each turn, with `p_cur` from the single-turn model and `prior` = exponentially
decayed average of the previous turns' probability vectors (weights 1, 0.6, 0.36, …):

| condition | action | rationale |
|---|---|---|
| greeting or explicit topic switch ("different question", "never mind") | use `p_cur`, expire topic | never let a stale topic hijack a reset |
| no fresh context (first turn, >15 min idle, or topic decayed over 3 unrelated turns) | use `p_cur` | context expiration |
| `max(p_cur) ≥ 0.70` and the turn is not a pure confirmation / bare time range | use `p_cur` | **current-turn semantics win** when the model is sure; this is what prevents blind inheritance ("how do I set alerts for it?" after a tracking question is knowledge_base) |
| turn is a detected follow-up (elliptical, pronoun, bare id, bare time range, confirmation) | `p = (1-w)·p_cur + w·prior`, w = 0.5 (0.7 for confirmations / bare time ranges), then bounded cue shifts within the operational triad | fragments carry little intent of their own; cues such as a past time range or aggregate word still steer realtime→analytics |
| low confidence, no follow-up signal | use `p_cur`, likely `needs_review` | an unrelated but unclear question should not inherit anything |

Entity resolution: new reference ids in the turn always override; otherwise a follow-up
inherits `active_reference_ids` and `active_time_expression` (so "where is it now?"
carries the load id to the realtime backend). Both are returned with the prediction so
the downstream product can execute the query.

Ambiguity classes handled: elliptical follow-ups ("what about last month?"), pronoun
references ("where is it now"), inherited shipment ids ("8006916513" after a status
question), inherited analysis windows ("and for Dallas?" keeps "last week"), intent
switches (confident current-turn semantics), topic changes (explicit markers, greetings,
decay), context expiration (time- and turn-based).

## 3. Contextual input representation (proposed, not trained)

Once real multi-turn labels exist, the encoder itself should see context:

```
[CONTEXT] prev_intent=analytics ; prev_turn="How many shipments were delayed last week?"
[CURRENT] What about last month?
```

* previous *user* turns only (assistant text is long and templated), most recent first,
  truncated to ~96 tokens of context + 128 of current;
* the previous predicted intent is injected as a text token so the model can learn
  "inherit unless contradicted" instead of hard-coding it;
* training data: (a) real labelled conversations, (b) synthetic augmentation that pairs a
  labelled single-turn question with a context-free rewrite ("what about last month?"),
  (c) *context-independent* pairs where the current turn is unrelated, so the model learns
  to ignore context — otherwise it learns to copy `prev_intent`.

The deterministic resolver above is the baseline this model must beat on the real
conversational benchmark; it also remains as the fallback when the session store is
unavailable.

## 4. Production safeguards

* The single-turn label is always computed and logged alongside the contextual label,
  so inheritance can be audited and its lift measured.
* Context is used only when the turn is *detectably* context-dependent **and** the model
  is not confident; a confident current-turn prediction is never overridden.
* Blended confidence below the review threshold raises `needs_review`; confirmations
  resolved purely from context are flagged when blended confidence is below 0.70.
* State is bounded (6 turns), expires (15 min), and decays (3 unrelated turns); it holds
  no PII beyond the user's own utterances and is keyed by session id.
* Any change to the resolver thresholds is covered by the scenario test-suite.

## 5. Evaluation methodology

The provided dataset has no conversation histories, so **no multi-turn accuracy is
claimed**. `synthetic_scenarios()` in `conversation.py` defines eight scripted
conversations used purely as a functional smoke test (they are run in
`tests/test_conversation.py` with a mock single-turn model so the resolver logic is tested
independently of model quality, and with the real model in `scripts/predict.py --conversation`).

Before deployment a real benchmark would be built as follows:

1. Sample ~2,000 production sessions stratified by length (2–8 turns) and by first-turn
   intent; anonymise as in the single-turn data.
2. Annotate every turn with the intent **given the history** (annotators see the whole
   conversation), with a second label for "context required? (yes/no)" and the resolved
   entities / time range. Double-annotate 20% to measure agreement (target κ ≥ 0.8 on
   intent; the single-turn data's 37 conflicting duplicate groups show why this matters).
3. Metrics: OVERALL (same formula) on all turns; OVERALL on the *context-required* subset
   (the quantity the resolver is supposed to improve); inheritance error rate on the
   context-*not*-required subset (the quantity blind inheritance would damage); entity
   resolution exact-match; review-flag precision/recall.
4. Compare: single-turn model alone → deterministic resolver → context-aware encoder.
   Ship the simplest variant whose gain on the context-required subset is significant
   (paired bootstrap over sessions) without regressing the context-not-required subset.
5. Online: shadow-mode the contextual layer, log disagreements with the single-turn
   label, sample them for weekly review; promote when the reviewed precision of the
   context-driven changes exceeds the single-turn baseline.
