"""Multi-turn contextual intent resolution for FourSight.

The supplied data is single-turn, so the single-turn classifier is the *measured*
component. This module is the *implemented-but-synthetically-tested* contextual layer
that sits on top of it in production. Design goals:

1. Never blindly inherit the previous intent: context is only consulted when the
   current turn is detectably context-dependent (elliptical, pronoun-only, bare
   reference id, bare time range, confirmation) AND the single-turn classifier is not
   confident on its own.
2. Everything is deterministic: regex detectors + fixed arithmetic over probabilities.
3. State is compact and serialisable (fits in a session store / Redis hash).

Pipeline per turn
-----------------
    text ──► single-turn classifier ──► p_current (7 probs)
         └─► follow-up detectors  ──► FollowUpSignals
    state (previous turns, intents, entities, time-range, topic, age)
    resolver:
        if not follow_up and p_current confident  -> use p_current
        elif follow_up and context fresh           -> blend p_current with context prior,
                                                      inherit entities / time range
        else                                       -> use p_current, flag needs_review
    update state (expire after N turns / minutes; reset on topic switch)

A production extension (documented, not implemented) is to fine-tune the encoder on a
`[CONTEXT] prev turns [CURRENT] text` input once real multi-turn labels exist; the
resolver below is the deterministic baseline it would be compared against.
"""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Sequence

import numpy as np

from foursight_intent.labels import ID2LABEL, LABEL2ID, LABELS
from foursight_intent.text import clean_text, extract_reference_ids

# --------------------------------------------------------------------------- detectors
_ELLIPSIS = re.compile(
    r"^(and|what about|how about|same for|also|now|then|ok(ay)?,?|yes,?|and for|for)\b|"
    r"\b(what about|how about|same (thing|for|query)|instead|as well|too)\b|^(the )?(same|again|other one)s?\b",
    re.I,
)
_PRONOUN_REF = re.compile(r"\b(it|this one|that one|these|those|them|they|the (first|second|last|other) one|the same|both of them|all of them)\b", re.I)
_REF_FILLER = {"please", "status", "track", "tracking", "eta", "check", "load", "loads", "shipment", "shipments", "po", "order", "for", "on", "of", "the", "and", "this", "these", "update", "pls", "thanks", "delivery", "date", "pick", "up", "pickup"}
_TIME_RANGE = re.compile(
    r"^(last|past|previous|this|next|yesterday|today|tomorrow|q[1-4]|ytd|mtd|since|in|for)\b.*|"
    r"^(\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?)(\s*(-|to)\s*\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?)?$|"
    r"^(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b.*",
    re.I,
)
_CONFIRM = re.compile(r"^(yes|yeah|yep|y|no|nope|correct|right|ok|okay|sure|please|both|all|go ahead|do it|continue|proceed|thanks|thank you)[.! ]*$", re.I)
_TOPIC_SWITCH = re.compile(r"\b(different question|new question|unrelated|by the way|btw|switching|another topic|forget that|never ?mind|actually,)\b", re.I)
_GREETING = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|what can you do|help)\b", re.I)
_PAST_TIME = re.compile(r"\b(last|past|previous|yesterday|ago|since|q[1-4]|quarter|ytd|mtd|month|week|year)\b", re.I)
_AGG = re.compile(r"\b(how many|count|number of|total|average|avg|percent|percentage|trend|breakdown|top \d+|most|least|rate|volume)\b", re.I)
_DISRUPT = re.compile(r"\b(weather|storm|hurricane|typhoon|wildfire|flood|congestion|strike|closure|disruption|impact\w*|port condition|border|canal)\b", re.I)


@dataclass
class FollowUpSignals:
    elliptical: bool = False
    pronoun_reference: bool = False
    bare_reference: bool = False
    bare_time_range: bool = False
    confirmation: bool = False
    topic_switch: bool = False
    greeting: bool = False
    n_words: int = 0
    new_reference_ids: list[str] = field(default_factory=list)
    new_time_expression: bool = False

    @property
    def is_follow_up(self) -> bool:
        return (self.elliptical or self.pronoun_reference or self.bare_reference or self.bare_time_range or self.confirmation) and not self.topic_switch

    def to_dict(self) -> dict:
        return asdict(self)


def detect_follow_up(text: str) -> FollowUpSignals:
    t = clean_text(text)
    words = t.split()
    ids = extract_reference_ids(t)
    stripped = re.sub(r"[^\w\s]", " ", t).strip()
    # "bare reference": the utterance is one or more identifiers plus at most two filler words
    residue = re.sub(r"[^\w\s]", " ", t)
    for ref in ids:
        residue = residue.replace(ref, " ")
    residue_words = [w for w in residue.lower().split() if w]
    only_ids = bool(ids) and len(residue_words) <= 2 and all(w in _REF_FILLER for w in residue_words)
    sig = FollowUpSignals(
        elliptical=bool(_ELLIPSIS.search(t)) and len(words) <= 12,
        pronoun_reference=bool(_PRONOUN_REF.search(t)) and len(words) <= 10 and not ids,
        bare_reference=only_ids,
        bare_time_range=bool(_TIME_RANGE.match(stripped)) and len(words) <= 6 and not ids,
        confirmation=bool(_CONFIRM.match(stripped)),
        topic_switch=bool(_TOPIC_SWITCH.search(t)),
        greeting=bool(_GREETING.match(t)) and len(words) <= 6,
        n_words=len(words),
        new_reference_ids=ids,
        new_time_expression=bool(_PAST_TIME.search(t)) or bool(re.search(r"\b(today|tomorrow|now|currently)\b", t, re.I)),
    )
    return sig


# --------------------------------------------------------------------------- state
@dataclass
class Turn:
    text: str
    intent: str
    confidence: float
    probs: list[float]
    ts: float
    used_context: bool = False


@dataclass
class ConversationState:
    """Compact per-session memory. Serialisable with ``to_dict``."""

    session_id: str = ""
    turns: list[Turn] = field(default_factory=list)
    active_reference_ids: list[str] = field(default_factory=list)
    active_topic: str | None = None       # last confidently-resolved intent
    active_time_expression: str | None = None
    turns_since_topic: int = 0
    max_turns: int = 6                    # context window (older turns dropped)
    expiry_seconds: float = 15 * 60       # context expires after inactivity
    topic_decay_turns: int = 3            # after this many unrelated turns the topic expires

    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def context_fresh(self, now: float | None = None) -> bool:
        lt = self.last_turn()
        if lt is None:
            return False
        now = time.time() if now is None else now
        return (now - lt.ts) <= self.expiry_seconds and self.turns_since_topic < self.topic_decay_turns

    def recent_user_turns(self, k: int = 3) -> list[str]:
        return [t.text for t in self.turns[-k:]]

    def context_prior(self) -> np.ndarray | None:
        """Exponentially-decayed average of previous turn probabilities (most recent first)."""
        if not self.turns:
            return None
        weights, acc = 0.0, np.zeros(len(LABELS))
        for i, t in enumerate(reversed(self.turns[-self.max_turns :])):
            w = 0.6**i
            acc += w * np.asarray(t.probs)
            weights += w
        return acc / weights

    def to_dict(self) -> dict:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d

    def reset(self) -> None:
        self.turns.clear()
        self.active_reference_ids.clear()
        self.active_topic = None
        self.active_time_expression = None
        self.turns_since_topic = 0


# --------------------------------------------------------------------------- resolver
@dataclass
class ContextualPrediction:
    intent: str
    confidence: float
    single_turn_intent: str
    single_turn_confidence: float
    used_context: bool
    reason: str
    needs_review: bool
    signals: FollowUpSignals
    resolved_reference_ids: list[str]
    resolved_time_expression: str | None
    probs: dict[str, float]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


class ContextualIntentResolver:
    """Combine a single-turn probability function with conversation state.

    ``single_turn``: callable(text) -> np.ndarray of 7 calibrated probabilities.
    """

    def __init__(
        self,
        single_turn: Callable[[str], np.ndarray],
        confident_threshold: float = 0.70,
        review_threshold: float = 0.50,
        context_weight: float = 0.5,
        min_context_gain: float = 0.05,
    ):
        self.single_turn = single_turn
        self.confident_threshold = confident_threshold
        self.review_threshold = review_threshold
        self.context_weight = context_weight
        self.min_context_gain = min_context_gain

    def resolve(self, text: str, state: ConversationState, now: float | None = None) -> ContextualPrediction:
        now = time.time() if now is None else now
        p_cur = np.asarray(self.single_turn(text), dtype=float)
        st_idx = int(np.argmax(p_cur))
        st_intent, st_conf = ID2LABEL[st_idx], float(p_cur[st_idx])
        sig = detect_follow_up(text)
        prior = state.context_prior() if state.context_fresh(now) else None

        used_context, reason = False, "current-turn semantics"
        p_final = p_cur
        if sig.topic_switch or sig.greeting:
            reason = "explicit topic switch / greeting: context ignored"
            state.turns_since_topic = state.topic_decay_turns  # expire topic
        elif prior is None:
            reason = "no fresh context"
        elif st_conf >= self.confident_threshold and not (sig.confirmation or sig.bare_time_range):
            reason = f"confident single-turn prediction ({st_conf:.2f})"
        elif sig.is_follow_up:
            # Elliptical / bare fragments inherit, but current-turn cues still steer
            # the *kind* of question (e.g. "what about last month" is analytics even
            # after a realtime turn if the previous intent was in the triad).
            w = self.context_weight
            if sig.confirmation or sig.bare_time_range:
                w = max(w, 0.7)  # these fragments carry almost no intent of their own
            p_mix = (1 - w) * p_cur + w * prior
            p_mix = self._apply_cue_shift(p_mix, text, sig, state)
            p_final = p_mix / p_mix.sum()
            used_context = True
            reason = f"follow-up ({self._signal_names(sig)}): blended with context prior (w={w:.2f})"
        else:
            reason = "low confidence but no follow-up signal: current-turn semantics"

        idx = int(np.argmax(p_final))
        intent, conf = ID2LABEL[idx], float(p_final[idx])
        # Entities: new ids override; otherwise a follow-up / pronoun turn inherits the active ids
        # even when the intent itself came from current-turn semantics (e.g. "how do I set alerts for it?").
        inherit = prior is not None and (used_context or sig.is_follow_up or sig.pronoun_reference)
        ref_ids = sig.new_reference_ids or (state.active_reference_ids if inherit else [])
        time_expr = _extract_time_expression(text) or (state.active_time_expression if inherit else None)
        needs_review = conf < self.review_threshold or (used_context and conf < self.confident_threshold and sig.confirmation)
        pred = ContextualPrediction(
            intent=intent, confidence=conf, single_turn_intent=st_intent, single_turn_confidence=st_conf, used_context=used_context,
            reason=reason, needs_review=bool(needs_review), signals=sig, resolved_reference_ids=list(ref_ids),
            resolved_time_expression=time_expr, probs={ID2LABEL[i]: float(p_final[i]) for i in range(len(LABELS))},
        )
        self._update_state(state, text, pred, now)
        return pred

    @staticmethod
    def _signal_names(sig: FollowUpSignals) -> str:
        names = [k for k in ("elliptical", "pronoun_reference", "bare_reference", "bare_time_range", "confirmation") if getattr(sig, k)]
        return ",".join(names) or "none"

    def _apply_cue_shift(self, p: np.ndarray, text: str, sig: FollowUpSignals, state: ConversationState) -> np.ndarray:
        """Small, auditable adjustments within the operational triad based on current-turn cues.

        * a bare past time range or aggregate cue after an operational turn -> analytics
        * a bare reference id after an operational turn -> realtime_query
        * disruption vocabulary -> disruptions
        Shifts move mass *within* the triad only when the context was operational, so a
        knowledge_base conversation is not hijacked by the word "last".
        """
        p = p.copy()
        triad = [LABEL2ID["shipment_information"], LABEL2ID["realtime_query"], LABEL2ID["analytics"]]
        triad_mass = p[triad].sum()
        if triad_mass < 0.4:
            return p
        if _DISRUPT.search(text):
            p[LABEL2ID["disruptions"]] += 0.25
            return p
        if (sig.bare_time_range or _PAST_TIME.search(text)) and (_AGG.search(text) or sig.bare_time_range or state.active_topic == "analytics"):
            shift = 0.5 * triad_mass
            for j in triad:
                p[j] *= 0.5
            p[LABEL2ID["analytics"]] += shift
        elif sig.bare_reference and state.active_topic in ("realtime_query", "shipment_information", None):
            shift = 0.5 * triad_mass
            for j in triad:
                p[j] *= 0.5
            p[LABEL2ID["realtime_query"]] += shift
        return p

    def _update_state(self, state: ConversationState, text: str, pred: ContextualPrediction, now: float) -> None:
        state.turns.append(Turn(text=clean_text(text), intent=pred.intent, confidence=pred.confidence, probs=[pred.probs[l] for l in LABELS], ts=now, used_context=pred.used_context))
        if len(state.turns) > state.max_turns:
            del state.turns[: len(state.turns) - state.max_turns]
        if pred.signals.new_reference_ids:
            state.active_reference_ids = list(pred.signals.new_reference_ids)
        te = _extract_time_expression(text)
        if te:
            state.active_time_expression = te
        if pred.intent in ("chitchat",):
            state.turns_since_topic += 1
            return
        if pred.confidence >= self.confident_threshold or pred.used_context:
            if state.active_topic == pred.intent or state.active_topic is None or not pred.used_context:
                state.active_topic = pred.intent
            state.turns_since_topic = 0
        else:
            state.turns_since_topic += 1


_TIME_EXPR = re.compile(
    r"\b((last|past|previous|this|next)\s+(\d+\s+)?(day|days|week|weeks|month|months|quarter|quarters|year|years|q[1-4])|yesterday|today|tomorrow|ytd|mtd|q[1-4]\s*\d{0,4}|"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)(\s+\d{4})?|\d{1,2}/\d{1,2}(/\d{2,4})?(\s*(-|to)\s*\d{1,2}/\d{1,2}(/\d{2,4})?)?)\b",
    re.I,
)


def _extract_time_expression(text: str) -> str | None:
    m = _TIME_EXPR.search(clean_text(text))
    return m.group(0).lower() if m else None


# --------------------------------------------------------------------------- scenarios
def synthetic_scenarios() -> list[dict]:
    """Hand-written smoke-test conversations. These are NOT labelled production data and
    must not be reported as measured multi-turn accuracy; they exercise the resolver's
    branches (inheritance, intent switch, expiry, entity carry-over)."""
    return [
        {
            "name": "analytics time-range follow-up",
            "turns": [("How many shipments were delayed last week?", "analytics"), ("What about last month?", "analytics"), ("and for Q2?", "analytics")],
        },
        {
            "name": "reference id carry-over then explicit switch",
            "turns": [("what is the status of load 00332009115438339", "realtime_query"), ("where is it now", "realtime_query"), ("how do I set up alerts for it?", "knowledge_base")],
        },
        {
            "name": "bare reference after status question",
            "turns": [("what is the status of load 8006907462", "realtime_query"), ("8006916513", "realtime_query"), ("8006909648 please", "realtime_query")],
        },
        {
            "name": "confirmation inherits topic",
            "turns": [("Can you show me all loads delivering to Portland OR today?", "realtime_query"), ("yes", "realtime_query")],
        },
        {
            "name": "topic switch to disruptions",
            "turns": [("how many loads were late in September?", "analytics"), ("are any of them impacted by the hurricane in the gulf?", "disruptions")],
        },
        {
            "name": "greeting resets context",
            "turns": [("what was my on-time percentage last quarter", "analytics"), ("hi", "chitchat"), ("how do I add a user", "knowledge_base")],
        },
        {
            "name": "knowledge base conversation is not hijacked by 'last'",
            "turns": [("how do I create a filter in my filters", "knowledge_base"), ("and how do I save the last one I made?", "knowledge_base")],
        },
        {
            "name": "aggregate follow-up after status",
            "turns": [("status of PO 91198329", "realtime_query"), ("how many of my loads are late this month?", "analytics")],
        },
    ]
