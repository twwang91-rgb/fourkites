# Handling previously unseen intents in production

## Two different problems

| situation | what it is | correct response |
|---|---|---|
| **Out-of-scope question** (existing `other` class) | A question FourSight cannot answer with any current product: "draft an email to the carrier", "freight quote from Ontario to Germantown" | predict `other`; route to the fallback response; nothing new is learned |
| **Emerging intent** (new routing destination) | A *cluster* of recurring questions that today land in `other` or are mis-routed, for which a product/backend now exists or is planned (e.g. appointment scheduling, carbon reporting) | discover → decide → version the taxonomy → retrain → regression-test → roll out |

The classifier must never invent an eighth label at inference time: the seven-label set is
a contract with downstream routing. Discovery happens offline.

## Detection signals (per prediction, already produced by the system)

1. **Calibrated confidence and margin** — temperature-scaled max probability below the
   validation-chosen `needs_review` threshold, or small top-1/top-2 margin.
2. **`other` probability** — a rising share of `other` predictions is the first symptom of
   a new intent.
3. **Out-of-distribution score** — cosine distance of the question's encoder embedding to
   its nearest training neighbours (k-NN on the [CLS] / mean-pooled embedding). High
   distance + low confidence ⇒ "novel", not merely "ambiguous". This is a monitoring
   signal, not a routing input, so it needs no label.
4. **Downstream failure** — the routed backend returned "no result" / the user rephrased
   or escalated within the same session.

## Discovery loop (weekly)

1. Pull flagged questions (any signal above) and embed them with the deployed encoder.
2. Cluster (HDBSCAN or agglomerative on cosine distance), surface clusters by size and
   growth; show 10 representative questions + current predicted labels per cluster.
3. A product owner decides: (a) noise, (b) existing intent (→ add the examples to
   training to fix a recall gap), or (c) **new intent candidate**.
4. For (c): write the definition, annotate ≥100 examples (and re-annotate ambiguous
   neighbours of the existing classes it borders), check inter-annotator agreement.

## Adding the class

1. **Version the taxonomy**: `labels.json` gets `taxonomy_version: v2` with the new label
   appended (never re-ordered — ids are stable); the model's `label_map.json` and
   `inference_config.json` carry the version; routing config maps `v2` labels to backends.
2. **Retrain** with the same seeded, grouped-CV pipeline; the OOF report now includes the
   new class. Acceptance: no existing class loses more than 1 F1 point on the frozen
   regression set, and the new class reaches the agreed minimum F1.
3. **Regression suite**: the frozen validation set plus a curated "golden" file of
   boundary cases (triad, rare classes, prior incidents) run in CI; identical predictions
   are required for the golden file unless the change is explicitly approved.
4. **Rollout**: shadow mode (new model logs, old model routes) → canary 5% of sessions →
   full. Monitor per-class volume, confidence distribution, review rate, downstream
   "no result" rate and user re-asks. Automatic rollback if the review rate or
   mis-route complaints exceed the baseline by a set margin.
5. **Downstream routing** is updated atomically with the model version: a `v2` model is
   only served by a router that knows the `v2` label set; the router rejects unknown
   labels (fail closed to `other`).

## Monitoring dashboards

* prediction volume per class and per taxonomy version
* calibrated confidence histogram and review rate (drift of the median is an alarm)
* OOD distance distribution, with weekly cluster report
* confusion between the operational triad classes estimated from reviewed samples
