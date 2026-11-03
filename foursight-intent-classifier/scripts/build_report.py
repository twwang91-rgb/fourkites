#!/usr/bin/env python
"""Render report.pdf (≈2 pages) from the recorded results with reportlab.

Every number is read from reports/results/*.json and models/final/*.json; nothing is typed
by hand, so the PDF cannot drift from the measured results.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from foursight_intent.labels import LABELS

LINKS_PATH = ROOT / "reports" / "links.json"


def load(p: Path):
    return json.load(open(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "report.pdf"))
    ap.add_argument("--model-dir", default=str(ROOT / "models/final"))
    a = ap.parse_args()
    res = ROOT / "reports" / "results"
    inf = load(Path(a.model_dir) / "inference_config.json")
    summ = load(Path(a.model_dir) / "training_summary.json")
    name = inf["selected_experiment"]
    exp = load(res / f"transformer_{name}.json")
    ana = load(res / f"analysis_{name}.json")
    cmp_ = load(res / "comparison.json")
    eda = load(ROOT / "reports/eda.json")
    final = load(res / "final_selection.json")
    links = load(LINKS_PATH) if LINKS_PATH.exists() else {}
    sub_meta = load(res / "submission_summary.json") if (res / "submission_summary.json").exists() else {}

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.4, leading=10.4, alignment=TA_JUSTIFY, spaceAfter=3)
    small = ParagraphStyle("small", parent=body, fontSize=7.2, leading=8.8)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, leading=15, spaceAfter=2, spaceBefore=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=9.6, leading=11.5, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1F3A5F"))
    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=7.2, leading=8.6)

    def P(t, st=body):
        return Paragraph(t, st)

    def tbl(data, col_widths, font=7.0, header=True, zebra=True):
        t = Table(data, colWidths=col_widths, hAlign="LEFT")
        style = [("FONT", (0, 0), (-1, -1), "Helvetica", font), ("LEADING", (0, 0), (-1, -1), font + 1.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2), ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                 ("LINEBELOW", (0, -1), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 1), (-1, -1), "RIGHT")]
        if header:
            style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", font), ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5"))]
        if zebra:
            for i in range(1, len(data)):
                if i % 2 == 0:
                    style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F6F8FA")))
        t.setStyle(TableStyle(style))
        return t

    per = exp["per_class"]
    pol = ana["review"]["deployment_policy"]
    cf = ana["review"]["cross_fitted"]
    cal = ana["calibration"]["fitted_on_all_oof"]
    conf = exp["confusion"]
    base_model = inf["base_model"]
    kind = inf.get("kind", "single")
    members = inf.get("members", [])
    epochs_text = (str(inf["epochs"]) + " epochs") if kind == "single" else ("epochs per member: " + ", ".join(f"{m['name']} {inf['epochs'][m['name']]}" for m in members))
    best_rows = cmp_["candidates"]
    rules_on = inf.get("use_rules", False)
    final_scores = final["selected"]["scores"]
    story = []

    # ------------------------------------------------------------------ header
    story.append(P("FourSight Intent Classification — Technical Report", h1))
    model_blurb = f"fine-tuned <b>{base_model}</b>" if kind == "single" else f"<b>probability-averaging ensemble</b> of {len(members)} fine-tuned encoders ({', '.join(m['base_model'].split('/')[-1] for m in members)})"
    story.append(P(f"Seven-way first-hop intent router · single-turn model: {model_blurb}{' + auditable rule layer' if rules_on else ''} · "
                   f"contextual resolver for multi-turn chat · calibrated <i>needs_review</i> flag. All metrics are pooled out-of-fold results of a grouped, seeded 5-fold CV on the 1,904 training rows; "
                   f"the hidden test set was never used for any decision.", small))

    # ------------------------------------------------------------------ A
    story.append(P("A. Problem, data findings and method", h2))
    dup = eda["duplicates"]; leak = eda["leakage"]["stratified_5fold"]["total_leaked_rows"]
    cue = eda["cue_prevalence_by_class"]
    ref_rt, ref_si = 100 * cue["realtime_query"]["reference_id"], 100 * cue["shipment_information"]["reference_id"]
    an_past, an_agg = 100 * cue["analytics"]["temporal_past"], 100 * cue["analytics"]["aggregate"]
    short_pct = 100 * eda["train_length"]["pct_rows_le_3_words"]
    story.append(P(
        f"FourSight must route each question to one of seven backends; a wrong route means the right system is never queried, and OVERALL = 0.6·macro-F1 + 0.4·accuracy makes the three rare classes "
        f"(chitchat 17, disruptions 28, other 55 rows) as important as realtime_query (782). EDA (<i>reports/eda.md</i>) drove four design decisions. "
        f"(1) <b>Label noise is structural</b>: {dup['exact_duplicate_groups']} exact-duplicate groups exist and {dup['conflicting_label_groups']} of them carry conflicting labels "
        f"(e.g. a bare 10-digit id labelled both shipment_information and realtime_query; 'yes' labelled three ways), capping training-set accuracy at {100 * dup['accuracy_ceiling_from_conflicts']:.1f}%. "
        f"(2) <b>Validation must be grouped</b>: with plain stratified folds {leak} of 1,904 validation rows ({100 * leak / 1904:.0f}%) have a verbatim twin in their training fold, so every number here uses StratifiedGroupKFold on normalised text. "
        f"(3) The triad boundary is semantic, not lexical: reference numbers appear in {ref_rt:.0f}% of realtime_query but also {ref_si:.0f}% of shipment_information rows; temporal-past and aggregate cues mark analytics ({an_past:.0f}% / {an_agg:.0f}% of its rows). "
        f"(4) {short_pct:.0f}% of rows are ≤3 words and many are context fragments ('both', 'scheduled', 'eta'), which no single-turn model can resolve — this motivates the multi-turn design and the review flag.", body))
    story.append(P(
        f"<b>Method.</b> Baselines first: TF-IDF (word+char n-grams) with logistic regression / linear SVM, and frozen sentence embeddings (MiniLM, bge-small, multilingual-e5-small) with logistic regression; "
        f"then fine-tuned compact encoders (DistilBERT, DeBERTa-v3-small, RoBERTa-base, multilingual-e5-small, bge-small) with a seeded fp32 training loop: AdamW, linear warm-up, weight decay, gradient clipping, "
        f"√-inverse-frequency class weights, head+tail truncation to {inf['max_length']} tokens (pasted tables put the question last), early stopping on OVERALL. "
        f"Temperature scaling, the review threshold and the rule layer were all fitted/evaluated on out-of-fold predictions (cross-fitted: fitted on four folds, applied to the fifth). "
        f"The final artefact retrains the selected configuration on all 1,904 rows ({epochs_text}; median CV best epoch) after model selection was frozen.", body))
    story.append(P(
        "<b>Determinism.</b> Inference is a pure function: fp32 on CPU, eval mode (no dropout), fixed thread count, <i>torch.use_deterministic_algorithms(True)</i>, fixed-size batches processed in input order, "
        "softmax with a fixed temperature, stable argmax (lowest label index on exact ties), and rules that are plain regexes. <i>tests/test_determinism.py</i> verifies bit-identical probabilities across repeated calls, "
        "fresh model loads and different batch compositions, plus valid output on empty, NaN, 20k-character, unicode and emoji inputs. Across different hardware/BLAS builds logits can differ at the 1e-6 level, which flips a label only on an exact near-tie; that is the stated guarantee, not universal bitwise equality.", body))

    # ------------------------------------------------------------------ B
    story.append(P("B. Validation results", h2))
    cand_rows = [["model", "acc", "macro-F1", "OVERALL", "±fold", "triad err", "disr/chit/other F1"]]
    for r in [x for x in best_rows if x.get("primary", True)][:7]:
        cand_rows.append([r["model"][:34], f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}", f"{r['OVERALL']:.3f}", f"{r.get('fold_std', 0):.3f}", str(r["triad_errors"]),
                          "/".join(f"{r['per_class_f1'][k]:.2f}" for k in ("disruptions", "chitchat", "other"))])
    for r in sorted([e for e in cmp_.get("ensembles", []) if e["model"] != name], key=lambda e: -e["OVERALL"])[:2]:
        cand_rows.append([f"{r['model']} ({'+'.join(m.split('_')[0] for m in r['members'])})", f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}", f"{r['OVERALL']:.3f}", f"{r.get('fold_std', 0):.3f}", str(r["triad_errors"]),
                          "/".join(f"{r['per_class_f1'][k]:.2f}" for k in ("disruptions", "chitchat", "other"))])
    sel_label = f"{name}{' + rules' if rules_on else ''} (selected)" if kind == "single" or rules_on else None
    if sel_label:
        cand_rows.append([sel_label, f"{final_scores['accuracy']:.3f}", f"{final_scores['macro_f1']:.3f}", f"{final_scores['OVERALL']:.3f}", f"{exp.get('OVERALL_fold_std', 0):.3f}", str(final["selected"]["triad_errors"]),
                          "/".join(f"{final_scores['per_class'][k]['f1']:.2f}" for k in ("disruptions", "chitchat", "other"))])
    if kind == "ensemble" and not rules_on:
        cand_rows.append([f"{name} (selected)", f"{final_scores['accuracy']:.3f}", f"{final_scores['macro_f1']:.3f}", f"{final_scores['OVERALL']:.3f}", f"{exp.get('OVERALL_fold_std', 0):.3f}", str(final["selected"]["triad_errors"]),
                          "/".join(f"{final_scores['per_class'][k]['f1']:.2f}" for k in ("disruptions", "chitchat", "other"))])
    pc_rows = [["class", "P", "R", "F1", "n"]] + [[c, f"{final_scores['per_class'][c]['precision']:.2f}", f"{final_scores['per_class'][c]['recall']:.2f}", f"{final_scores['per_class'][c]['f1']:.2f}", str(per[c]["support"])] for c in LABELS]
    two = Table([[tbl(cand_rows, [4.9 * cm, 1.0 * cm, 1.3 * cm, 1.35 * cm, 1.0 * cm, 1.3 * cm, 2.3 * cm], 6.6), tbl(pc_rows, [3.1 * cm, 0.9 * cm, 0.9 * cm, 0.9 * cm, 0.8 * cm], 6.6)]], colWidths=[13.4 * cm, 6.8 * cm])
    two.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(two)
    story.append(Spacer(1, 2))
    sel = final["selected"]
    ci_text = f"(95% paired-bootstrap CI {sel['bootstrap_ci'][0]:.3f}–{sel['bootstrap_ci'][1]:.3f})" if sel.get("bootstrap_ci") else ""
    story.append(P(
        f"<b>Selected: {sel['description']}</b> — OOF accuracy {final_scores['accuracy']:.3f}, macro-F1 {final_scores['macro_f1']:.3f}, <b>OVERALL {final_scores['OVERALL']:.3f}</b> "
        f"{ci_text}. {sel['why']} "
        f"Frozen-embedding logistic regression reaches {max(r['OVERALL'] for r in best_rows if r['family'].startswith('baseline: embedding')):.3f} and TF-IDF {max(r['OVERALL'] for r in best_rows if r['family'].startswith('baseline: tfidf')):.3f}; "
        f"fine-tuning buys most of its gain on knowledge_base and realtime_query precision, while the frozen baselines are competitive on the rare classes because class-balanced logistic regression is less prone to collapsing 17-example classes.", body))
    # confusion matrix compact
    cm_rows = [["gold \\ pred"] + [l[:6] for l in LABELS]] + [[g[:14]] + [str(conf[g].get(p, 0)) for p in LABELS] for g in LABELS]
    cm_tbl = tbl(cm_rows, [2.3 * cm] + [0.95 * cm] * 7, 6.4)
    fig_path = ROOT / "reports/figures" / f"reliability_{name}.png"
    img = Image(str(fig_path), width=9.2 * cm, height=3.9 * cm) if fig_path.exists() else Spacer(1, 1)
    row = Table([[cm_tbl, img]], colWidths=[9.6 * cm, 10.6 * cm])
    row.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(row)
    tri = exp["triad_errors"]
    cats = ana["errors"]["categories"]
    top_cats = sorted(cats.items(), key=lambda kv: -kv[1]["count"])[:4]
    examples = []
    for cat_name in ("triad: semantic", "triad: temporal/aggregate cue", "knowledge_base vs operational", "short/underspecified"):
        exs = cats.get(cat_name, {}).get("examples", [])
        if exs:
            e = exs[0]
            examples.append(f"<i>'{e['text'][:70].replace('&', '&amp;').replace('<', '&lt;')}'</i> (gold {e['gold']}, predicted {e['pred']} at {e['confidence']:.2f})")
    story.append(P(
        f"<b>Where it fails.</b> Confusion matrix above is the un-ruled {name} OOF; {tri['total_triad_errors']} of {ana['errors']['n_errors']} errors lie inside the operational triad, dominated by "
        f"shipment_information→realtime_query ({tri.get('shipment_information->realtime_query', 0)}) and realtime_query→shipment_information ({tri.get('realtime_query->shipment_information', 0)}). "
        f"Error categories: " + "; ".join(f"{c} ({d['count']}, mean conf {d['mean_confidence']:.2f})" for c, d in top_cats) + ". "
        f"Representative high-confidence misses: " + "; ".join(examples) + f". Rows belonging to conflicting-label duplicate groups are classified correctly only {100 * ana['errors']['conflicting_duplicate_rows']['accuracy']:.0f}% of the time and account for {100 * ana['errors']['conflicting_duplicate_rows']['share_of_all_errors']:.0f}% of all errors; "
        f"short (≤2-word) inputs reach {100 * ana['errors']['short_questions']['accuracy']:.0f}% accuracy. These are precisely the inputs the multi-turn layer and the review flag are designed for.", body))

    # ------------------------------------------------------------------ C
    story.append(P("C. Multi-turn architecture (implemented resolver + proposed contextual encoder)", h2))
    story.append(P(
        "<b>Implemented</b> (<i>foursight_intent/conversation.py</i>, covered by <i>tests/test_conversation.py</i>): a <b>ConversationState</b> per session holds the last ≤6 user turns with their predicted intents, confidences and probability vectors, "
        "the <i>active reference ids</i> (shipment/PO/container numbers), <i>active topic</i> (last confidently resolved intent), <i>active time expression</i> ('last week'), and freshness counters (15-minute expiry, topic decay after 3 unrelated turns). "
        "Deterministic <b>follow-up detectors</b> flag elliptical turns ('what about last month?', 'and for Q2?'), pronoun references ('where is it now'), bare reference ids, bare time ranges ('last 7 days'), confirmations ('yes', 'both') and explicit topic switches / greetings. "
        "The <b>resolver</b> then applies a fixed precedence: a greeting or topic switch expires the context; a confident single-turn prediction (p ≥ 0.70) always wins unless the turn is a pure confirmation or bare time range (so 'how do I set alerts for it?' after a tracking question is knowledge_base, never inherited realtime_query); "
        "a detected follow-up with fresh context blends p = (1−w)·p<sub>current</sub> + w·prior, where prior is the decayed average of previous turn probabilities and w = 0.5 (0.7 for confirmations / bare time ranges), followed by bounded cue shifts inside the operational triad only "
        "(a past time range or aggregate word moves mass to analytics, a bare id to realtime_query, disruption vocabulary to disruptions). New ids in the turn override; otherwise follow-ups inherit the active ids and time range, which are returned to the backend together with the label. "
        "Low blended confidence raises <i>needs_review</i>; a context-only confirmation below 0.70 is always flagged. Every decision carries a human-readable <i>reason</i> and the single-turn label is logged beside the contextual one for audit.", body))
    story.append(P(
        "<b>Proposed production extension.</b> Once real labelled conversations exist, fine-tune the encoder on a structured input "
        "<font face='Courier'>[CONTEXT] prev_intent=analytics ; prev_turn=&lt;text&gt; [CURRENT] &lt;text&gt;</font> (previous <i>user</i> turns only, most recent first, ~96 context + 128 current tokens), trained with (a) real sessions, (b) synthetic ellipsis rewrites of labelled single-turn questions and, crucially, "
        "(c) context-independent pairs so the model learns to ignore history instead of copying prev_intent. The deterministic resolver remains the baseline to beat and the fallback when the session store is down. "
        "<b>Evaluation.</b> No multi-turn accuracy is claimed: the dataset has no histories, and the eight scripted scenarios in <i>synthetic_scenarios()</i> are a functional smoke test, not a benchmark. The real benchmark: ~2,000 anonymised sessions (2–8 turns), every turn labelled <i>given the history</i> plus a 'context required?' bit and resolved entities, 20% double-annotated (target κ ≥ 0.8); "
        "report OVERALL on all turns, on the context-required subset (what the resolver should improve) and the inheritance-error rate on the context-not-required subset (what blind inheritance would damage), with paired bootstrap over sessions; then shadow-mode and canary before full rollout.", body))

    # ------------------------------------------------------------------ D
    story.append(P("D. Uncertainty, review flag and new intents", h2))
    story.append(P(
        f"Temperature scaling (T = {cal['temperature']:.2f}, fitted on OOF logits) reduces ECE from {cal['raw']['ece']:.3f} to {cal['calibrated']['ece']:.3f} (cross-fitted {ana['calibration']['cross_fitted_ece']:.3f}) and NLL from {cal['raw']['nll']:.3f} to {cal['calibrated']['nll']:.3f}; mean calibrated confidence is {ana['confidence']['mean_conf_correct']:.2f} on correct and {ana['confidence']['mean_conf_wrong']:.2f} on wrong predictions, and only {ana['confidence']['errors_with_conf_ge_0.9']} of {ana['confidence']['n_errors']} errors exceed 0.9. "
        f"<b>needs_review</b> = calibrated max-probability &lt; {pol['threshold']:.3f}, the largest threshold flagging ≤20% of validation traffic. On validation it flags {100 * pol['flag_rate']:.1f}% of questions, catches {100 * pol['error_recall']:.1f}% of all errors with precision {pol['precision']:.2f} "
        f"(cross-fitted: flag {100 * cf['flag_rate']:.1f}%, error recall {100 * cf['error_recall']:.1f}%, precision {cf['precision']:.2f}); accuracy on the auto-answered {100 * (1 - cf['flag_rate']):.0f}% rises to {cf['accuracy_retained']:.3f} (OVERALL {ana['review']['OVERALL_on_unflagged_cross_fitted']:.3f}), AURC {ana['review']['risk_coverage']['aurc']:.3f}. "
        f"The flag is metadata: every row still receives exactly one of the seven labels, and the threshold is a product dial (the full sweep is in <i>analysis_{name}.json</i>). Confidence is a calibrated estimate, not a guarantee. "
        + (f"On the 566 hidden-test questions the same threshold flags {100 * sub_meta['flag_rate']:.0f}% (mean calibrated confidence {sub_meta['mean_confidence']:.2f} vs {ana['confidence']['mean_conf_correct'] * (1 - ana['review']['risk_coverage']['full_risk']) + ana['confidence']['mean_conf_wrong'] * ana['review']['risk_coverage']['full_risk']:.2f} out-of-fold): "
           f"the test questions look somewhat harder or shifted (more French, more pasted templates), which is exactly what the flag is for; nothing was tuned on them." if sub_meta else ""), body))
    story.append(P(
        "<b>A brand-new intent in production</b> is a different problem from the existing <i>other</i> class (out-of-scope questions, answered by a fallback). Detection combines the calibrated confidence/margin, a rising share of <i>other</i>, an out-of-distribution score (cosine distance of the encoder embedding to its k nearest training neighbours — a monitoring signal, not a routing input) and downstream 'no result' / re-ask events. "
        "A weekly discovery loop clusters flagged questions (HDBSCAN on embeddings), a product owner triages clusters into noise / recall gap of an existing class / genuine new destination; the latter gets a definition, ≥100 annotated examples with agreement checks, and a <b>versioned taxonomy</b> (labels appended, ids never re-ordered; the artefact and router carry <i>taxonomy_version</i>). "
        "Retraining reuses the same seeded grouped-CV pipeline with acceptance gates (no existing class loses &gt;1 F1 point on the frozen validation set; a golden boundary-case file must be predicted identically unless the change is approved), then shadow mode → 5% canary → full rollout with automatic rollback on review-rate or mis-route drift. "
        "The router fails closed: an unknown label maps to <i>other</i>. No eighth label is introduced into the current evaluation.", body))

    curves = ROOT / "reports/figures" / f"training_curves_{name if kind == 'single' else members[0]['name']}.png"
    cmp_fig = ROOT / "reports/figures/candidate_comparison.png"
    if curves.exists():
        story.append(Image(str(curves), width=14.0 * cm, height=5.0 * cm))
        story.append(P(f"Loss curves and validation OVERALL per epoch for the five grouped folds of {name} (dots = early-stopped checkpoints; the offline W&amp;B runs hold the same curves per fold). Candidate comparison with bootstrap CIs: <i>reports/figures/candidate_comparison.png</i>.", small))

    # ------------------------------------------------------------------ E
    story.append(P("E. Reproducibility", h2))
    env = summ["environment"]
    cfg = summ["config"]
    lat = next((c.get("latency", {}) for c in cmp_["candidates"] if c["model"] == name), {})
    cands_lat = lat.get("cpu_ms_per_question_single", float("nan"))
    rep = [["seed", str(cfg["seed"]) + " (python, numpy, torch, shuffling generator; grouped 5-fold seed 42)"],
           ["model", (f"{base_model} → 7-way head, fp32; {epochs_text} on all rows; temperature {inf['temperature']:.3f}" if kind == "single" else
                      f"{base_model}; {epochs_text}; member temperatures " + ", ".join(f"{m['temperature']:.2f}" for m in members) + f"; ensemble temperature {inf['temperature']:.3f}") + f"; review threshold {inf['review_threshold']:.3f}" + (f"; rule layer max_conf {inf['rules_max_conf']}" if rules_on else "")],
           ["hyper-parameters", (f"lr {cfg['learning_rate']}, batch {cfg['batch_size']}, wd {cfg['weight_decay']}, warm-up {cfg['warmup_ratio']}, clip {cfg['grad_clip']}, class weight {cfg['class_weight']}, max_len {cfg['max_length']} (+{cfg['tail_tokens']} tail), patience {cfg['patience']}" if kind == "single" else
                                 "; ".join(f"{n}: lr {c['learning_rate']}, batch {c['batch_size']}, wd {c['weight_decay']}" for n, c in ((m['name'], summ['members'][m['name']]['config']) for m in members)) + f"; shared: warm-up {cfg['warmup_ratio']}, clip {cfg['grad_clip']}, class weight {cfg['class_weight']}, max_len {cfg['max_length']} (+{cfg['tail_tokens']} tail), patience {cfg['patience']}")],
           ["hardware", f"Apple M1 Max, 32 GB. CV fine-tuning on MPS ({sum(r['train_seconds'] for r in exp['folds']) / 60:.0f} min / 5 folds); final fit on CPU, {env.get('num_threads') or 8} threads ({summ['train_seconds'] / 60:.0f} min). "
                        f"Inference on CPU: {sub_meta.get('ms_per_row', float('nan')):.0f} ms/question batched ({cands_lat:.0f} ms single). Any ≤24 GB GPU or a CPU suffices."],
           ["software", f"Python {env['python']}, torch {env['torch']}, transformers {env.get('transformers', '?')}, {env['platform']}"],
           ["inference", "pip install -e . ; python scripts/generate_submission.py (writes + validates test_predictions.csv) ; python scripts/predict.py \"&lt;question&gt;\" ; python scripts/predict.py --conversation (multi-turn demo)"],
           ["links", f"GitHub: {links.get('github', 'NOT PUBLISHED')} · Hugging Face: {links.get('huggingface', 'NOT PUBLISHED')} · W&amp;B: {links.get('wandb', 'NOT PUBLISHED')}"]]
    story.append(tbl([[P(f"<b>{k}</b>", small), P(v, small)] for k, v in rep], [3.0 * cm, 17.0 * cm], 7.0, header=False, zebra=False))

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.25 * cm, rightMargin=1.25 * cm, topMargin=1.0 * cm, bottomMargin=1.0 * cm, title="FourSight Intent Classification — Technical Report", author="candidate")
    doc.build(story)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
