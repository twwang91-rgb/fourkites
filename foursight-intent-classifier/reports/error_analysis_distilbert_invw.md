# Error analysis — distilbert_invw (pooled out-of-fold, grouped CV)

accuracy 0.6922 · macro-F1 0.6148 · **OVERALL 0.6458** · errors 586 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.372 | 0.449 | 0.407 | 256 |
| realtime_query | 0.799 | 0.708 | 0.751 | 782 |
| analytics | 0.572 | 0.619 | 0.594 | 257 |
| disruptions | 0.714 | 0.714 | 0.714 | 28 |
| chitchat | 0.846 | 0.647 | 0.733 | 17 |
| knowledge_base | 0.854 | 0.874 | 0.864 | 509 |
| other | 0.226 | 0.255 | 0.239 | 55 |

## Most confused pairs (gold → predicted)

- realtime_query → shipment_information: 140
- shipment_information → realtime_query: 86
- realtime_query → analytics: 51
- analytics → realtime_query: 39
- shipment_information → analytics: 34
- analytics → shipment_information: 27
- analytics → knowledge_base: 27
- knowledge_base → analytics: 24
- realtime_query → other: 21
- other → knowledge_base: 18

Operational-triad errors: {'realtime_query->shipment_information': 140, 'shipment_information->realtime_query': 86, 'realtime_query->analytics': 51, 'analytics->realtime_query': 39, 'shipment_information->analytics': 34, 'analytics->shipment_information': 27, 'total_triad_errors': 377}

## Calibration and review flag

- temperature T = 1.189; ECE 0.0765 → 0.0453; NLL 0.9063 → 0.8890; Brier 0.4456 → 0.4363
- mean calibrated confidence: correct 0.758 vs wrong 0.607; margin 0.627 vs 0.390; 14 errors with confidence ≥ 0.9
- needs_review = confidence < 0.529: flags 20.0% of rows, error recall 34.8%, precision 0.54, accuracy on unflagged 0.749
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 20.3%, error recall 35.0%, precision 0.53, accuracy on unflagged 0.749, OVERALL on unflagged 0.6894
- AURC 0.1506 (full-coverage risk 0.3078)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.6922 | 0.6148 | 0.6458 | 377 |
| 0.50 | 293 | 6 | 4 | 2 | 0.6933 | 0.6135 | 0.6454 | 376 |
| 0.60 | 554 | 9 | 4 | 5 | 0.6917 | 0.6121 | 0.6439 | 379 |
| 0.70 | 806 | 9 | 4 | 5 | 0.6917 | 0.6121 | 0.6439 | 379 |
| 0.80 | 1135 | 9 | 4 | 5 | 0.6917 | 0.6121 | 0.6439 | 379 |
| 1.01 | 1904 | 9 | 4 | 5 | 0.6917 | 0.6121 | 0.6439 | 379 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.618. Rows in conflicting-label duplicate groups: n=81, accuracy 0.358 (9% of all errors). Rare-class recall: disruptions 0.71, chitchat 0.65, other 0.25.

### triad: semantic — 286 errors, mean confidence 0.60

- `PO UCCA9069 and invoice 03251349` — gold **shipment_information**, predicted realtime_query (0.91)
- `PO UCCA9069 and invoice 03251349` — gold **shipment_information**, predicted realtime_query (0.91)
- `TMS Shipment ID Order Creation Date ETA at POD Booking # Container # Carrier 2066922253 2026-07-31 2026-10-15 ` — gold **realtime_query**, predicted analytics (0.90)
- `IS THERE A DELIVERY TODAY BY A TRAILER WITH THE DIGITS 1159` — gold **shipment_information**, predicted realtime_query (0.89)
- `only TL shipments` — gold **realtime_query**, predicted shipment_information (0.89)
- `truck number 208398576-249` — gold **shipment_information**, predicted realtime_query (0.89)
- `Show me fresno to Dallas loads` — gold **realtime_query**, predicted shipment_information (0.88)
- `Loads IDs. i need on time pick up and on time delivery % for the shipments listed` — gold **realtime_query**, predicted analytics (0.88)

### knowledge_base vs operational — 133 errors, mean confidence 0.62

- `how can i have a monthly report that shows me how many road shipments was happend in the last month.` — gold **knowledge_base**, predicted analytics (0.93)
- `Honestly, I'm not sure how we flag them. We do them as exceptions within SAP` — gold **realtime_query**, predicted knowledge_base (0.93)
- `what's the bread crumbs mention on the metrics?` — gold **knowledge_base**, predicted analytics (0.93)
- `what's the bread crumbs mention on the metrics?` — gold **knowledge_base**, predicted analytics (0.93)
- `Who do I contact about a driver issue?` — gold **other**, predicted knowledge_base (0.92)
- `how to send the link` — gold **other**, predicted knowledge_base (0.92)
- `how do i make it to be only 1 contract` — gold **other**, predicted knowledge_base (0.91)
- `how can I connect you to a powerBI report from where you can read shipment data` — gold **other**, predicted knowledge_base (0.90)

### short/underspecified — 78 errors, mean confidence 0.63

- `1ZX600550392692343` — gold **shipment_information**, predicted realtime_query (0.88)
- `20260901` — gold **shipment_information**, predicted realtime_query (0.87)
- `load number` — gold **realtime_query**, predicted shipment_information (0.87)
- `524104441364` — gold **shipment_information**, predicted realtime_query (0.87)
- `4308462335` — gold **shipment_information**, predicted realtime_query (0.86)
- `all shipments` — gold **realtime_query**, predicted shipment_information (0.83)
- `PATX008410` — gold **shipment_information**, predicted realtime_query (0.82)
- `8006874591` — gold **shipment_information**, predicted realtime_query (0.82)

### triad: temporal/aggregate cue — 38 errors, mean confidence 0.63

- `show me all departures for carrier DLS in the past 7 days` — gold **shipment_information**, predicted analytics (0.93)
- `I need to know how many shipments were tracked in Fourkites for the past 12 months.` — gold **realtime_query**, predicted analytics (0.90)
- `How many shipmets total last week` — gold **realtime_query**, predicted analytics (0.87)
- `I want to see last month, TL, source from hazelton, geodis nashville and sparks. Just give me FourKites status` — gold **realtime_query**, predicted analytics (0.83)
- `The average transit time from ontario, canada to Puerto Rico US` — gold **realtime_query**, predicted analytics (0.82)
- `how many loads do we have since 9/1/2026 marked as delivered with actual time of arrival = auto arrival` — gold **realtime_query**, predicted analytics (0.82)
- `what deliveries are behind schedule for delivery tomorrow in Newnan GA, Hazleton PA and Marshall MO? also summ` — gold **shipment_information**, predicted realtime_query (0.81)
- `how many loads do i have that have not been collected from Bayer Raceway this week` — gold **realtime_query**, predicted analytics (0.81)

### other / out-of-scope boundary — 37 errors, mean confidence 0.57

- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.89)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.89)
- `make a dashboard 07A` — gold **other**, predicted shipment_information (0.86)
- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.85)
- `Delivery: America Chung Nam Inc – ND Paper 621 N Biron Dr, Wisconsin Rapids, WI 54494` — gold **other**, predicted realtime_query (0.83)
- `get me a spread sheet with the SCAC code NTDE` — gold **realtime_query**, predicted other (0.82)
- `remove the reyadh, and also take the median` — gold **analytics**, predicted other (0.74)
- `Pickup and delivery number to use for this will be N1091681` — gold **other**, predicted realtime_query (0.74)

### disruptions boundary — 12 errors, mean confidence 0.56

- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.87)
- `please focus analysis only on ocean lanes.` — gold **shipment_information**, predicted disruptions (0.79)
- `What is the wait time for carriers crossing the World Trade Bridge in Laredo today 7/29/2026?` — gold **realtime_query**, predicted disruptions (0.71)
- `provide summary of plant wise, shipments impacted by dolpin` — gold **disruptions**, predicted analytics (0.64)
- `any issues happened in the past one week for prince rupert port? Thanks` — gold **disruptions**, predicted analytics (0.63)
- `I need to understand which vessels have omitted Montevideo port between August 7th y 10th` — gold **analytics**, predicted disruptions (0.59)
- `can you look at my transit time during the last two weeks of july vs. previous performance. Show any trends du` — gold **analytics**, predicted disruptions (0.57)
- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.54)

### chitchat boundary — 2 errors, mean confidence 0.61

- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted analytics (0.87)
- `ya do that` — gold **realtime_query**, predicted chitchat (0.34)
