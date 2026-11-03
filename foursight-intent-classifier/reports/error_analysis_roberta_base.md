# Error analysis — roberta_base (pooled out-of-fold, grouped CV)

accuracy 0.7169 · macro-F1 0.5683 · **OVERALL 0.6277** · errors 539 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.370 | 0.410 | 0.389 | 256 |
| realtime_query | 0.770 | 0.781 | 0.776 | 782 |
| analytics | 0.672 | 0.623 | 0.646 | 257 |
| disruptions | 0.808 | 0.750 | 0.778 | 28 |
| chitchat | 0.571 | 0.235 | 0.333 | 17 |
| knowledge_base | 0.889 | 0.896 | 0.892 | 509 |
| other | 0.186 | 0.145 | 0.163 | 55 |

## Most confused pairs (gold → predicted)

- realtime_query → shipment_information: 115
- shipment_information → realtime_query: 114
- analytics → realtime_query: 46
- realtime_query → analytics: 34
- analytics → shipment_information: 28
- shipment_information → analytics: 24
- analytics → knowledge_base: 20
- other → knowledge_base: 17
- knowledge_base → analytics: 16
- knowledge_base → shipment_information: 15

Operational-triad errors: {'realtime_query->shipment_information': 115, 'shipment_information->realtime_query': 114, 'analytics->realtime_query': 46, 'realtime_query->analytics': 34, 'analytics->shipment_information': 28, 'shipment_information->analytics': 24, 'total_triad_errors': 361}

## Calibration and review flag

- temperature T = 1.254; ECE 0.0810 → 0.0286; NLL 0.8380 → 0.8102; Brier 0.4035 → 0.3938
- mean calibrated confidence: correct 0.785 vs wrong 0.603; margin 0.660 vs 0.378; 30 errors with confidence ≥ 0.9
- needs_review = confidence < 0.530: flags 19.9% of rows, error recall 40.6%, precision 0.58, accuracy on unflagged 0.790
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 20.0%, error recall 41.0%, precision 0.58, accuracy on unflagged 0.791, OVERALL on unflagged 0.6834
- AURC 0.1291 (full-coverage risk 0.2831)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.7169 | 0.5683 | 0.6277 | 361 |
| 0.50 | 286 | 10 | 7 | 3 | 0.7190 | 0.6090 | 0.6530 | 363 |
| 0.60 | 531 | 12 | 9 | 3 | 0.7201 | 0.6095 | 0.6537 | 361 |
| 0.70 | 723 | 12 | 9 | 3 | 0.7201 | 0.6095 | 0.6537 | 361 |
| 0.80 | 1010 | 12 | 9 | 3 | 0.7201 | 0.6095 | 0.6537 | 361 |
| 1.01 | 1904 | 12 | 9 | 3 | 0.7201 | 0.6095 | 0.6537 | 361 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.603. Rows in conflicting-label duplicate groups: n=81, accuracy 0.383 (9% of all errors). Rare-class recall: disruptions 0.75, chitchat 0.24, other 0.15.

### triad: semantic — 272 errors, mean confidence 0.60

- `why was this shipment lane -00332009115314183` — gold **shipment_information**, predicted realtime_query (0.93)
- `where is sn 0034819281` — gold **shipment_information**, predicted realtime_query (0.93)
- `388990085 what is the current locaiton of this truck?` — gold **shipment_information**, predicted realtime_query (0.93)
- `can you give me an update on PO 10217255` — gold **analytics**, predicted realtime_query (0.93)
- `When will SID 00332009115347990 be delivered?` — gold **shipment_information**, predicted realtime_query (0.92)
- `0034902422 0034902434 0034898649 0034903814 0034899888 0034899896 how are these loads for on time preformance` — gold **analytics**, predicted realtime_query (0.92)
- `8006895826 is there an appt yet` — gold **shipment_information**, predicted realtime_query (0.92)
- `I would like to have all of the updates on this load, please tell me where it's at and the expected delivery d` — gold **analytics**, predicted realtime_query (0.91)

### knowledge_base vs operational — 104 errors, mean confidence 0.63

- `how can I connect you to a powerBI report from where you can read shipment data` — gold **other**, predicted knowledge_base (0.95)
- `where can I see carriers set up and their status` — gold **analytics**, predicted knowledge_base (0.95)
- `is Fourkites using Grapphopper? to get geolocations?` — gold **other**, predicted knowledge_base (0.95)
- `where can I review these metrics?` — gold **analytics**, predicted knowledge_base (0.95)
- `como passar isso pra transportadora?` — gold **other**, predicted knowledge_base (0.95)
- `How is Hill Brothers sending Four Kites location updates? They asked us and I don't have an answer.` — gold **analytics**, predicted knowledge_base (0.94)
- `what is the precise GPS ping` — gold **realtime_query**, predicted knowledge_base (0.94)
- `is there a way to see which set up we have with each carrier in fourkites, if eld or phone number` — gold **analytics**, predicted knowledge_base (0.94)

### short/underspecified — 81 errors, mean confidence 0.58

- ` ECDC27655765` — gold **shipment_information**, predicted realtime_query (0.90)
- `4308462335` — gold **shipment_information**, predicted realtime_query (0.90)
- `4385338688` — gold **shipment_information**, predicted realtime_query (0.90)
- `Exemplo: R_6661671` — gold **shipment_information**, predicted realtime_query (0.90)
- `SOU1001399023` — gold **shipment_information**, predicted realtime_query (0.89)
- `8006899320` — gold **shipment_information**, predicted realtime_query (0.88)
- `CMV3913820` — gold **shipment_information**, predicted realtime_query (0.87)
- `524104441364` — gold **shipment_information**, predicted realtime_query (0.86)

### other / out-of-scope boundary — 39 errors, mean confidence 0.56

- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.95)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.93)
- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.93)
- `Pickup and delivery number to use for this will be N1091681` — gold **other**, predicted realtime_query (0.86)
- `Please put the results in presentation format` — gold **analytics**, predicted other (0.83)
- `Pull up all Mann Lake orders for delvieyr this week` — gold **other**, predicted analytics (0.80)
- `Put your response in a format that can be copied and shared` — gold **realtime_query**, predicted other (0.74)
- `this is awesome !` — gold **chitchat**, predicted other (0.72)

### triad: temporal/aggregate cue — 35 errors, mean confidence 0.63

- `How many shipments with tag APMA are currently stuck at a transshipment port for more than 5 days` — gold **analytics**, predicted realtime_query (0.88)
- `How many PO's were missed pickup yesterday inbounding Canada` — gold **analytics**, predicted realtime_query (0.88)
- `can you tell me how many loads for the scac SLSV that has reason codes for lates inputted into the loads` — gold **realtime_query**, predicted analytics (0.88)
- `show me all departures for carrier DLS in the past 7 days` — gold **shipment_information**, predicted analytics (0.87)
- `I need to know how many shipments were tracked in Fourkites for the past 12 months.` — gold **realtime_query**, predicted analytics (0.85)
- `How many shipmets total last week` — gold **realtime_query**, predicted analytics (0.84)
- `What trucks to i have in transit that are between Laredo, TX and Pharr, TX within the last 2 hours?` — gold **analytics**, predicted realtime_query (0.78)
- `how many trucks were late or very late last week` — gold **shipment_information**, predicted realtime_query (0.76)

### disruptions boundary — 7 errors, mean confidence 0.63

- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.92)
- `What is the wait time for carriers crossing the World Trade Bridge in Laredo today 7/29/2026?` — gold **realtime_query**, predicted disruptions (0.75)
- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.75)
- `can you look at my transit time during the last two weeks of july vs. previous performance. Show any trends du` — gold **analytics**, predicted disruptions (0.65)
- `What shipments are at risk` — gold **realtime_query**, predicted disruptions (0.56)
- `any issues happened in the past one week for prince rupert port? Thanks` — gold **disruptions**, predicted analytics (0.50)
- `What is the current port condition at Shanghai port` — gold **disruptions**, predicted realtime_query (0.29)

### chitchat boundary — 1 errors, mean confidence 0.54

- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted shipment_information (0.54)
