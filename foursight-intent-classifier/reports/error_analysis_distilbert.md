# Error analysis — distilbert (pooled out-of-fold, grouped CV)

accuracy 0.6943 · macro-F1 0.5696 · **OVERALL 0.6195** · errors 582 / 1904

## Per-class

| class | precision | recall | F1 | support |
|---|---:|---:|---:|---:|
| shipment_information | 0.347 | 0.426 | 0.382 | 256 |
| realtime_query | 0.782 | 0.746 | 0.763 | 782 |
| analytics | 0.600 | 0.584 | 0.592 | 257 |
| disruptions | 0.826 | 0.679 | 0.745 | 28 |
| chitchat | 0.625 | 0.294 | 0.400 | 17 |
| knowledge_base | 0.857 | 0.872 | 0.865 | 509 |
| other | 0.267 | 0.218 | 0.240 | 55 |

## Most confused pairs (gold → predicted)

- realtime_query → shipment_information: 130
- shipment_information → realtime_query: 101
- analytics → realtime_query: 47
- realtime_query → analytics: 44
- analytics → shipment_information: 32
- shipment_information → analytics: 25
- analytics → knowledge_base: 25
- knowledge_base → analytics: 22
- knowledge_base → shipment_information: 21
- other → knowledge_base: 16

Operational-triad errors: {'realtime_query->shipment_information': 130, 'shipment_information->realtime_query': 101, 'analytics->realtime_query': 47, 'realtime_query->analytics': 44, 'analytics->shipment_information': 32, 'shipment_information->analytics': 25, 'total_triad_errors': 379}

## Calibration and review flag

- temperature T = 1.195; ECE 0.0752 → 0.0349; NLL 0.9050 → 0.8877; Brier 0.4336 → 0.4269
- mean calibrated confidence: correct 0.762 vs wrong 0.578; margin 0.634 vs 0.354; 34 errors with confidence ≥ 0.9
- needs_review = confidence < 0.509: flags 20.0% of rows, error recall 39.5%, precision 0.61, accuracy on unflagged 0.769
- cross-fitted (threshold chosen on 4 folds, applied to the 5th): flag 20.7%, error recall 40.9%, precision 0.60, accuracy on unflagged 0.772, OVERALL on unflagged 0.6621
- AURC 0.1519 (full-coverage risk 0.3057)

## Rule-layer ablation (calibrated probabilities)

| max_conf | eligible | changed | fixed | broke | accuracy | macro-F1 | OVERALL | triad errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0 | 0 | 0 | 0 | 0.6943 | 0.5696 | 0.6195 | 379 |
| 0.50 | 356 | 9 | 4 | 4 | 0.6943 | 0.5736 | 0.6219 | 379 |
| 0.60 | 631 | 16 | 9 | 6 | 0.6959 | 0.6086 | 0.6435 | 380 |
| 0.70 | 856 | 16 | 9 | 6 | 0.6959 | 0.6086 | 0.6435 | 380 |
| 0.80 | 1105 | 16 | 9 | 6 | 0.6959 | 0.6086 | 0.6435 | 380 |
| 1.01 | 1904 | 16 | 9 | 6 | 0.6959 | 0.6086 | 0.6435 | 380 |

## Failure categories

Short (≤2 words) inputs: n=204, accuracy 0.618. Rows in conflicting-label duplicate groups: n=81, accuracy 0.370 (9% of all errors). Rare-class recall: disruptions 0.68, chitchat 0.29, other 0.22.

### triad: semantic — 285 errors, mean confidence 0.58

- `schneider trailer TA214262` — gold **shipment_information**, predicted realtime_query (0.96)
- `I would like to have all of the updates on this load, please tell me where it's at and the expected delivery d` — gold **analytics**, predicted realtime_query (0.96)
- `When will SID 00332009115347990 be delivered?` — gold **shipment_information**, predicted realtime_query (0.96)
- `Origin is Store #3115, 7100 Tecumseh Road East Windsor` — gold **shipment_information**, predicted realtime_query (0.95)
- `why was this shipment lane -00332009115314183` — gold **shipment_information**, predicted realtime_query (0.95)
- `Give me a list of loads scheduled to deliver to the locations below. P-JWLO-DFW-TX, P-JWLO-HOU-TX, P-PTMG-SAN-` — gold **shipment_information**, predicted realtime_query (0.95)
- `Can you please show only the loads with inbound tags` — gold **shipment_information**, predicted realtime_query (0.94)
- `What was the late load?` — gold **shipment_information**, predicted realtime_query (0.93)

### knowledge_base vs operational — 134 errors, mean confidence 0.57

- `where can I see carriers set up and their status` — gold **analytics**, predicted knowledge_base (0.97)
- `how do I know if a truck is a team driver` — gold **other**, predicted knowledge_base (0.97)
- `where can I review these metrics?` — gold **analytics**, predicted knowledge_base (0.97)
- `How is Hill Brothers sending Four Kites location updates? They asked us and I don't have an answer.` — gold **analytics**, predicted knowledge_base (0.96)
- `is Fourkites using Grapphopper? to get geolocations?` — gold **other**, predicted knowledge_base (0.95)
- `For this delivery, What are the updates done by driver` — gold **realtime_query**, predicted knowledge_base (0.95)
- `ok am now in customs export but i can see OTR and not Tl/LTL also i can't find delivered staus. also i don't l` — gold **knowledge_base**, predicted analytics (0.94)
- `is there a way to see which set up we have with each carrier in fourkites, if eld or phone number` — gold **analytics**, predicted knowledge_base (0.93)

### short/underspecified — 78 errors, mean confidence 0.58

- `Exemplo: R_6661671` — gold **shipment_information**, predicted realtime_query (0.96)
- `73694283` — gold **shipment_information**, predicted realtime_query (0.96)
- `4385338688` — gold **shipment_information**, predicted realtime_query (0.95)
- ` ECDC27655765` — gold **shipment_information**, predicted realtime_query (0.94)
- `STORE NUMBER` — gold **shipment_information**, predicted realtime_query (0.89)
- `1ZX600550392692343` — gold **shipment_information**, predicted realtime_query (0.86)
- `524104441364` — gold **shipment_information**, predicted realtime_query (0.84)
- `4308462335` — gold **shipment_information**, predicted realtime_query (0.83)

### triad: temporal/aggregate cue — 40 errors, mean confidence 0.61

- `How many shipments with tag APMA are currently stuck at a transshipment port for more than 5 days` — gold **analytics**, predicted realtime_query (0.94)
- `How many shipmets total last week` — gold **realtime_query**, predicted analytics (0.91)
- `how many tl moves took place yesterday?` — gold **shipment_information**, predicted analytics (0.90)
- `What trucks to i have in transit that are between Laredo, TX and Pharr, TX within the last 2 hours?` — gold **analytics**, predicted realtime_query (0.85)
- `How many missed deliveries have we had for the locations Butner, Maulden, Elloree and Dunn in the past 30 days` — gold **realtime_query**, predicted analytics (0.84)
- `how many loads do i have that have not been collected from Bayer Raceway this week` — gold **realtime_query**, predicted analytics (0.84)
- `I need to know how many shipments were tracked in Fourkites for the past 12 months.` — gold **realtime_query**, predicted analytics (0.83)
- `How many PO's were missed pickup yesterday inbounding Canada` — gold **analytics**, predicted realtime_query (0.83)

### other / out-of-scope boundary — 34 errors, mean confidence 0.56

- `Pickup and delivery number to use for this will be N1091681` — gold **other**, predicted realtime_query (0.96)
- `CMA BEX. Is there any benchmark on transit time?` — gold **other**, predicted analytics (0.91)
- `purchasing document status - 4507125913` — gold **other**, predicted realtime_query (0.87)
- `purchasing document status-PO 4507153487` — gold **other**, predicted realtime_query (0.86)
- `Freight quote from Ontario, CA to Germantown , WI` — gold **other**, predicted shipment_information (0.86)
- `What was the late` — gold **other**, predicted realtime_query (0.74)
- `can you help me delivery?` — gold **other**, predicted shipment_information (0.72)
- `Please put the results in presentation format` — gold **analytics**, predicted other (0.70)

### disruptions boundary — 9 errors, mean confidence 0.60

- `me refiero solo al BS20260723-1235. Tiene o pasara poe algun punto de riesgo?` — gold **disruptions**, predicted realtime_query (0.83)
- `provide summary of plant wise, shipments impacted by dolpin` — gold **disruptions**, predicted analytics (0.78)
- `What is the wait time for carriers crossing the World Trade Bridge in Laredo today 7/29/2026?` — gold **realtime_query**, predicted disruptions (0.72)
- `any issues happened in the past one week for prince rupert port? Thanks` — gold **disruptions**, predicted analytics (0.71)
- `the route of BS20260723-1235 have montain regions with curves?` — gold **disruptions**, predicted realtime_query (0.65)
- `What is the current port condition at Shanghai port` — gold **disruptions**, predicted realtime_query (0.55)
- `Can you show me all of our containers that will experience delays due to having to go around the Panama Canal` — gold **disruptions**, predicted shipment_information (0.44)
- `What shipments are at risk` — gold **realtime_query**, predicted disruptions (0.41)

### chitchat boundary — 2 errors, mean confidence 0.55

- `Hello You are my supply chain analyst who is dealing with very complex supply chain from north america to phil` — gold **chitchat**, predicted analytics (0.76)
- `okay what u find` — gold **realtime_query**, predicted chitchat (0.34)
