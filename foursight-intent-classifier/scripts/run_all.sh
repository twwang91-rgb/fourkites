#!/usr/bin/env bash
# End-to-end reproduction: EDA -> baselines -> encoder CV -> analysis -> selection -> final fit -> submission -> tests -> report.
# Usage: bash scripts/run_all.sh [device]      device = auto | cpu | mps | cuda  (default auto)
set -euo pipefail
cd "$(dirname "$0")/.."
DEVICE="${1:-auto}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_SILENT=true
PY="${PYTHON:-.venv/bin/python}"

echo "== 1. EDA";            $PY scripts/run_eda.py
echo "== 2. baselines";      $PY scripts/run_baselines.py
echo "== 3. encoder CV";
$PY scripts/run_experiments.py --name distilbert       --model distilbert/distilbert-base-uncased --lr 3e-5 --device "$DEVICE"
$PY scripts/run_experiments.py --name e5_small_ft      --model intfloat/multilingual-e5-small     --lr 5e-5 --device "$DEVICE"
$PY scripts/run_experiments.py --name deberta_v3_small --model microsoft/deberta-v3-small         --lr 5e-5 --device "$DEVICE"
$PY scripts/run_experiments.py --name roberta_base     --model FacebookAI/roberta-base            --lr 2e-5 --device "$DEVICE"
$PY scripts/run_experiments.py --name bge_small_ft     --model BAAI/bge-small-en-v1.5             --lr 5e-5 --device "$DEVICE"
# optional robustness runs (second seed for the leader, alternative class weighting); not needed for the deliverable
if [ "${ROBUSTNESS:-0}" = "1" ]; then
  $PY scripts/run_experiments.py --name deberta_v3_small_seed7 --model microsoft/deberta-v3-small --lr 5e-5 --seed 7 --device "$DEVICE"
  $PY scripts/run_experiments.py --name distilbert_invw --model distilbert/distilbert-base-uncased --lr 3e-5 --class-weight inv --device "$DEVICE"
  $PY scripts/run_experiments.py --name deberta_v3_small_invw --model microsoft/deberta-v3-small --lr 5e-5 --class-weight inv --device "$DEVICE"
fi
echo "== 4. analysis";
for f in reports/results/transformer_*.json; do n=$(basename "$f" .json); n=${n#transformer_}; $PY scripts/analyze_results.py --name "$n"; done
$PY scripts/study_logit_bias.py
$PY scripts/compare_candidates.py --latency
echo "== 5. selection";      $PY scripts/select_final.py
SEL_NAME=$($PY -c "import json;print(json.load(open('reports/results/final_selection.json'))['selected']['name'])")
SEL_RULES=$($PY -c "import json;print('--use-rules' if json.load(open('reports/results/final_selection.json'))['selected']['use_rules'] else '')")
echo "selected: $SEL_NAME $SEL_RULES"
echo "== 6. final fit";      $PY scripts/train_final.py --experiment "$SEL_NAME" $SEL_RULES --device "$DEVICE"
$PY scripts/write_model_card.py
echo "== 7. submission";     $PY scripts/generate_submission.py
echo "== 7b. official scorer on out-of-fold validation files"; $PY scripts/score_oof.py --name "$SEL_NAME" $SEL_RULES
$PY scripts/make_figures.py
echo "== 8. tests";          $PY -m pytest -q
echo "== 9. report";         $PY scripts/build_report.py
echo "done: test_predictions.csv, report.pdf, models/final/"
