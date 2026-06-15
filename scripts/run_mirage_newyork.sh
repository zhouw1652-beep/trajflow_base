#!/bin/bash
# run_mirage_newyork.sh  --  NewYork only
# Usage:  bash scripts/run_mirage_newyork.sh
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAJFLOW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TRAJFLOW_ROOT"
CITY="NewYork"

mkdir -p "outputs/mirage_baseline/$CITY/train"
mkdir -p "outputs/mirage_baseline/$CITY/generation"

echo "[1/6] Preparing data..."
python scripts/prepare_mirage_trajflow_data.py --city "$CITY"

echo "[2/6] Generating config..."
python scripts/make_mirage_configs.py --city "$CITY"

echo "[3/6] Training..."
python train.py --config "src/config/config_mirage_newyork.yaml"

CHECKPOINT=$(find "outputs/mirage_baseline/$CITY/train" -name "best_model.pt" -type f 2>/dev/null | sort | tail -1)
if [ -z "$CHECKPOINT" ]; then echo "ERROR: no checkpoint"; exit 1; fi
echo "[4/6] Generating (checkpoint: $CHECKPOINT)..."
TEST_SIZE=$(python -c "import pickle; print(len(pickle.load(open('data/mirage_trajflow/${CITY}/test_metadata.pkl','rb'))))" 2>/dev/null || echo "0")
if [ "$TEST_SIZE" -eq 0 ]; then
    echo "ERROR: Could not determine test size for $CITY"
    exit 1
fi
echo "  Test set size: $TEST_SIZE"
python generate.py \
    --config "src/config/config_mirage_newyork.yaml" \
    --checkpoint "$CHECKPOINT" \
    --generate_num "$TEST_SIZE" \
    --generate_results_dir "outputs/mirage_baseline/$CITY/generation"

echo "[5/6] Postprocessing..."
python scripts/postprocess_mirage_trajflow_generated.py --city "$CITY" --skip_gps_check

echo "[6/6] Evaluating..."
python scripts/evaluate_mirage_5d.py --city "$CITY"

echo "Done. Metrics: outputs/mirage_baseline/$CITY/metrics_5d.txt"
cat outputs/mirage_baseline/$CITY/metrics_5d.txt
