#!/bin/bash
# run_mirage_trajflow_all.sh
# Run the full MIRAGE baseline pipeline for all three cities.
#
# Prerequisites:
#   - MIRAGE processed pkls exist at data/mirage_data/mirage_{city.lower()}_processed.pkl
#   - TrajFlow dependencies installed (pip install -r requirements.txt)
#
# Usage:
#   bash scripts/run_mirage_trajflow_all.sh
#   bash scripts/run_mirage_trajflow_all.sh NewYork   # single city

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAJFLOW_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$TRAJFLOW_ROOT"

CITIES=("NewYork" "Tokyo" "Istanbul")

if [ "$#" -ge 1 ]; then
    CITIES=("$@")
fi

for CITY in "${CITIES[@]}"; do
    echo ""
    echo "============================================================"
    echo "MIRAGE Baseline -- $CITY"
    echo "============================================================"

    # -- Step 0: Create output directories --
    mkdir -p "outputs/mirage_baseline/$CITY/train"
    mkdir -p "outputs/mirage_baseline/$CITY/generation"

    # -- Step 1: Data preparation --
    echo ""
    echo "[Step 1/6] Preparing data for $CITY..."
    python scripts/prepare_mirage_trajflow_data.py --city "$CITY"

    # -- Step 2: Generate config --
    echo ""
    echo "[Step 2/6] Generating TrajFlow config for $CITY..."
    python scripts/make_mirage_configs.py --city "$CITY"

    # -- Step 3: Train --
    echo ""
    echo "[Step 3/6] Training TrajFlow for $CITY..."
    CITY_LOWER=$(echo "$CITY" | tr '[:upper:]' '[:lower:]')
    CONFIG="src/config/config_mirage_${CITY_LOWER}.yaml"
    python train.py --config "$CONFIG"

    # -- Step 4: Find checkpoint and generate --
    echo ""
    echo "[Step 4/6] Generating for $CITY..."

    CHECKPOINT=$(find "outputs/mirage_baseline/$CITY/train" -name "best_model.pt" -type f 2>/dev/null | sort | tail -1)
    if [ -z "$CHECKPOINT" ]; then
        echo "ERROR: No best_model.pt found in outputs/mirage_baseline/$CITY/train"
        echo "Skipping generation for $CITY"
        continue
    fi
    echo "  Using checkpoint: $CHECKPOINT"

    TEST_SIZE=$(python -c "import pickle; print(len(pickle.load(open('data/mirage_trajflow/${CITY}/test_metadata.pkl','rb'))))" 2>/dev/null || echo "0")
    if [ "$TEST_SIZE" -eq 0 ]; then
        echo "ERROR: Could not determine test size for $CITY"
        continue
    fi
    echo "  Test set size: $TEST_SIZE"

    python generate.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --generate_num "$TEST_SIZE" \
        --generate_results_dir "outputs/mirage_baseline/$CITY/generation"

    # -- Step 5: Postprocess --
    echo ""
    echo "[Step 5/6] Postprocessing for $CITY..."
    python scripts/postprocess_mirage_trajflow_generated.py --city "$CITY" --skip_gps_check

    # -- Step 6: Evaluate --
    echo ""
    echo "[Step 6/6] Evaluating $CITY..."
    python scripts/evaluate_mirage_5d.py --city "$CITY"

    echo ""
    echo "Results for $CITY:"
    echo "  Config:   $CONFIG"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Metrics: outputs/mirage_baseline/$CITY/metrics_5d.txt"
    echo "  Generated pkl: outputs/mirage_baseline/$CITY/generated.pkl"

    echo ""
    echo "============================================================"
    echo "MIRAGE Baseline -- $CITY -- DONE"
    echo "============================================================"
    echo ""

done

echo ""
echo "============================================================"
echo "All cities complete!"
echo "============================================================"
echo ""
echo "Results summary:"
for CITY in "${CITIES[@]}"; do
    METRICS_FILE="outputs/mirage_baseline/$CITY/metrics_5d.txt"
    if [ -f "$METRICS_FILE" ]; then
        echo ""
        echo "=== $CITY ==="
        cat "$METRICS_FILE"
    else
        echo "=== $CITY: metrics not available ==="
    fi
done
