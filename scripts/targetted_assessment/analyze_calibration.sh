#!/bin/bash

################################################################################
# Analyze Model Calibration
# 
# Adds confidence metrics and creates calibration visualizations
################################################################################

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo ""
    echo "Examples:"
    echo "  $0 gpt2"
    echo "  $0 gpt2-large"
    exit 1
fi

MODEL_NAME="$1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "======================================================================"
echo "  Calibration Analysis for: $MODEL_NAME"
echo "======================================================================"
echo ""

echo "📊 Step 1: Adding confidence metrics to results..."
python3 scripts/targetted_assessment/analysis/add_confidence_metrics.py --model "$MODEL_NAME"

echo ""
echo "📊 Step 2: Creating calibration visualization..."
python3 scripts/targetted_assessment/visualization/visualize_calibration.py --model "$MODEL_NAME"

echo ""
echo "======================================================================"
echo "  ✅ Complete!"
echo "======================================================================"
echo ""
echo "📁 Files created in: results/targetted_assessment/"
SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
echo "  • AnaphoraGym_Results_${SAFE_MODEL_NAME}_with_confidence.csv"
echo "  • calibration_analysis_${SAFE_MODEL_NAME}.png"
echo ""
