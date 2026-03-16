#!/bin/bash

################################################################################
# Run assessment and visualization for a single model
################################################################################

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <model_name>"
    echo ""
    echo "Examples:"
    echo "  $0 gpt2"
    echo "  $0 gpt2-large"
    echo "  $0 meta-llama/Llama-3.2-1B"
    exit 1
fi

MODEL_NAME="$1"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "======================================================================"
echo "  Running Assessment for: $MODEL_NAME"
echo "======================================================================"
echo ""

# Run assessment
echo "📊 Step 1: Running assessment..."
python3 scripts/targeted_assessment/experiments/run_experiment.py --model "$MODEL_NAME"

echo ""
echo "📊 Step 2: Creating visualization..."
python3 scripts/targeted_assessment/visualization/visualize_single_model.py --model "$MODEL_NAME"

echo ""
echo "======================================================================"
echo "  ✅ Complete!"
echo "======================================================================"
echo ""
echo "📁 Results saved in: results/targetted_assessment/"
echo ""
echo "Files created:"
SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
echo "  • AnaphoraGym_Results_${SAFE_MODEL_NAME}.csv"
echo "  • accuracy_by_condition_${SAFE_MODEL_NAME}.csv"
echo "  • single_model_results_${SAFE_MODEL_NAME}.png"
echo ""
