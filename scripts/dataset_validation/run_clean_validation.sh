#!/bin/bash

################################################################################
# Clean Dataset Validation - Focused Analysis
# 
# Shows:
# 1. Performance vs text length/complexity
# 2. Input similarity (proves task isn't trivial guessing)
# 3. Combined insight (similarity + performance = thoughtful decisions)
################################################################################

set -e

echo "======================================================================"
echo "  🎯 Clean Dataset Validation - Focused Analysis"
echo "======================================================================"
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Check dataset
DATASET_PATH="dataset/AnaphoraGym.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

echo "✅ Found dataset: $DATASET_PATH"
echo ""

# Create output directory
OUTPUT_DIR="results/dataset_validation_clean"
mkdir -p "$OUTPUT_DIR"

echo "📊 Running clean validation analysis..."
echo ""

# Run the clean validation script
python3 scripts/dataset_validation/validate_dataset_clean.py \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR"

echo ""
echo "======================================================================"
echo "  ✅ Clean Analysis Complete!"
echo "======================================================================"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📊 Generated visualizations:"
echo "   • performance_vs_complexity.png"
echo "     → Shows how text length/readability affects accuracy"
echo ""
echo "   • input_similarity_analysis.png"
echo "     → Proves inputs are similar (task requires careful thought)"
echo ""
echo "   • similarity_vs_performance_insight.png"
echo "     → KEY INSIGHT: High similarity + good accuracy = not guessing!"
echo ""
echo "📄 Report: CLEAN_VALIDATION_REPORT.md"
echo ""
