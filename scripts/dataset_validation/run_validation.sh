#!/bin/bash

################################################################################
# Dataset Validation Runner
# 
# This script runs the comprehensive linguistic validation analysis on the
# AnaphoraGym dataset to establish its credibility through objective metrics.
################################################################################

set -e  # Exit on error

echo "========================================================================"
echo "  AnaphoraGym Dataset Validation Analysis"
echo "========================================================================"
echo ""

# Get the project root directory (two levels up from this script)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "📁 Project root: $PROJECT_ROOT"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check if dataset exists
DATASET_PATH="dataset/AnaphoraGym.csv"
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found at $DATASET_PATH"
    exit 1
fi

echo "✅ Found dataset: $DATASET_PATH"
echo ""

# Check if textstat is installed
echo "🔍 Checking dependencies..."
if ! python -c "import textstat" 2>/dev/null; then
    echo "⚠️  textstat not found. Installing..."
    pip install textstat
    echo ""
fi

if ! python -c "import pandas" 2>/dev/null; then
    echo "⚠️  pandas not found. Installing required packages..."
    pip install -r requirements.txt
    echo ""
fi

echo "✅ All dependencies satisfied"
echo ""

# Create output directory
OUTPUT_DIR="results/dataset_validation"
mkdir -p "$OUTPUT_DIR"

echo "📊 Running validation analysis..."
echo ""

# Run the validation script
python scripts/dataset_validation/validate_dataset.py \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  ✅ Analysis Complete!"
echo "========================================================================"
echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "📄 Generated files:"
echo "   - VALIDATION_REPORT.md          (Comprehensive report)"
echo "   - readability_distribution.png  (Readability metrics)"
echo "   - category_comparison.png       (Category analysis)"
echo "   - input_vs_continuation.png     (Input/continuation comparison)"
echo "   - metrics_heatmap.png           (Comprehensive heatmap)"
echo "   - complexity_vs_performance.png (Complexity vs accuracy, if available)"
echo "   - *.csv                         (Detailed metric data)"
echo ""
echo "📖 Next steps:"
echo "   1. Review VALIDATION_REPORT.md for key findings"
echo "   2. Use visualizations in your paper/presentation"
echo "   3. Cite the objective metrics to establish credibility"
echo ""
