#!/bin/bash

################################################################################
# Analyze Calibration for ALL Models
################################################################################

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

cd "$PROJECT_ROOT"

echo "======================================================================"
echo "  Calibration Analysis for ALL Models"
echo "======================================================================"
echo ""

echo "📊 Step 1: Adding confidence metrics to all model results..."
python scripts/targetted_assessment/analysis/add_calibration_to_all_models.py

echo ""
echo "📊 Step 2: Creating calibration comparison visualization..."
python scripts/targetted_assessment/visualization/compare_model_calibration.py

echo ""
echo "======================================================================"
echo "  ✅ Complete!"
echo "======================================================================"
echo ""
echo "📁 Files created in: results/targetted_assessment/"
echo ""
echo "Generated files:"
echo "  • model_calibration_summary.csv - Summary of all models"
echo "  • model_calibration_comparison.png - Comparison visualization"
echo "  • *_with_confidence.csv - Enhanced results for each model"
echo ""
