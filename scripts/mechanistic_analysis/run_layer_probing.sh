#!/bin/bash
# Layer-wise Probing Analysis for Anaphora Resolution
# This script runs the complete pipeline for analyzing which layers
# of Llama-2-7b-chat-hf encode anaphoric information.

set -e  # Exit on error

# Configuration
MODEL="meta-llama/Llama-2-7b-chat-hf"
DATASET="dataset/AnaphoraGym.csv"
RESULTS_DIR="results/mechanistic_analysis/layer_probing"
IMAGES_DIR="images/layer_probing"
MAX_SAMPLES=10  # Number of samples per condition (increase for more comprehensive analysis)

# Optional: Specific conditions to analyze (comment out to analyze all)
# CONDITIONS="stripping_VPE joins"

echo "============================================"
echo "Layer-wise Anaphora Probing Analysis"
echo "============================================"
echo ""
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Max samples per condition: $MAX_SAMPLES"
echo "Results directory: $RESULTS_DIR"
echo "Images directory: $IMAGES_DIR"
echo ""

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$IMAGES_DIR"

# Step 1: Run layer probing
echo "----------------------------------------"
echo "Step 1: Running layer-wise probing..."
echo "----------------------------------------"
echo ""

if [ -z "$CONDITIONS" ]; then
    # Run on all conditions
    python scripts/mechanistic_analysis/layer_wise_probing.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output_dir "$RESULTS_DIR" \
        --max_samples $MAX_SAMPLES
else
    # Run on specific conditions
    python scripts/mechanistic_analysis/layer_wise_probing.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --output_dir "$RESULTS_DIR" \
        --max_samples $MAX_SAMPLES \
        --conditions $CONDITIONS
fi

echo ""
echo "✓ Probing complete!"
echo ""

# Step 2: Create visualizations
echo "----------------------------------------"
echo "Step 2: Creating visualizations..."
echo "----------------------------------------"
echo ""

python scripts/mechanistic_analysis/visualize_layer_probing.py \
    --results_dir "$RESULTS_DIR" \
    --output_dir "$IMAGES_DIR"

echo ""
echo "✓ Visualizations complete!"
echo ""

# Step 3: Display summary
echo "============================================"
echo "ANALYSIS COMPLETE!"
echo "============================================"
echo ""
echo "Results saved to:"
echo "  • Statistics: $RESULTS_DIR/layer_statistics.csv"
echo "  • Detailed results: $RESULTS_DIR/detailed_layer_results.csv"
echo "  • Summary: $RESULTS_DIR/summary.json"
echo ""
echo "Visualizations saved to:"
echo "  • $IMAGES_DIR/layer_performance.png"
echo "  • $IMAGES_DIR/layer_trajectory.png"
echo "  • $IMAGES_DIR/layer_comparison.png"
echo "  • $IMAGES_DIR/layer_condition_heatmap.png"
echo ""
echo "Summary report:"
echo "  • $IMAGES_DIR/analysis_report.txt"
echo ""

# Display quick summary if summary.json exists
if [ -f "$RESULTS_DIR/summary.json" ]; then
    echo "Quick Summary:"
    echo "----------------------------------------"
    python3 << EOF
import json
with open('$RESULTS_DIR/summary.json', 'r') as f:
    summary = json.load(f)
print(f"Model: {summary['model']}")
print(f"Total layers analyzed: {summary['num_layers']}")
print(f"Examples processed: {summary['num_examples']}")
print(f"")
print(f"🎯 Best performing layer: Layer {summary['best_layer']}")
print(f"   Score: {summary['best_layer_score']:.4f}")
print(f"")
print("Top 5 layers:")
for i, layer_info in enumerate(summary['layer_scores'][:5], 1):
    print(f"  {i}. Layer {layer_info['layer']:2d}: "
          f"Score = {layer_info['mean_logprob_diff']:6.4f}, "
          f"Accuracy = {layer_info['accuracy']:.2%}")
EOF
    echo ""
fi

echo "============================================"
echo ""
echo "To view the visualizations:"
echo "  open $IMAGES_DIR/layer_performance.png"
echo ""
echo "To read the detailed report:"
echo "  cat $IMAGES_DIR/analysis_report.txt"
echo ""
