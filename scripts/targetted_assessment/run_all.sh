#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --time=04:00:00
#SBATCH --mem=20gb
#SBATCH --gres=gpu:1

module load devel/miniforge
conda activate virtual_env
echo $(which python)

# Pin HuggingFace cache to home dir so weights are reused across jobs
export HF_HOME=~/.cache/huggingface

# This script orchestrates the entire targeted assessment pipeline.
# It should be run from the project's root directory.

echo "--- Starting Targeted Assessment ---"

# ============================================================
# DATASET CONFIGURATION — only change this line to switch datasets:
#   dataset/AnaphoraGym.csv            → main AnaphoraGym benchmark
#   dataset/AnaphoraGym_Subconditions.csv → subconditions analysis
# ============================================================
DATASET_PATH="dataset/AnaphoraGym.csv"
# DATASET_PATH="dataset/AnaphoraGym_Subconditions.csv"

# Auto-detect dataset type from filename (no manual change needed)
if [[ "$DATASET_PATH" == *"Subconditions"* ]]; then
  DATASET_TYPE="subconditions"
else
  DATASET_TYPE="anaphoragym"
fi
echo "=> Dataset: $DATASET_PATH  (type: $DATASET_TYPE)"

# --- Define Paths to Scripts ---
EXPERIMENT_SCRIPT_PATH="scripts/targetted_assessment/experiments/run_experiment.py"
VISUALIZE_SINGLE_MODEL_PATH="scripts/targetted_assessment/visualization/visualize_single_model.py"
ANALYZE_RESULTS_PATH="scripts/targetted_assessment/analysis/aggregate_results.py"
CREATE_ENRICHED_DATASET_PATH="scripts/targetted_assessment/data/create_enriched_dataset.py"
COMPARE_TYPES_PATH="scripts/targetted_assessment/analysis/compare_model_types.py"
CREATE_FACETED_CHART_PATH="scripts/targetted_assessment/visualization/create_faceted_chart.py"
CREATE_BAR_CHART_PATH="scripts/targetted_assessment/visualization/create_bar_chart.py"
COMPARE_TUNING_PAIRS_PATH="scripts/targetted_assessment/analysis/compare_tuning_pairs.py"
CREATE_HEATMAP_PATH="scripts/targetted_assessment/visualization/create_heatmap.py"
CREATE_RADAR_PATH="scripts/targetted_assessment/visualization/create_radar_chart.py"
CREATE_COMPARISON_CHARTS_PATH="scripts/targetted_assessment/visualization/create_comparison_charts.py"

# --- Experiment Configuration ---
MODELS_TO_TEST=(
  # Qwen2.5 Family
  "Qwen/Qwen2.5-0.5B"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-72B"
  "Qwen/Qwen2.5-72B-Instruct"
  # Olmo3 Family
  # "allenai/Olmo-3-7B-Think"
  # "allenai/OLMo-3-7B-Instruct"
  # "allenai/Olmo-3-1125-32B"
  # "allenai/Olmo-3-32B-Think"
  #Gemma2 Family
  # "google/gemma-2-2b"
  # "google/gemma-2-2b-it"
  # "google/gemma-2-9b"
  # "google/gemma-2-9b-it"
  # "google/gemma-2-27b"
  # "google/gemma-2-27b-it"
  )

# --- 1. Run the Experiments ---
echo "--- Starting $DATASET_TYPE Targeted Assessment ---"
for model_name in "${MODELS_TO_TEST[@]}"
do
  echo ""
  echo "============================================================="
  echo "=> Running experiment for model: $model_name"
  echo "============================================================="
  python3 "$EXPERIMENT_SCRIPT_PATH" --model "$model_name" --dataset "$DATASET_PATH" --dataset-type "$DATASET_TYPE"

  echo ""
  echo "=> Creating visualization for $model_name"
  python3 "$VISUALIZE_SINGLE_MODEL_PATH" --model "$model_name" --dataset-type "$DATASET_TYPE"
done

# --- 2. Run All Analysis and Plotting Scripts ---
echo ""
echo "--- All model experiments are complete. ---"
echo ""

echo "============================================================="
echo "=> Running main analysis and creating primary bar chart..."
echo "============================================================="
python3 "$ANALYZE_RESULTS_PATH" --dataset-type "$DATASET_TYPE"
python3 "$CREATE_BAR_CHART_PATH" --dataset-type "$DATASET_TYPE"

# echo ""
# echo "============================================================="
# echo "=> Creating the consolidated, enriched dataset CSV..."
# echo "============================================================="
# python3 "$CREATE_ENRICHED_DATASET_PATH"

# echo ""
# echo "============================================================="
# echo "=> Creating the 'Base vs. Instruction-Tuned' comparison chart..."
# echo "============================================================="
# python3 "$COMPARE_TYPES_PATH"
# python3 "$CREATE_COMPARISON_CHARTS_PATH"

# echo ""
# echo "============================================================="
# echo "=> Creating the faceted (small multiples) chart..."
# echo "============================================================="
# python3 "$CREATE_FACETED_CHART_PATH"

# echo ""
# echo "============================================================="
# echo "=> Creating the compare tuned pairs chart..."
# echo "============================================================="
# python3 "$COMPARE_TUNING_PAIRS_PATH"

# echo ""
# echo "============================================================="
# echo "=> Creating the heatmap chart..."
# echo "============================================================="
# python3 "$CREATE_HEATMAP_PATH"

# echo ""
# echo "============================================================="
# echo "=> Creating the radar chart..."
# echo "============================================================="
# python3 "$CREATE_RADAR_PATH"

echo ""
echo "--- Pipeline Complete ---"

exit 0

