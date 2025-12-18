#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=200gb
#SBATCH --gres=gpu:A100:1


module load devel/miniforge
conda activate virtual_env
echo $(which python)

# This script orchestrates the entire targeted assessment pipeline.
# It should be run from the project's root directory.

echo "--- Starting Full AnaphoraGym Targeted Assessment ---"
# --- Define Paths to Scripts ---

EXPERIMENT_SCRIPT_PATH="scripts/targetted_assessment/test_anaphoragym.py"
ANALYZE_RESULTS_PATH="scripts/targetted_assessment/count_tests.py"
CREATE_ENRICHED_DATASET_PATH="scripts/targetted_assessment/concatenate_outputs.py"
COMPARE_TYPES_PATH="scripts/targetted_assessment/compare_model_types.py"
CREATE_FACETED_CHART_PATH="scripts/targetted_assessment/create_faceted_chart.py"
COMPARE_TUNING_PAIRS_PATH="scripts/targetted_assessment/compare_tuning_pairs.py"
CREATE_HEATMAP_PATH="scripts/targetted_assessment/create_heatmap.py"

# --- Experiment Configuration ---
MODELS_TO_TEST=(
  # "gpt2"
  # "gpt2-medium"
  # "gpt2-large"
  # "EleutherAI/pythia-410m-deduped"
  # "meta-llama/Llama-3.2-1B"
  # "meta-llama/Llama-2-7b-hf"
  # "meta-llama/Llama-2-7b-chat-hf"
  # "meta-llama/Llama-2-13b-hf"
  # "meta-llama/Llama-3-8B"
  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
  # "lmsys/vicuna-7b-v1.5"
  # "lmsys/vicuna-13b-v1.3"
  "mistralai/Mistral-7B-Instruct-v0.3"
  )

# --- 1. Run the Experiments ---
echo "--- Starting Full AnaphoraGym Targeted Assessment ---"
for model_name in "${MODELS_TO_TEST[@]}"
do
  echo ""
  echo "============================================================="
  echo "=> Running experiment for model: $model_name"
  echo "============================================================="
  python3 "$EXPERIMENT_SCRIPT_PATH" --model "$model_name"
done

# --- 2. Run All Analysis and Plotting Scripts ---
echo ""
echo "--- All model experiments are complete. ---"
echo ""

echo "============================================================="
echo "=> Running main analysis and creating primary bar chart..."
echo "============================================================="
python3 "$ANALYZE_RESULTS_PATH"

echo ""
echo "============================================================="
echo "=> Creating the consolidated, enriched dataset CSV..."
echo "============================================================="
# python3 "$CREATE_ENRICHED_DATASET_PATH"

# echo ""
# echo "============================================================="
# echo "=> Creating the 'Base vs. Instruction-Tuned' comparison chart..."
# echo "============================================================="
# python3 "$COMPARE_TYPES_PATH"

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
# echo "--- Pipeline Complete ---"
