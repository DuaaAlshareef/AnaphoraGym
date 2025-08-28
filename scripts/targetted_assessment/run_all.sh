#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --time=00:05:00
#SBATCH --mem=2gb
#SBATCH --gres=gpu:A40:1


module load devel/miniforge
conda activate anaphoragym_env
echo $(which python)



# This script orchestrates the entire targeted assessment pipeline.
# It should be run from the project's root directory.

echo "--- Starting Full AnaphoraGym Targeted Assessment ---"

# --- Define Paths to Scripts ---
EXPERIMENT_SCRIPT_PATH="scripts/targetted_assessment/test_anaphoragym.py"
ANALYSIS_SCRIPT_PATH="scripts/targetted_assessment/analyze_results.py"

# --- Define Experiment Configuration ---
# This list now contains CORRECT and VALID Hugging Face model identifiers.
# I have included Llama-3-8B as a powerful, state-of-the-art option.
MODELS_TO_TEST=(
  "gpt2"
  "gpt2-medium"
  "gpt2-large"
  "EleutherAI/pythia-410m-deduped"
  # "meta-llama/Meta-Llama-3-8B" 
)

# --- Run Experiments in a Loop ---
for model_name in "${MODELS_TO_TEST[@]}"
do
  echo ""
  echo "============================================================="
  echo "=> Running experiment for model: $model_name"
  echo "============================================================="
  
  # Call the Python script using its full path from the root.
  python3 "$EXPERIMENT_SCRIPT_PATH" --model "$model_name"
done

echo ""
echo "--- All model experiments are complete. ---"
echo "--- Now running the final analysis and plotting... ---"

# Call the analysis script.
python3 "$ANALYSIS_SCRIPT_PATH"

echo ""
echo "--- Pipeline Complete ---"