#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --time=0:20:00
#SBATCH --mem=100gb
#SBATCH --gres=gpu:A100:1


module load devel/miniforge
conda activate virtual_env
echo $(which python)

# --- Environment Setup ---
echo "--- Starting SLURM Job for Patchscopes Analysis ---"
echo "Job running on node: $(hostname)"
echo "Job started at: $(date)"

echo "--- Starting Full AnaphoraGym Patchscopes Analysis ---"

# --- Define Paths to Script ---
# The path is from the root to your python script
PATCHSCOPE_SCRIPT_PATH="scripts/mechanistic_analysis/patchscopes_w_models.py"

# --- Define Experiment Configuration ---
# Add or remove any model from this list to change the experiment.
MODELS_TO_TEST=(
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  # "meta-llama/Llama-2-7b-chat-hf"
  # "EleutherAI/gpt-j-6b"
  # "EleutherAI/pythia-12b"
  # "meta-llama/Llama-2-7b-chat-hf"
  # NOTE: Llama-2-7b-chat is a gated model. Ensure you are logged in.
  # "meta-llama/Llama-2-13b-hf"
)

# --- Run Experiments in a Loop ---
for model_name in "${MODELS_TO_TEST[@]}"
do
  echo ""
  echo "============================================================="
  echo "=> Now running Patchscopes for model: $model_name"
  echo "============================================================="
  
  python3 "$PATCHSCOPE_SCRIPT_PATH" --model "$model_name"
done

echo ""
echo "--- All Patchscopes experiments are complete. ---"