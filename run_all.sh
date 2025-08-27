#!/bin/bash

# This script orchestrates the entire targeted assessment pipeline.
# It should be run from the project's root directory.

echo "--- Starting Full AnaphoraGym Targeted Assessment ---"

# --- Define Paths to Scripts ---
# Using the correct paths and filenames as specified.
EXPERIMENT_SCRIPT_PATH="scripts/targetted_assessment/test_anaphoragym.py"
ANALYSIS_SCRIPT_PATH="scripts/targetted_assessment/analyze_results.py"

# --- Define Experiment Configuration ---
# The list of models to test. Add or remove from this list to change the experiment.
MODELS_TO_TEST=(
  "gpt2"
  "gpt2-medium"
  "gpt2-large"
  "EleutherAI/pythia-410m-deduped"
  "meta-llama_Llama-3.2-1B"
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

# Call the analysis script using its correct name. It will automatically
# find all the result CSVs that were just created.
python3 "$ANALYSIS_SCRIPT_PATH"

echo ""
echo "--- Pipeline Complete ---"