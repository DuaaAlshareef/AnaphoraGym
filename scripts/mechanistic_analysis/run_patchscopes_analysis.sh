#!/bin/bash
#SBATCH --partition=gpu-single
#SBATCH --tasks=1
#SBATCH --time=0:40:00
#SBATCH --mem=128gb
#SBATCH --gres=gpu:A100:1


module load devel/miniforge
conda activate virtual_env
echo $(which python)

# --- Environment Setup ---
echo "--- Starting SLURM Job for Patchscopes Analysis ---"
echo "Job running on node: $(hostname)"
echo "Job started at: $(date)"

# --- Script Execution ---
echo ""
echo "============================================================="
echo "=> Executing the Patchscopes Python script..."
echo "============================================================="

# Run the Python script. The Python script itself handles all the
# details like which model to use and where to save the results.
python3 scripts/mechanistic_analysis/patchscopes_w.py

echo ""
echo "--- SLURM Job Complete ---"
echo "Job finished at: $(date)"