# ==============================================================================
# SCRIPT TO COMPARE BASE VS. INSTRUCTION-TUNED MODELS
#
# This script:
#   1. Automatically finds all result files.
#   2. Classifies each model as 'Base' or 'Instruction-Tuned'.
#   3. Calculates the average accuracy for each group on each task.
#   4. Generates a clean bar chart directly comparing the two groups.
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# --- 1. DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def main():
    """Finds all result files, classifies models, and creates a comparison chart."""
    
    # ================== MODEL CLASSIFICATION ==================
    # Dictionary mapping model names (as in filenames) to their type.
    # Add any new models you test to this dictionary.
    model_types = {
        'gpt2': 'Base',
        'gpt2-medium': 'Base',
        'gpt2-large': 'Base',
        'EleutherAI_pythia-410m-deduped': 'Base',
        'meta-llama_Llama-2-7b-hf': 'Base',
        'meta-llama_Llama-2-13b-hf': 'Base',
        'meta-llama_Meta-Llama-3-8B': 'Base',
        'meta-llama_Llama-2-7b-chat-hf': 'Instruction-Tuned',
        'meta-llama_Meta-Llama-3-8B-Instruct': 'Instruction-Tuned',
        'lmsys_vicuna-7b-v1.5': 'Instruction-Tuned',
        'lmsys_vicuna-13b-v1.3': 'Instruction-Tuned'
    }
    # ==========================================================

    search_pattern = os.path.join(RESULTS_DIR, "AnaphoraGym_Results_*.csv")
    result_files = glob.glob(search_pattern)

    if not result_files:
        print(f"Analysis failed: No result files found in '{RESULTS_DIR}'.")
        return

    print(f"Found {len(result_files)} result files to consolidate.")
    
    all_results_dfs = []
    for filepath in result_files:
        df = pd.read_csv(filepath)
        all_results_dfs.append(df)

    master_df = pd.concat(all_results_dfs, ignore_index=True)
    
    # --- Add the 'model_type' column based on our dictionary ---
    # Sanitize the 'model_source' column to match our dictionary keys
    master_df['model_key'] = master_df['model_source'].str.replace('/', '_')
    master_df['model_type'] = master_df['model_key'].map(model_types)

    # --- Calculate Average Accuracy for Each Group ---
    # Group by both condition and model_type, then calculate the mean accuracy
    group_summary = master_df.groupby(['condition', 'model_type'])['test_passed'].mean().reset_index()
    group_summary['accuracy'] = (group_summary['test_passed'] * 100).round(2)
    
    print("\n--- AVERAGE PERFORMANCE: BASE VS. INSTRUCTION-TUNED ---")
    print(group_summary)
    
    # --- Generate the Comparison Chart ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Use a simple, high-contrast two-color palette
    palette = {"Base": "steelblue", "Instruction-Tuned": "darkorange"}

    sns.barplot(data=group_summary, x='condition', y='accuracy', hue='model_type', ax=ax, palette=palette)

    # --- Customization ---
    ax.set_title('Base vs. Instruction-Tuned Model Performance on AnaphoraGym', fontsize=16, pad=20, weight='bold')
    ax.set_xlabel('Linguistic Condition', fontsize=12, weight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, weight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=11)
    plt.yticks(np.arange(0, 101, 10), fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    ax.legend(title='Model Type', title_fontsize='12', fontsize=11)
    plt.tight_layout()

    # --- Save the Chart ---
    output_path = os.path.join(IMAGES_DIR, "base_vs_instruct_comparison_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to '{output_path}'")

if __name__ == "__main__":
    main()