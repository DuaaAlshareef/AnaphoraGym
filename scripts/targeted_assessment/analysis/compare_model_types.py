"""
Compare base vs. instruction-tuned model performance.

This script classifies models and compares average performance
between base and instruction-tuned variants.
"""
import pandas as pd
import glob
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir


# Model classification dictionary
MODEL_TYPES = {
    'gpt2': 'Base',
    'gpt2-medium': 'Base',
    'gpt2-large': 'Base',
    'EleutherAI_pythia-410m-deduped': 'Base',
    'meta-llama_Llama-2-7b-hf': 'Base',
    'meta-llama_Llama-2-13b-hf': 'Base',
    'meta-llama_Meta-Llama-3-8B': 'Base',
    'meta-llama_Llama-2-7b-chat-hf': 'Instruction-Tuned',
    'meta-llama_Meta-Llama-3-8B-Instruct': 'Instruction-Tuned',
    'meta-llama_Meta-Llama-3.1-8B-Instruct': 'Instruction-Tuned',
    'lmsys_vicuna-7b-v1.5': 'Instruction-Tuned',
    'lmsys_vicuna-13b-v1.3': 'Instruction-Tuned',
}


def compare_model_types():
    """
    Compare average performance of base vs. instruction-tuned models.
    
    Returns:
        pd.DataFrame: Summary with condition, model_type, and accuracy
    """
    results_dir = get_results_dir()
    search_pattern = os.path.join(results_dir, "AnaphoraGym_Results_*.csv")
    result_files = glob.glob(search_pattern)

    if not result_files:
        raise FileNotFoundError(f"No result files found in '{results_dir}'.")

    print(f"Found {len(result_files)} result files to consolidate.")
    
    all_results_dfs = []
    for filepath in result_files:
        df = pd.read_csv(filepath)
        all_results_dfs.append(df)

    master_df = pd.concat(all_results_dfs, ignore_index=True)
    
    # Add model_type column
    master_df['model_key'] = master_df['model_source'].str.replace('/', '_')
    master_df['model_type'] = master_df['model_key'].map(MODEL_TYPES)

    # Calculate average accuracy for each group
    group_summary = master_df.groupby(['condition', 'model_type'])['test_passed'].mean().reset_index()
    group_summary['accuracy'] = (group_summary['test_passed'] * 100).round(2)
    
    print("\n--- AVERAGE PERFORMANCE: BASE VS. INSTRUCTION-TUNED ---")
    print(group_summary)
    
    return group_summary


if __name__ == "__main__":
    try:
        compare_model_types()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

