"""
Compare paired base vs. instruction-tuned models.

This script performs head-to-head comparisons of model pairs
from the same family (e.g., Llama-2-7b base vs. chat).
"""
import pandas as pd
import glob
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir


# Define model pairs for comparison
MODEL_PAIRS = [
    ('meta-llama_Llama-2-7b-hf', 'meta-llama_Llama-2-7b-chat-hf'),
    ('meta-llama_Llama-2-13b-hf', 'meta-llama_Llama-2-13b-chat-hf'),
    # Add more pairs as needed
]


def compare_tuning_pairs():
    """
    Compare performance of paired base and instruction-tuned models.
    
    Returns:
        pd.DataFrame: Comparison results
    """
    results_dir = get_results_dir()
    search_pattern = os.path.join(results_dir, "AnaphoraGym_Results_*.csv")
    result_files = glob.glob(search_pattern)

    if not result_files:
        raise FileNotFoundError(f"No result files found in '{results_dir}'.")

    # Load all results
    all_results_dfs = []
    for filepath in result_files:
        df = pd.read_csv(filepath)
        all_results_dfs.append(df)

    master_df = pd.concat(all_results_dfs, ignore_index=True)
    
    # Process each pair
    comparison_results = []
    for base_model, tuned_model in MODEL_PAIRS:
        base_df = master_df[master_df['model_source'].str.replace('/', '_') == base_model]
        tuned_df = master_df[master_df['model_source'].str.replace('/', '_') == tuned_model]
        
        if base_df.empty or tuned_df.empty:
            print(f"Skipping pair ({base_model}, {tuned_model}) - missing data")
            continue
        
        # Calculate accuracy per condition for each model
        base_acc = base_df.groupby('condition')['test_passed'].mean() * 100
        tuned_acc = tuned_df.groupby('condition')['test_passed'].mean() * 100
        
        for condition in base_acc.index:
            comparison_results.append({
                'condition': condition,
                'base_model': base_model,
                'tuned_model': tuned_model,
                'base_accuracy': round(base_acc[condition], 2),
                'tuned_accuracy': round(tuned_acc[condition], 2),
                'improvement': round(tuned_acc[condition] - base_acc[condition], 2)
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    if not comparison_df.empty:
        print("\n--- PAIRED MODEL COMPARISON ---")
        print(comparison_df)
    
    return comparison_df


if __name__ == "__main__":
    try:
        compare_tuning_pairs()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

