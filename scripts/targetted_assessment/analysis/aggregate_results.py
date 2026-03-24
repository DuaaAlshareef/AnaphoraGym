"""
Aggregate and analyze model performance results.

This script processes individual model result files and creates a summary
comparison table with accuracy per condition.
"""
import pandas as pd
import glob
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir, get_result_prefix, resolve_dataset_type


# Model size mapping for ordering (in millions of parameters)
MODEL_SIZES = {
    'gpt2': 124,
    'gpt2-medium': 355,
    'EleutherAI_pythia-410m-deduped': 410,
    'gpt2-large': 774,
    'meta-llama_Llama-3.2-1B': 1000,
    'lmsys_vicuna-7b-v1.5': 7000,
    'meta-llama_Llama-2-7b-chat-hf': 7000,
    'meta-llama_Meta-Llama-3.1-8B-Instruct': 8000,
    'lmsys_vicuna-13b-v1.3': 13000,
    'meta-llama_Llama-2-13b-hf': 13000,
    'meta-llama_Llama-2-7b-hf': 7000,
    'meta-llama_Meta-Llama-3-8B': 8000,
}


def analyze_model_performance(results_filename, group_col: str = "condition"):
    """
    Calculate accuracy per condition (or sub_cond) from a single results CSV file.

    Args:
        results_filename: Path to the results CSV file
        group_col: Column to group by — "condition" or "sub_cond"

    Returns:
        pd.DataFrame: DataFrame with group_col renamed to "condition" plus accuracy column,
                      or None if error.
    """
    try:
        df = pd.read_csv(results_filename)
        if df.empty or 'test_passed' not in df.columns:
            return None

        # Fall back to "condition" if requested column is absent
        if group_col not in df.columns:
            group_col = "condition"

        df_valid = df[df['test_passed'].isin([True, False])].copy()
        if df_valid.empty:
            return None

        df_valid['test_passed'] = df_valid['test_passed'].astype(bool)
        accuracy_df = df_valid.groupby(group_col)['test_passed'].mean().reset_index()
        # Always name the grouping column "condition" so downstream code is uniform
        accuracy_df = accuracy_df.rename(columns={group_col: 'condition', 'test_passed': 'accuracy'})
        accuracy_df['accuracy'] = (accuracy_df['accuracy'] * 100).round(2)
        return accuracy_df
    except FileNotFoundError:
        print(f"  - Error: Could not find '{results_filename}'")
        return None


def aggregate_all_results(dataset_type: str = "anaphoragym"):
    """
    Find all result files, analyze them, and create a comparison summary.
    
    Returns:
        tuple: (summary_df, sorted_model_names)
    """
    dataset_type = resolve_dataset_type(dataset_type)
    # Use sub_cond as grouping for subconditions dataset to show fine-grained results
    group_col = "sub_cond" if dataset_type == "subconditions" else "condition"
    results_dir = get_results_dir(dataset_type)
    prefix = get_result_prefix(dataset_type)
    search_pattern = os.path.join(results_dir, f"{prefix}*.csv")
    result_files = glob.glob(search_pattern)

    if not result_files:
        raise FileNotFoundError(f"No result files found in '{results_dir}'.")

    print(f"Found {len(result_files)} result files to compare.")
    
    found_model_names = [
        os.path.basename(f).split('Results_')[-1].replace('.csv', '')
        for f in result_files
    ]
    sorted_model_names = sorted(
        found_model_names,
        key=lambda name: MODEL_SIZES.get(name, float('inf'))
    )
    
    print(f"Processing models in this order: {sorted_model_names}")
    
    comparison_df = None
    for model_name in sorted_model_names:
        filepath = os.path.join(results_dir, f"{prefix}{model_name}.csv")
        print(f"Analyzing {model_name}...")
        accuracy_df = analyze_model_performance(filepath, group_col=group_col)
        
        if accuracy_df is not None:
            accuracy_df = accuracy_df.rename(columns={'accuracy': f'accuracy_{model_name}'})
            if comparison_df is None:
                comparison_df = accuracy_df
            else:
                comparison_df = pd.merge(comparison_df, accuracy_df, on='condition', how='outer')
    
    if comparison_df is None:
        raise ValueError("No valid data could be processed.")
    
    final_comparison_table = comparison_df.fillna(0)
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(final_comparison_table)
    
    # Save the summary table
    summary_path = os.path.join(results_dir, "model_comparison_summary.csv")
    final_comparison_table.to_csv(summary_path, index=False)
    print(f"\nComparison summary saved to '{summary_path}'")
    
    return final_comparison_table, sorted_model_names


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate model performance files.")
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="anaphoragym",
        choices=["anaphoragym", "subconditions"],
        help="Which dataset result bucket to aggregate.",
    )
    args = parser.parse_args()
    try:
        aggregate_all_results(dataset_type=args.dataset_type)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


