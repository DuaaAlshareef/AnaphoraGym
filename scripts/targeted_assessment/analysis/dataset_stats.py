"""
Calculate dataset statistics and overview.

This script analyzes the AnaphoraGym dataset to provide
statistics about conditions, items, and tests.
"""
import pandas as pd
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_dataset


def calculate_dataset_stats():
    """
    Calculate and return dataset statistics.
    
    Returns:
        pd.DataFrame: Summary statistics per condition
    """
    df = load_dataset()
    
    # Ensure n_tests is numeric
    df['n_tests'] = pd.to_numeric(df['n_tests'], errors='coerce')
    df.dropna(subset=['n_tests'], inplace=True)
    
    # Calculate statistics per condition
    samples_per_condition = df.groupby('condition')['item'].nunique().reset_index()
    samples_per_condition.rename(columns={'item': 'num_samples'}, inplace=True)
    
    tests_per_condition = df.groupby('condition')['n_tests'].sum().reset_index()
    tests_per_condition.rename(columns={'n_tests': 'total_tests'}, inplace=True)
    
    summary_df = pd.merge(samples_per_condition, tests_per_condition, on='condition')
    summary_df.sort_values('condition', inplace=True)
    
    print("\nDataset Statistics:")
    print(summary_df)
    
    return summary_df


if __name__ == "__main__":
    try:
        calculate_dataset_stats()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

