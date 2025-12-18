"""
Concatenate individual model result files into a single consolidated CSV.
"""
import pandas as pd
import glob
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir


def concatenate_model_results():
    """
    Concatenate all individual model result files into one CSV.
    
    Returns:
        pd.DataFrame: The concatenated results
    """
    results_dir = get_results_dir()
    print(f"--- Starting concatenation of individual model results ---")
    print(f"Searching for result files in: {results_dir}")

    # Find all individual model result files
    search_pattern = os.path.join(results_dir, "AnaphoraGym_Results_*.csv")
    all_files = glob.glob(search_pattern)

    # Filter out aggregated results
    model_result_files = [
        f for f in all_files
        if "Enriched" not in os.path.basename(f)
        and "summary" not in os.path.basename(f).lower()
        and "Concatenated" not in os.path.basename(f)
    ]

    if not model_result_files:
        raise FileNotFoundError(
            f"No individual model result files found in '{results_dir}'.\n"
            "Please ensure your 'AnaphoraGym_Results_modelname.csv' files are present."
        )

    print(f"\nFound {len(model_result_files)} individual model result files to concatenate:")
    
    # Read and consolidate all result files
    all_results_dfs = []
    for filepath in model_result_files:
        filename = os.path.basename(filepath)
        print(f"  - Reading {filename}")
        try:
            df = pd.read_csv(filepath)
            all_results_dfs.append(df)
        except Exception as e:
            print(f"    [ERROR] Could not read {filename}. Reason: {e}")
            continue

    if not all_results_dfs:
        raise ValueError("No dataframes were successfully loaded.")

    # Concatenate all DataFrames
    concatenated_df = pd.concat(all_results_dfs, ignore_index=True)
    print(f"\nSuccessfully consolidated {len(concatenated_df)} rows from all files.")

    # Save the concatenated CSV
    output_filename = "AnaphoraGym_All_Model_Results_Concatenated.csv"
    output_path = os.path.join(results_dir, output_filename)
    
    concatenated_df.to_csv(output_path, index=False)
    print(f"\nSUCCESS: All individual model results concatenated into '{output_path}'")
    print("\nPreview of the concatenated file (first 5 rows):")
    print(concatenated_df.head())
    print(f"\nTotal unique models in concatenated file: {concatenated_df['model_source'].nunique()}")
    
    return concatenated_df


if __name__ == "__main__":
    try:
        concatenate_model_results()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

