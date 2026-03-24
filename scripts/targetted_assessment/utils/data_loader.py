"""
Shared data loading utilities for the AnaphoraGym project.
"""
import pandas as pd
import os
from typing import Optional
from .paths import (
    get_dataset_path,
    get_subconditions_dataset_path,
    detect_dataset_type_from_path,
    get_results_dir,
    get_result_prefix,
    resolve_dataset_type,
)


def load_dataset(dataset_path: Optional[str] = None, dataset_type: Optional[str] = None) -> pd.DataFrame:
    """
    Load the dataset.

    Path resolution order:
      1. Use explicit dataset_path if given.
      2. If dataset_type is given, pick the matching default CSV.
      3. If neither, fall back to AnaphoraGym.csv.

    Raises:
        FileNotFoundError: If the dataset file is not found
    """
    if dataset_path is None:
        if dataset_type is not None:
            dtype = resolve_dataset_type(dataset_type)
            dataset_path = (
                get_subconditions_dataset_path() if dtype == "subconditions"
                else get_dataset_path()
            )
        else:
            dataset_path = get_dataset_path()
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    return pd.read_csv(dataset_path)


def load_model_results(model_name: Optional[str] = None, dataset_type: str = None) -> pd.DataFrame:
    """
    Load results for a specific model or all models.
    
    Args:
        model_name: If provided, load results for this specific model.
                   If None, load all model results.
    
    Returns:
        pd.DataFrame: The loaded results
    """
    dataset_type = resolve_dataset_type(dataset_type)
    results_dir = get_results_dir(dataset_type)
    prefix = get_result_prefix(dataset_type)
    
    if model_name:
        # Load specific model results
        safe_model_name = model_name.replace('/', '_')
        filepath = os.path.join(results_dir, f"{prefix}{safe_model_name}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Results not found for model: {model_name}")
        return pd.read_csv(filepath)
    else:
        # Load all model results
        import glob
        pattern = os.path.join(results_dir, f"{prefix}*.csv")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No result files found in: {results_dir}")
        
        dfs = []
        for filepath in sorted(files):
            # Skip aggregated files
            filename = os.path.basename(filepath)
            if "Enriched" in filename or "summary" in filename.lower() or "Concatenated" in filename:
                continue
            dfs.append(pd.read_csv(filepath))
        
        if not dfs:
            raise ValueError("No valid result files found")
        
        return pd.concat(dfs, ignore_index=True)


def load_summary_results(dataset_type: str = None) -> pd.DataFrame:
    """
    Load the model comparison summary CSV.
    
    Returns:
        pd.DataFrame: The summary results
    """
    results_dir = get_results_dir(dataset_type)
    summary_path = os.path.join(results_dir, "model_comparison_summary.csv")
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"Summary file not found at: {summary_path}\n"
            "Please run the analysis script first."
        )
    
    return pd.read_csv(summary_path)


