"""
Shared path utilities for the AnaphoraGym project.

This module provides a centralized way to handle project paths,
eliminating duplication across scripts.
"""
import os


def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        str: Absolute path to the project root
    """
    try:
        # Try to get the path relative to this file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up: utils -> targetted_assessment -> scripts -> project_root
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    except NameError:
        # Fallback for interactive environments
        project_root = os.path.abspath('.')
    return project_root


def get_dataset_path():
    """Get the path to the AnaphoraGym dataset."""
    return os.path.join(get_project_root(), 'dataset', 'AnaphoraGym.csv')


def get_subconditions_dataset_path():
    """Get the path to the AnaphoraGym subconditions dataset."""
    return os.path.join(get_project_root(), 'dataset', 'AnaphoraGym_Subconditions.csv')


def detect_dataset_type_from_path(dataset_path: str) -> str:
    """
    Infer dataset type from the filename.

    Rules (case-insensitive):
      - filename contains "subconditions" → "subconditions"
      - anything else                     → "anaphoragym"

    This means the user only needs to change DATASET_PATH in run_all.sh
    and the whole pipeline adapts automatically.
    """
    filename = os.path.basename(dataset_path).lower()
    return "subconditions" if "subconditions" in filename else "anaphoragym"


def resolve_dataset_type(dataset_type: str = None) -> str:
    """
    Normalize dataset type from arg/env.

    Supported:
      - anaphoragym
      - subconditions
    """
    if dataset_type is None:
        dataset_type = os.getenv("TARGETTED_DATASET_TYPE", "anaphoragym")
    dataset_type = str(dataset_type).strip().lower()
    if dataset_type not in {"anaphoragym", "subconditions"}:
        raise ValueError(
            f"Invalid dataset_type '{dataset_type}'. Use 'anaphoragym' or 'subconditions'."
        )
    return dataset_type


def get_result_prefix(dataset_type: str = None) -> str:
    """Return the file prefix for result CSVs by dataset type."""
    dtype = resolve_dataset_type(dataset_type)
    return "Subconditions_Results_" if dtype == "subconditions" else "AnaphoraGym_Results_"


def get_results_root_dir():
    """Get the root targeted assessment results directory."""
    root_dir = os.path.join(get_project_root(), 'results', 'targetted_assessment')
    os.makedirs(root_dir, exist_ok=True)
    return root_dir


def get_results_dir(dataset_type: str = None):
    """
    Get dataset-specific CSV results directory.

    - anaphoragym -> results/targetted_assessment/anaphoragym_csv
    - subconditions -> results/targetted_assessment/subconditions
    """
    dtype = resolve_dataset_type(dataset_type)
    subdir = "subconditions" if dtype == "subconditions" else "anaphoragym_csv"
    results_dir = os.path.join(get_results_root_dir(), subdir)
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_images_dir():
    """
    Get the path used for generated plots.

    Plots are stored in a shared images folder:
      results/targetted_assessment/images
    """
    images_dir = os.path.join(get_results_root_dir(), "images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def get_mechanistic_results_dir():
    """Get the path to the mechanistic analysis results directory."""
    results_dir = os.path.join(get_project_root(), 'results', 'mechanistic_analysis')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

