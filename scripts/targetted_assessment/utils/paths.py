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


def get_results_dir():
    """Get the path to the targeted assessment results directory."""
    results_dir = os.path.join(get_project_root(), 'results', 'targetted_assessment')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_images_dir():
    """
    Get the path used for generated plots.

    Plots are now stored directly under results/targetted_assessment
    (no nested images directory).
    """
    return get_results_dir()


def get_mechanistic_results_dir():
    """Get the path to the mechanistic analysis results directory."""
    results_dir = os.path.join(get_project_root(), 'results', 'mechanistic_analysis')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

