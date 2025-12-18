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
        # Go up: utils -> targeted_assessment -> scripts -> project_root
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    except NameError:
        # Fallback for interactive environments
        project_root = os.path.abspath('.')
    return project_root


def get_dataset_path():
    """Get the path to the AnaphoraGym dataset."""
    return os.path.join(get_project_root(), 'dataset', 'AnaphoraGym.csv')


def get_results_dir():
    """
    Get the path to the results directory.
    
    Note: Uses 'targeted_assessment' (correct spelling) but will also check
    for 'targetted_assessment' (old typo) for backward compatibility.
    """
    project_root = get_project_root()
    # Try the correct spelling first
    results_dir = os.path.join(project_root, 'results', 'targeted_assessment')
    # Check if old directory exists (for backward compatibility)
    old_dir = os.path.join(project_root, 'results', 'targetted_assessment')
    if os.path.exists(old_dir) and not os.path.exists(results_dir):
        results_dir = old_dir
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_images_dir():
    """Get the path to the images directory."""
    images_dir = os.path.join(get_project_root(), 'images')
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def get_mechanistic_results_dir():
    """Get the path to the mechanistic analysis results directory."""
    results_dir = os.path.join(get_project_root(), 'results', 'mechanistic_analysis')
    os.makedirs(results_dir, exist_ok=True)
    return results_dir

