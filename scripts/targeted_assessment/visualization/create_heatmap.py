"""
Create a heatmap showing model performance across conditions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_images_dir, load_summary_results


def create_heatmap(summary_df=None):
    """
    Generate a heatmap visualization of model performance.
    
    Args:
        summary_df: Summary DataFrame (if None, loads from file)
    """
    if summary_df is None:
        summary_df = load_summary_results()
    
    # Prepare data
    df = summary_df.set_index('condition')
    df.columns = df.columns.str.replace('accuracy_', '').str.replace('_', '/')
    df_transposed = df.transpose()
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(
        df_transposed,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title(
        'Model Performance Heatmap on AnaphoraGym',
        fontsize=16,
        pad=20,
        weight='bold'
    )
    ax.set_xlabel('Linguistic Condition', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.xticks(rotation=40, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "model_comparison_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap chart saved as '{output_path}'")
    plt.close()


if __name__ == "__main__":
    create_heatmap()

