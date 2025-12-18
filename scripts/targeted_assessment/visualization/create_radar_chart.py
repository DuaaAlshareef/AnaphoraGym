"""
Create a radar chart showing model skill profiles across conditions.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_images_dir, load_summary_results


def create_radar_chart(summary_df=None):
    """
    Generate a radar chart showing model performance profiles.
    
    Args:
        summary_df: Summary DataFrame (if None, loads from file)
    """
    if summary_df is None:
        summary_df = load_summary_results()
    
    # Prepare data
    labels = summary_df['condition'].values
    num_vars = len(labels)
    
    # Calculate angles for radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.rcParams['font.family'] = 'serif'
    
    # Plot each model
    model_cols = [col for col in summary_df.columns if 'accuracy_' in col]
    for col in model_cols:
        model_name = col.replace('accuracy_', '').replace('_', '/')
        values = summary_df[col].values.tolist()
        values += values[:1]  # Close the circle
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.2)
    
    # Customization
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    
    ax.set_rlabel_position(0)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="grey", size=9)
    ax.set_ylim(0, 105)
    
    ax.set_title(
        'Model Skill Profiles on AnaphoraGym',
        size=16,
        color='black',
        y=1.12
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
    
    # Save
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "model_comparison_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nRadar chart saved as '{output_path}'")
    plt.close()


if __name__ == "__main__":
    create_radar_chart()

