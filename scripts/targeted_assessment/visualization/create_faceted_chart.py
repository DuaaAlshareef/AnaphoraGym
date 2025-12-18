"""
Create a faceted (small multiples) bar chart showing model performance by condition.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_images_dir, load_summary_results

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


def create_faceted_chart(summary_df=None, model_order=None):
    """
    Generate a faceted bar chart with one subplot per condition.
    
    Args:
        summary_df: Summary DataFrame (if None, loads from file)
        model_order: List of model names in desired order (if None, infers from data)
    """
    if summary_df is None:
        summary_df = load_summary_results()
    
    if model_order is None:
        model_order = [
            col.replace('accuracy_', '')
            for col in summary_df.columns
            if 'accuracy_' in col
        ]
        model_order = sorted(
            model_order,
            key=lambda name: MODEL_SIZES.get(name, float('inf'))
        )
    
    # Prepare data
    id_vars = ['condition']
    value_vars = [
        f'accuracy_{model}'
        for model in model_order
        if f'accuracy_{model}' in summary_df.columns
    ]
    
    df_long = pd.melt(
        summary_df,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='model_name',
        value_name='accuracy'
    )
    
    df_long['model_name'] = (
        df_long['model_name']
        .str.replace('accuracy_', '')
        .str.replace('_', '/')
    )
    
    cleaned_model_order = [name.replace('_', '/') for name in model_order]
    
    # Create faceted plot
    g = sns.FacetGrid(
        df_long,
        col="condition",
        col_wrap=3,
        height=5,
        aspect=1.2,
        sharex=False,
        sharey=True
    )
    
    g.map_dataframe(
        sns.barplot,
        x="model_name",
        y="accuracy",
        hue="model_name",
        order=cleaned_model_order,
        palette="deep",
        dodge=False
    )
    
    # Customization
    g.fig.suptitle(
        'Model Performance by Linguistic Condition',
        y=1.03,
        fontsize=16,
        weight='bold'
    )
    g.set_axis_labels("", "Accuracy (%)")
    g.set_titles("Condition: {col_name}")
    g.set(ylim=(0, 105))
    
    # Handle labels - only show on bottom row
    axes_2d = np.atleast_2d(g.axes)
    n_rows, n_cols = axes_2d.shape
    
    for r, ax_row in enumerate(axes_2d):
        for ax in ax_row:
            if ax is None:
                continue
            
            # Hide legend from each subplot
            if ax.get_legend():
                ax.get_legend().remove()
            
            if r == n_rows - 1:
                # Bottom row: show tick labels
                ax.tick_params(axis='x', labelrotation=90, labelsize=8)
            else:
                # Other rows: hide tick labels
                for tick in ax.get_xticklabels():
                    tick.set_visible(False)
                ax.set_xlabel("")
    
    # Create shared legend
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = []
    ordered_labels = []
    for model_name in cleaned_model_order:
        try:
            idx = labels.index(model_name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
        except ValueError:
            pass
    
    g.fig.legend(
        ordered_handles,
        ordered_labels,
        loc='lower center',
        ncol=min(len(ordered_labels), 6),
        bbox_to_anchor=(0.5, 0.05),
        title="Model",
        fontsize=8,
        title_fontsize=9
    )
    
    plt.subplots_adjust(bottom=0.35, top=0.9, hspace=0.4)
    
    # Save
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "model_comparison_faceted.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFaceted chart saved as '{output_path}'")
    plt.close(g.fig)


if __name__ == "__main__":
    create_faceted_chart()

