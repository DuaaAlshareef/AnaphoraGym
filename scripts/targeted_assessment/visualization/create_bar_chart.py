"""
Create a grouped bar chart comparing model performance across conditions.
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


def create_bar_chart(summary_df=None, model_order=None):
    """
    Generate and save a publication-quality grouped bar chart.
    
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
    
    # Prepare data for plotting
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
    df_long['model_name'] = pd.Categorical(
        df_long['model_name'],
        categories=cleaned_model_order,
        ordered=True
    )
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    palette = sns.color_palette('deep', n_colors=len(df_long['model_name'].unique()))
    
    sns.barplot(
        data=df_long,
        x='condition',
        y='accuracy',
        hue='model_name',
        ax=ax,
        palette=palette
    )
    
    ax.set_title(
        'Figure 1: Model Performance on the AnaphoraGym Benchmark',
        fontsize=16,
        pad=20,
        weight='bold',
        loc='center'
    )
    ax.set_xlabel('Linguistic Condition', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    
    ax.legend(
        title='Model',
        title_fontsize='12',
        fontsize=10,
        bbox_to_anchor=(1.02, 1),
        loc='upper left'
    )
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    # Save
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "model_comparison_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPublication-quality chart saved to '{output_path}'")
    plt.close()


if __name__ == "__main__":
    create_bar_chart()

