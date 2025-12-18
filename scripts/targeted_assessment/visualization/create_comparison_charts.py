"""
Create comparison charts for model types and tuning pairs.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_images_dir

# Import analysis functions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from compare_model_types import compare_model_types
from compare_tuning_pairs import compare_tuning_pairs


def create_model_type_comparison():
    """Create a bar chart comparing base vs. instruction-tuned models."""
    group_summary = compare_model_types()
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    palette = {"Base": "steelblue", "Instruction-Tuned": "darkorange"}
    
    sns.barplot(
        data=group_summary,
        x='condition',
        y='accuracy',
        hue='model_type',
        ax=ax,
        palette=palette
    )
    
    ax.set_title(
        'Base vs. Instruction-Tuned Model Performance on AnaphoraGym',
        fontsize=16,
        pad=20,
        weight='bold'
    )
    ax.set_xlabel('Linguistic Condition', fontsize=12, weight='bold')
    ax.set_ylabel('Average Accuracy (%)', fontsize=12, weight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=11)
    plt.yticks(np.arange(0, 101, 10), fontsize=11)
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    ax.legend(title='Model Type', title_fontsize='12', fontsize=11)
    plt.tight_layout()
    
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "base_vs_instruct_comparison_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison chart saved to '{output_path}'")
    plt.close()


def create_tuning_pairs_comparison():
    """Create a comparison chart for paired models."""
    comparison_df = compare_tuning_pairs()
    
    if comparison_df.empty:
        print("No paired comparison data available.")
        return
    
    # Create visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot base vs tuned for each condition
    x = np.arange(len(comparison_df['condition'].unique()))
    width = 0.35
    
    conditions = comparison_df['condition'].unique()
    base_accs = []
    tuned_accs = []
    
    for condition in conditions:
        cond_data = comparison_df[comparison_df['condition'] == condition]
        base_accs.append(cond_data['base_accuracy'].mean())
        tuned_accs.append(cond_data['tuned_accuracy'].mean())
    
    ax.bar(x - width/2, base_accs, width, label='Base', color='steelblue')
    ax.bar(x + width/2, tuned_accs, width, label='Instruction-Tuned', color='darkorange')
    
    ax.set_xlabel('Linguistic Condition', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
    ax.set_title('Paired Model Comparison', fontsize=16, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=40, ha='right')
    ax.legend()
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)
    plt.tight_layout()
    
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "paired_tuning_comparison_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPaired comparison chart saved to '{output_path}'")
    plt.close()


def create_dataset_overview():
    """Create an overview visualization of the dataset."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
    from dataset_stats import calculate_dataset_stats
    
    summary_df = calculate_dataset_stats()
    
    sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.4)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=False)
    
    # Plot 1: Number of Samples per Condition
    sns.barplot(
        ax=axes[0],
        x='num_samples',
        y='condition',
        data=summary_df,
        orient='h',
        hue='num_samples',
        dodge=False,
        palette='Blues_r'
    )
    axes[0].set_title('Number of Samples (Items) per Condition', fontsize=18, fontweight='bold')
    axes[0].set_xlabel('Number of Samples', fontsize=16)
    axes[0].set_ylabel('Condition', fontsize=16)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    axes[0].grid(axis='x', linestyle='--', alpha=0.7)
    
    for index, value in enumerate(summary_df['num_samples']):
        axes[0].text(value + 0.1, index, str(int(value)), color='black', va='center', fontsize=12)
    axes[0].legend_.remove()
    
    # Plot 2: Total Number of Tests per Condition
    sns.barplot(
        ax=axes[1],
        x='total_tests',
        y='condition',
        data=summary_df,
        orient='h',
        hue='total_tests',
        dodge=False,
        palette='Reds_r'
    )
    axes[1].set_title('Total Number of Tests per Condition', fontsize=18, fontweight='bold')
    axes[1].set_xlabel('Total Number of Tests', fontsize=16)
    axes[1].set_ylabel('', fontsize=16)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    axes[1].grid(axis='x', linestyle='--', alpha=0.7)
    
    for index, value in enumerate(summary_df['total_tests']):
        axes[1].text(value + 0.1, index, str(int(value)), color='black', va='center', fontsize=12)
    axes[1].legend_.remove()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle('AnaphoraGym Dataset Overview', fontsize=22, fontweight='bold', y=0.98)
    
    images_dir = get_images_dir()
    output_path = os.path.join(images_dir, "dataset_overview_for_poster.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDataset overview saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_model_type_comparison()
    create_tuning_pairs_comparison()
    create_dataset_overview()

