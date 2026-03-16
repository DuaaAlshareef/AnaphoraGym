#!/usr/bin/env python3
# coding=utf-8
"""
Visualization for Layer-wise Anaphora Probing Results
Creates comprehensive visualizations showing which layers encode anaphoric information.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_layer_performance(stats_df, output_path, title="Layer-wise Anaphora Encoding"):
    """
    Create a comprehensive plot showing layer performance.
    
    Args:
        stats_df: DataFrame with layer statistics
        output_path: Path to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Mean log-probability difference by layer
    ax1 = axes[0, 0]
    ax1.plot(stats_df['layer'], stats_df['mean_logprob_diff'], 
             marker='o', linewidth=2, markersize=6, color='#2E86AB')
    ax1.fill_between(stats_df['layer'], 
                      stats_df['mean_logprob_diff'] - stats_df['std_logprob_diff'],
                      stats_df['mean_logprob_diff'] + stats_df['std_logprob_diff'],
                      alpha=0.3, color='#2E86AB')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Chance level')
    ax1.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Log-Probability Difference', fontsize=12, fontweight='bold')
    ax1.set_title('Anaphora Resolution Performance Across Layers', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight best layer
    best_layer = stats_df.loc[stats_df['mean_logprob_diff'].idxmax(), 'layer']
    best_score = stats_df['mean_logprob_diff'].max()
    ax1.scatter([best_layer], [best_score], color='red', s=200, 
                marker='*', zorder=5, label=f'Best Layer: {best_layer}')
    ax1.legend()
    
    # 2. Accuracy by layer
    ax2 = axes[0, 1]
    colors = ['#A23B72' if acc > 0.5 else '#C73E1D' for acc in stats_df['accuracy']]
    ax2.bar(stats_df['layer'], stats_df['accuracy'], color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax2.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Classification Accuracy by Layer', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # 3. Median log-probability difference
    ax3 = axes[1, 0]
    ax3.plot(stats_df['layer'], stats_df['median_logprob_diff'], 
             marker='s', linewidth=2, markersize=6, color='#F18F01')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Median Log-Probability Difference', fontsize=12, fontweight='bold')
    ax3.set_title('Median Performance (Robust to Outliers)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Layer ranking (top 10)
    ax4 = axes[1, 1]
    top_layers = stats_df.nlargest(10, 'mean_logprob_diff')
    ax4.barh(top_layers['layer'].astype(str), top_layers['mean_logprob_diff'],
             color='#6A994E', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Mean Log-Probability Difference', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax4.set_title('Top 10 Layers for Anaphora Encoding', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer performance plot to: {output_path}")
    plt.close()


def plot_layer_heatmap(detailed_df, output_path, title="Layer Performance by Condition"):
    """
    Create a heatmap showing layer performance across different anaphora conditions.
    
    Args:
        detailed_df: DataFrame with detailed results per example
        output_path: Path to save the plot
        title: Plot title
    """
    if detailed_df is None or len(detailed_df) == 0:
        print("No detailed data available for heatmap")
        return
    
    # Get layer columns
    layer_cols = [col for col in detailed_df.columns if isinstance(col, (int, np.integer)) or 
                  (isinstance(col, str) and col.isdigit())]
    
    if not layer_cols:
        print("No layer columns found in detailed data")
        return
    
    # Group by condition and calculate mean for each layer
    condition_layer_scores = detailed_df.groupby('condition')[layer_cols].mean()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Convert column names to integers for proper sorting
    condition_layer_scores.columns = [int(col) if str(col).isdigit() else col 
                                       for col in condition_layer_scores.columns]
    condition_layer_scores = condition_layer_scores.sort_index(axis=1)
    
    sns.heatmap(condition_layer_scores, cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Log-Probability Difference'},
                linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Anaphora Condition', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer heatmap to: {output_path}")
    plt.close()


def plot_layer_trajectory(stats_df, output_path, title="Anaphora Encoding Trajectory"):
    """
    Create a trajectory plot showing how anaphora encoding evolves through layers.
    
    Args:
        stats_df: DataFrame with layer statistics
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create gradient effect
    layers = stats_df['layer'].values
    scores = stats_df['mean_logprob_diff'].values
    
    # Plot line with gradient
    for i in range(len(layers) - 1):
        ax.plot(layers[i:i+2], scores[i:i+2], 
                color=plt.cm.viridis(i / len(layers)),
                linewidth=3, alpha=0.8)
    
    # Add markers
    scatter = ax.scatter(layers, scores, c=layers, cmap='viridis', 
                        s=100, edgecolor='black', linewidth=1.5, zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Layer Depth', fontsize=12, fontweight='bold')
    
    # Highlight critical layers
    best_layer = stats_df.loc[stats_df['mean_logprob_diff'].idxmax(), 'layer']
    best_score = stats_df['mean_logprob_diff'].max()
    ax.scatter([best_layer], [best_score], color='red', s=400, 
               marker='*', zorder=10, edgecolor='darkred', linewidth=2,
               label=f'Peak Layer: {best_layer}')
    
    # Add reference line
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, 
               label='Chance Level', linewidth=2)
    
    # Labels and title
    ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Anaphora Resolution Score\n(Log-Probability Difference)', 
                  fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best')
    
    # Annotate regions
    num_layers = len(layers)
    ax.axvspan(0, num_layers * 0.3, alpha=0.1, color='blue', label='Early Layers')
    ax.axvspan(num_layers * 0.3, num_layers * 0.7, alpha=0.1, color='green', label='Middle Layers')
    ax.axvspan(num_layers * 0.7, num_layers, alpha=0.1, color='orange', label='Late Layers')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer trajectory plot to: {output_path}")
    plt.close()


def plot_layer_comparison(stats_df, output_path, title="Layer Performance Comparison"):
    """
    Create a comparison plot showing multiple metrics across layers.
    
    Args:
        stats_df: DataFrame with layer statistics
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Normalize metrics to [0, 1] for comparison
    norm_mean = (stats_df['mean_logprob_diff'] - stats_df['mean_logprob_diff'].min()) / \
                (stats_df['mean_logprob_diff'].max() - stats_df['mean_logprob_diff'].min())
    norm_median = (stats_df['median_logprob_diff'] - stats_df['median_logprob_diff'].min()) / \
                  (stats_df['median_logprob_diff'].max() - stats_df['median_logprob_diff'].min())
    accuracy = stats_df['accuracy']
    
    # Plot multiple metrics
    ax.plot(stats_df['layer'], norm_mean, marker='o', linewidth=2.5, 
            markersize=8, label='Mean Performance (normalized)', color='#2E86AB')
    ax.plot(stats_df['layer'], norm_median, marker='s', linewidth=2.5, 
            markersize=8, label='Median Performance (normalized)', color='#F18F01')
    ax.plot(stats_df['layer'], accuracy, marker='^', linewidth=2.5, 
            markersize=8, label='Accuracy', color='#A23B72')
    
    # Add reference line
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, 
               label='Baseline', linewidth=2)
    
    # Labels and title
    ax.set_xlabel('Layer Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance (Normalized)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12, loc='best', framealpha=0.9)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved layer comparison plot to: {output_path}")
    plt.close()


def create_summary_report(summary_path, stats_df, output_path):
    """
    Create a text summary report of the layer probing results.
    
    Args:
        summary_path: Path to summary JSON
        stats_df: DataFrame with layer statistics
        output_path: Path to save the text report
    """
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    report_lines = [
        "="*80,
        "LAYER-WISE ANAPHORA ENCODING ANALYSIS REPORT",
        "="*80,
        "",
        f"Model: {summary['model']}",
        f"Total Layers: {summary['num_layers']}",
        f"Examples Analyzed: {summary['num_examples']}",
        "",
        "-"*80,
        "KEY FINDINGS",
        "-"*80,
        "",
        f"Best Performing Layer: Layer {summary['best_layer']}",
        f"Best Layer Score: {summary['best_layer_score']:.4f}",
        "",
        "Top 5 Layers:",
    ]
    
    top5 = stats_df.nlargest(5, 'mean_logprob_diff')
    for idx, row in top5.iterrows():
        report_lines.append(
            f"  Layer {int(row['layer']):2d}: "
            f"Score = {row['mean_logprob_diff']:6.4f}, "
            f"Accuracy = {row['accuracy']:.2%}"
        )
    
    report_lines.extend([
        "",
        "-"*80,
        "LAYER REGIONS ANALYSIS",
        "-"*80,
        ""
    ])
    
    # Analyze by regions
    num_layers = summary['num_layers']
    early_layers = stats_df[stats_df['layer'] < num_layers * 0.3]
    middle_layers = stats_df[(stats_df['layer'] >= num_layers * 0.3) & 
                             (stats_df['layer'] < num_layers * 0.7)]
    late_layers = stats_df[stats_df['layer'] >= num_layers * 0.7]
    
    report_lines.extend([
        f"Early Layers (0-{int(num_layers * 0.3)}):",
        f"  Mean Score: {early_layers['mean_logprob_diff'].mean():.4f}",
        f"  Mean Accuracy: {early_layers['accuracy'].mean():.2%}",
        "",
        f"Middle Layers ({int(num_layers * 0.3)}-{int(num_layers * 0.7)}):",
        f"  Mean Score: {middle_layers['mean_logprob_diff'].mean():.4f}",
        f"  Mean Accuracy: {middle_layers['accuracy'].mean():.2%}",
        "",
        f"Late Layers ({int(num_layers * 0.7)}-{num_layers}):",
        f"  Mean Score: {late_layers['mean_logprob_diff'].mean():.4f}",
        f"  Mean Accuracy: {late_layers['accuracy'].mean():.2%}",
        "",
        "-"*80,
        "INTERPRETATION",
        "-"*80,
        "",
    ])
    
    # Add interpretation
    best_region = "early" if summary['best_layer'] < num_layers * 0.3 else \
                  "middle" if summary['best_layer'] < num_layers * 0.7 else "late"
    
    report_lines.extend([
        f"The model encodes anaphoric information most strongly in the {best_region}",
        f"layers, with peak performance at layer {summary['best_layer']}.",
        "",
        "This suggests that anaphora resolution mechanisms are primarily active in",
        f"the {best_region} stages of the transformer's processing pipeline.",
        "",
        "="*80,
    ])
    
    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved summary report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize layer-wise anaphora probing results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/mechanistic_analysis/layer_probing",
        help="Directory containing probing results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="images/layer_probing",
        help="Directory to save visualizations",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    stats_path = os.path.join(args.results_dir, 'layer_statistics.csv')
    detailed_path = os.path.join(args.results_dir, 'detailed_layer_results.csv')
    summary_path = os.path.join(args.results_dir, 'summary.json')
    
    if not os.path.exists(stats_path):
        print(f"Error: Statistics file not found at {stats_path}")
        return
    
    print(f"Loading results from: {args.results_dir}")
    stats_df = pd.read_csv(stats_path)
    
    detailed_df = None
    if os.path.exists(detailed_path):
        detailed_df = pd.read_csv(detailed_path)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # 1. Main layer performance plot
    plot_layer_performance(
        stats_df,
        os.path.join(args.output_dir, 'layer_performance.png'),
        title="Layer-wise Anaphora Encoding in Llama-2-7b-chat-hf"
    )
    
    # 2. Layer trajectory plot
    plot_layer_trajectory(
        stats_df,
        os.path.join(args.output_dir, 'layer_trajectory.png'),
        title="Anaphora Encoding Trajectory Across Layers"
    )
    
    # 3. Layer comparison plot
    plot_layer_comparison(
        stats_df,
        os.path.join(args.output_dir, 'layer_comparison.png'),
        title="Multi-Metric Layer Performance Comparison"
    )
    
    # 4. Heatmap (if detailed data available)
    if detailed_df is not None:
        plot_layer_heatmap(
            detailed_df,
            os.path.join(args.output_dir, 'layer_condition_heatmap.png'),
            title="Layer Performance by Anaphora Condition"
        )
    
    # 5. Create summary report
    if os.path.exists(summary_path):
        create_summary_report(
            summary_path,
            stats_df,
            os.path.join(args.output_dir, 'analysis_report.txt')
        )
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("\nGenerated files:")
    print("  - layer_performance.png: Comprehensive 4-panel performance plot")
    print("  - layer_trajectory.png: Gradient trajectory showing encoding evolution")
    print("  - layer_comparison.png: Multi-metric comparison across layers")
    if detailed_df is not None:
        print("  - layer_condition_heatmap.png: Performance by condition")
    if os.path.exists(summary_path):
        print("  - analysis_report.txt: Text summary of findings")


if __name__ == "__main__":
    main()
