"""
Compare calibration across all models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def compare_calibration():
    """Create ONE clear calibration visualization."""
    results_dir = get_results_dir()
    
    # Load calibration summary
    summary_path = os.path.join(results_dir, 'model_calibration_summary.csv')
    
    if not os.path.exists(summary_path):
        print("❌ Calibration summary not found!")
        print("\nRun this first:")
        print("python scripts/targeted_assessment/analysis/add_calibration_to_all_models.py")
        return
    
    summary_df = pd.read_csv(summary_path)
    summary_df = summary_df.sort_values('correlation', ascending=True)
    
    print(f"✅ Loaded calibration data for {len(summary_df)} models")
    
    # Create ONE clear visualization
    fig, ax = plt.subplots(figsize=(12, max(8, len(summary_df) * 0.5)))
    
    # Assign colors based on calibration quality
    def get_color(corr):
        if corr >= 0.3:
            return '#2ecc71'  # Green - well calibrated
        elif corr >= 0.1:
            return '#f39c12'  # Orange - moderate
        else:
            return '#e74c3c'  # Red - poor
    
    colors = [get_color(c) for c in summary_df['correlation']]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(summary_df)), summary_df['correlation'], 
                   color=colors, edgecolor='black', linewidth=1.5, height=0.7)
    
    # Labels
    ax.set_yticks(range(len(summary_df)))
    ax.set_yticklabels(summary_df['model'], fontsize=11)
    ax.set_xlabel('Calibration Score (Correlation: Confidence ↔ Correctness)', 
                 fontsize=13, fontweight='bold')
    ax.set_title('Model Calibration: How Well Does Confidence Match Correctness?', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Reference lines
    ax.axvline(0, color='black', linewidth=2)
    ax.axvline(0.3, color='darkgreen', linestyle='--', linewidth=2.5, alpha=0.7,
              label='Well Calibrated (≥0.3)')
    ax.axvline(0.1, color='darkorange', linestyle='--', linewidth=2.5, alpha=0.7,
              label='Moderate (≥0.1)')
    
    # Add value labels with accuracy info
    for i, (corr, model) in enumerate(zip(summary_df['correlation'], summary_df['model'])):
        acc = summary_df.iloc[i]['accuracy']
        label_text = f" {corr:.3f} | Acc: {acc*100:.1f}%"
        ax.text(corr + 0.01, i, label_text, va='center', ha='left',
               fontsize=9, fontweight='bold')
    
    ax.grid(axis='x', alpha=0.4, linewidth=0.8)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.set_xlim(-0.05, max(summary_df['correlation'].max() + 0.15, 0.5))
    
    # Add interpretation box
    interpretation = (
        "🎯 Calibration Score:\n"
        "• ≥0.3 = Can trust confidence\n"
        "• 0.1-0.3 = Some reliability\n"
        "• <0.1 = Confidence unreliable"
    )
    ax.text(0.02, 0.98, interpretation, transform=ax.transAxes,
           verticalalignment='top', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', 
                    edgecolor='navy', linewidth=2, alpha=0.9))
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(results_dir, 'model_calibration_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 CALIBRATION RANKING")
    print("="*70)
    print(f"{'Rank':<6} {'Model':<35} {'Correlation':<12} {'Acc':<8}")
    print("-"*70)
    
    sorted_df = summary_df.sort_values('correlation', ascending=False)
    for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
        status = "🏆" if i == 1 else "✓" if row['correlation'] > 0.3 else "~" if row['correlation'] > 0.1 else "⚠"
        print(f"{i:<6} {row['model']:<35} {row['correlation']:<12.3f} {row['accuracy']*100:<8.1f}% {status}")


if __name__ == "__main__":
    compare_calibration()
