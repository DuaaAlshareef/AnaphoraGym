"""
Visualize model calibration and confidence

Shows:
1. Calibration curve (confidence vs accuracy)
2. Confidence distribution for correct vs incorrect
3. Accuracy by confidence bin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir

# Set clean style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def visualize_calibration(model_name: str):
    """Create calibration visualization."""
    results_dir = get_results_dir()
    safe_model_name = model_name.replace('/', '_')
    
    # Try to load confidence-enhanced results first
    results_path = os.path.join(results_dir, f"AnaphoraGym_Results_{safe_model_name}_with_confidence.csv")
    
    if not os.path.exists(results_path):
        # Fall back to regular results and add confidence on the fly
        results_path = os.path.join(results_dir, f"AnaphoraGym_Results_{safe_model_name}.csv")
        if not os.path.exists(results_path):
            print(f"❌ Results not found for {model_name}")
            print(f"\nRun the assessment first:")
            print(f"python scripts/targetted_assessment/experiments/run_experiment.py --model {model_name}")
            return
        
        df = pd.read_csv(results_path)
        df['confidence'] = np.abs(df['logOdds'])
        df['correct'] = df['test_passed'].astype(int)
    else:
        df = pd.read_csv(results_path)
    
    print(f"✅ Loaded results for {model_name}")
    print(f"   Total tests: {len(df)}")
    print(f"   Overall accuracy: {df['correct'].mean()*100:.1f}%")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Calibration & Confidence Analysis: {model_name}', 
                fontsize=15, fontweight='bold')
    
    # 1. Calibration Curve
    ax = axes[0, 0]
    
    # Bin by confidence and calculate accuracy per bin
    n_bins = 10
    df['conf_bin_num'] = pd.qcut(df['confidence'], q=n_bins, duplicates='drop', labels=False)
    
    calibration_data = df.groupby('conf_bin_num').agg({
        'confidence': 'mean',
        'correct': 'mean'
    }).reset_index()
    
    ax.plot(calibration_data['confidence'], calibration_data['correct'], 
           'o-', linewidth=2, markersize=8, color='steelblue', label='Model')
    ax.plot([0, df['confidence'].max()], [0, 1], 
           'r--', linewidth=2, alpha=0.5, label='Perfect calibration')
    
    ax.set_xlabel('Confidence (|log odds|)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_title('Calibration Curve', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add correlation text
    correlation = df['confidence'].corr(df['correct'])
    ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           fontsize=10, fontweight='bold')
    
    # 2. Confidence Distribution (Correct vs Incorrect)
    ax = axes[0, 1]
    
    correct_conf = df[df['correct'] == 1]['confidence']
    incorrect_conf = df[df['correct'] == 0]['confidence']
    
    ax.hist(correct_conf, bins=30, alpha=0.6, color='green', 
           label=f'Correct (n={len(correct_conf)})', edgecolor='black', linewidth=0.5)
    ax.hist(incorrect_conf, bins=30, alpha=0.6, color='red', 
           label=f'Incorrect (n={len(incorrect_conf)})', edgecolor='black', linewidth=0.5)
    
    ax.axvline(correct_conf.mean(), color='darkgreen', linestyle='--', 
              linewidth=2, label=f'Mean correct: {correct_conf.mean():.2f}')
    ax.axvline(incorrect_conf.mean(), color='darkred', linestyle='--', 
              linewidth=2, label=f'Mean incorrect: {incorrect_conf.mean():.2f}')
    
    ax.set_xlabel('Confidence (|log odds|)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Confidence Distribution', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Accuracy by Confidence Bin
    ax = axes[1, 0]
    
    # Create confidence bins with labels
    df['conf_category'] = pd.cut(df['confidence'], 
                                  bins=5, 
                                  labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    bin_stats = df.groupby('conf_category', observed=True).agg({
        'correct': ['sum', 'count', 'mean']
    }).reset_index()
    bin_stats.columns = ['Confidence', 'Correct', 'Total', 'Accuracy']
    
    colors = plt.cm.RdYlGn(bin_stats['Accuracy'])
    bars = ax.bar(range(len(bin_stats)), bin_stats['Accuracy'], 
                  color=colors, edgecolor='black', linewidth=1.5)
    
    ax.set_xticks(range(len(bin_stats)))
    ax.set_xticklabels(bin_stats['Confidence'], rotation=0, fontsize=10)
    ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax.set_xlabel('Confidence Level', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy by Confidence Level', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.grid(alpha=0.3, axis='y')
    ax.legend()
    
    # Add value labels
    for i, (acc, n) in enumerate(zip(bin_stats['Accuracy'], bin_stats['Total'])):
        ax.text(i, acc + 0.02, f'{acc:.2f}\n(n={n})', 
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate summary stats
    overall_acc = df['correct'].mean()
    mean_conf_correct = correct_conf.mean()
    mean_conf_incorrect = incorrect_conf.mean()
    conf_diff = mean_conf_correct - mean_conf_incorrect
    
    # Determine calibration quality
    if correlation > 0.3:
        calib_quality = "WELL CALIBRATED ✓"
        calib_color = 'lightgreen'
    elif correlation > 0.1:
        calib_quality = "MODERATELY CALIBRATED"
        calib_color = 'lightyellow'
    else:
        calib_quality = "POORLY CALIBRATED ⚠"
        calib_color = 'lightcoral'
    
    summary = f"""CALIBRATION SUMMARY

Model: {model_name}

Overall Accuracy: {overall_acc:.1%}

Confidence Statistics:
  Mean (all):     {df['confidence'].mean():.3f}
  Mean (correct): {mean_conf_correct:.3f}
  Mean (wrong):   {mean_conf_incorrect:.3f}
  Difference:     {conf_diff:.3f}

Calibration Metrics:
  Correlation:    {correlation:.3f}
  Status:         {calib_quality}

Interpretation:
{'  Model is more confident' if conf_diff > 0.1 else '  Similar confidence'}
{'  when making correct' if conf_diff > 0.1 else '  for both correct'}
{'  predictions.' if conf_diff > 0.1 else '  and incorrect predictions.'}

{f'  Confidence difference: {conf_diff:.3f}' if conf_diff > 0.1 else '  Confidence difference is small.'}
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor=calib_color, 
                    alpha=0.5, edgecolor='gray', linewidth=1.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save
    output_path = os.path.join(results_dir, f"calibration_analysis_{safe_model_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved calibration visualization: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize model calibration and confidence"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., gpt2)")
    args = parser.parse_args()
    
    visualize_calibration(args.model)
