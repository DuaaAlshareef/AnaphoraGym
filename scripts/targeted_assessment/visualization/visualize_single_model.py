"""
Simple visualization for a single model's results
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir, get_images_dir

# Set better defaults for clean text rendering
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def visualize_single_model(model_name: str):
    """Create visualization for a single model's results."""
    
    # Load results
    results_dir = get_results_dir()
    safe_model_name = model_name.replace('/', '_')
    results_path = os.path.join(results_dir, f"AnaphoraGym_Results_{safe_model_name}.csv")
    
    if not os.path.exists(results_path):
        print(f"❌ Results not found: {results_path}")
        print(f"\nRun the assessment first:")
        print(f"python3 scripts/targeted_assessment/experiments/run_experiment.py --model {model_name}")
        return
    
    df = pd.read_csv(results_path)
    print(f"✅ Loaded results for {model_name}")
    print(f"   Total tests: {len(df)}")
    print(f"   Tests passed: {df['test_passed'].sum()}")
    print(f"   Overall accuracy: {df['test_passed'].mean() * 100:.1f}%")
    
    # Calculate accuracy by condition
    accuracy_by_condition = df.groupby('condition').agg({
        'test_passed': ['sum', 'count', 'mean']
    }).reset_index()
    accuracy_by_condition.columns = ['condition', 'passed', 'total', 'accuracy']
    accuracy_by_condition['accuracy_pct'] = accuracy_by_condition['accuracy'] * 100
    accuracy_by_condition = accuracy_by_condition.sort_values('accuracy_pct', ascending=False)
    
    print(f"\n📊 Accuracy by condition:")
    for _, row in accuracy_by_condition.iterrows():
        print(f"   {row['condition']:30s}: {row['accuracy_pct']:5.1f}% ({row['passed']:.0f}/{row['total']:.0f})")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(f'AnaphoraGym Assessment Results: {model_name}', 
                fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Accuracy by condition (bar chart)
    ax = axes[0, 0]
    colors = plt.cm.RdYlGn(accuracy_by_condition['accuracy_pct'] / 100)
    bars = ax.barh(range(len(accuracy_by_condition)), accuracy_by_condition['accuracy_pct'], 
                   color=colors, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(accuracy_by_condition)))
    ax.set_yticklabels(accuracy_by_condition['condition'], fontsize=10)
    ax.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Accuracy by Category', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(0, 105)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Chance')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, val in enumerate(accuracy_by_condition['accuracy_pct']):
        ax.text(val + 1.5, i, f'{val:.1f}%', va='center', ha='left', 
               fontsize=9, fontweight='bold')
    
    # 2. Test difficulty (log odds distribution)
    ax = axes[0, 1]
    ax.hist(df['logOdds'], bins=35, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', linewidth=2.5, label='Decision boundary')
    ax.set_xlabel('Log Odds (left vs right)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Log Odds Scores', fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # Add text annotation
    passed_pct = df['test_passed'].mean() * 100
    stats_text = (f'Passed: {df["test_passed"].sum()}/{len(df)}\n'
                 f'Accuracy: {passed_pct:.1f}%')
    ax.text(0.03, 0.97, stats_text,
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7, pad=0.7),
           fontsize=10, fontweight='bold', family='monospace')
    
    # 3. Tests passed vs failed by condition
    ax = axes[1, 0]
    pass_fail = df.groupby(['condition', 'test_passed']).size().unstack(fill_value=0)
    
    x = np.arange(len(pass_fail))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pass_fail[False], width, label='Failed', 
                  color='#ff7675', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, pass_fail[True], width, label='Passed', 
                  color='#74b9ff', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Category', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Tests', fontsize=11, fontweight='bold')
    ax.set_title('Pass/Fail Distribution by Category', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(pass_fail.index, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    # Calculate statistics
    overall_acc = df['test_passed'].mean() * 100
    total_tests = len(df)
    passed_tests = df['test_passed'].sum()
    failed_tests = total_tests - passed_tests
    n_conditions = df['condition'].nunique()
    best_condition = accuracy_by_condition.iloc[0]
    worst_condition = accuracy_by_condition.iloc[-1]
    
    # Create summary text with better formatting
    summary = f"""SUMMARY STATISTICS

Model: {model_name}

Overall Performance:
  Total tests:  {total_tests:4d}
  Passed:       {passed_tests:4d} ({overall_acc:5.1f}%)
  Failed:       {failed_tests:4d} ({100-overall_acc:5.1f}%)

Categories:     {n_conditions}

Best Performance:
  {best_condition['condition'][:25]}
  Accuracy: {best_condition['accuracy_pct']:.1f}%

Worst Performance:
  {worst_condition['condition'][:25]}
  Accuracy: {worst_condition['accuracy_pct']:.1f}%

Log Odds Stats:
  Mean:         {df['logOdds'].mean():7.3f}
  Median:       {df['logOdds'].median():7.3f}
  Std Dev:      {df['logOdds'].std():7.3f}
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='#f0f0f0', 
                    alpha=0.8, edgecolor='gray', linewidth=1.5))
    
    # Adjust layout with more space
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # Save figure to images/
    images_dir = get_images_dir()
    output_filename = f"single_model_results_{safe_model_name}.png"
    output_path = os.path.join(images_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✅ Saved visualization: {output_path}")
    
    # Save accuracy table to results/
    table_path = os.path.join(results_dir, f"accuracy_by_condition_{safe_model_name}.csv")
    accuracy_by_condition.to_csv(table_path, index=False)
    print(f"✅ Saved accuracy table: {table_path}")
    
    plt.close()  # Close instead of show to avoid display issues


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize results for a single model")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (e.g., gpt2)")
    args = parser.parse_args()
    
    visualize_single_model(args.model)
