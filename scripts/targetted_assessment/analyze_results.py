
# ==============================================================================
# FINAL SCRIPT: AUTOMATIC ANALYSIS AND PLOTTING (CORRECT PATHS)
#
# This script is designed to be run from the `run_all.sh` orchestrator.
# It automatically finds all result files from the correct directory,
# creates a summary table, and saves a final, publication-quality chart.
# ==============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
import os

# --- 1. DEFINE PROJECT PATHS ---
# This makes the script runnable from anywhere, including the bash script.
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # This fallback is for running in interactive environments like Jupyter
    PROJECT_ROOT = os.path.abspath('.')

# Define the specific directories for results and images
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')

# Ensure the output directory for images exists
os.makedirs(IMAGES_DIR, exist_ok=True)


def analyze_model_performance(results_filename):
    """Calculates the accuracy per condition from a single results CSV file."""
    try:
        df = pd.read_csv(results_filename)
        if df.empty or 'test_passed' not in df.columns: return None
        df_valid = df[df['test_passed'].isin([True, False])].copy()
        if df_valid.empty: return None
        
        df_valid['test_passed'] = df_valid['test_passed'].astype(bool)
        accuracy_df = df_valid.groupby('condition')['test_passed'].mean().reset_index()
        accuracy_df = accuracy_df.rename(columns={'test_passed': 'accuracy'})
        accuracy_df['accuracy'] = (accuracy_df['accuracy'] * 100).round(2)
        return accuracy_df
    except FileNotFoundError:
        print(f"  - Error: Could not find '{results_filename}'")
        return None

def create_comparison_chart(summary_df, model_order):
    """Generates and saves a publication-quality grouped bar chart."""
    id_vars = ['condition']
    value_vars = [f'accuracy_{model}' for model in model_order if f'accuracy_{model}' in summary_df.columns]
    
    df_long = pd.melt(summary_df, id_vars=id_vars, value_vars=value_vars, 
                      var_name='model_name', value_name='accuracy')
    
    df_long['model_name'] = df_long['model_name'].str.replace('accuracy_', '').str.replace('_', '/')
    
    cleaned_model_order = [name.replace('_', '/') for name in model_order]
    df_long['model_name'] = pd.Categorical(df_long['model_name'], categories=cleaned_model_order, ordered=True)
    
    # --- Plotting Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 7))
    
    palette = sns.color_palette('deep', n_colors=len(df_long['model_name'].unique()))

    sns.barplot(data=df_long, x='condition', y='accuracy', hue='model_name', ax=ax, palette=palette)

    # --- Customization ---
    ax.set_title('Figure 1: Model Performance on the AnaphoraGym Benchmark', fontsize=16, pad=20, weight='bold', loc='center')
    ax.set_xlabel('Linguistic Condition', fontsize=12, weight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, weight='bold')
    plt.xticks(rotation=40, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    ax.legend(title='Model', title_fontsize='12', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    # --- Save the Chart to the correct images folder ---
    output_path = os.path.join(IMAGES_DIR, "model_comparison_chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPublication-quality chart saved to '{output_path}'")

def main():
    """Finds all result files, analyzes them, and creates an ordered comparison."""
    
    model_sizes = {
        'gpt2': 124,
        'gpt2-medium': 355,
        'EleutherAI_pythia-410m-deduped': 410,
        'gpt2-large': 774,
        'meta-llama_Llama-3-8B': 8000
    }
    
    # Use the correct path to search for the result files
    search_pattern = os.path.join(RESULTS_DIR, "AnaphoraGym_Results_*.csv")
    result_files = glob.glob(search_pattern)

    if not result_files:
        print(f"Analysis failed: No result files found in '{RESULTS_DIR}'")
        return

    print(f"Found {len(result_files)} result files to compare.")
    
    found_model_names = [os.path.basename(f).split('Results_')[-1].replace('.csv', '') for f in result_files]
    sorted_model_names = sorted(found_model_names, key=lambda name: model_sizes.get(name, float('inf')))
    
    print(f"Processing models in this order: {sorted_model_names}")
    
    comparison_df = None
    for model_name in sorted_model_names:
        filepath = os.path.join(RESULTS_DIR, f"AnaphoraGym_Results_{model_name}.csv")
        print(f"Analyzing {model_name}...")
        accuracy_df = analyze_model_performance(filepath)
        
        if accuracy_df is not None:
            accuracy_df = accuracy_df.rename(columns={'accuracy': f'accuracy_{model_name}'})
            if comparison_df is None:
                comparison_df = accuracy_df
            else:
                comparison_df = pd.merge(comparison_df, accuracy_df, on='condition', how='outer')
    
    if comparison_df is None:
        print("\nAnalysis failed: No valid data could be processed.")
        return

    final_comparison_table = comparison_df.fillna(0)
    print("\n--- MODEL PERFORMANCE COMPARISON ---")
    print(final_comparison_table)
    
    # Save the summary table to the correct results folder
    summary_path = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    final_comparison_table.to_csv(summary_path, index=False)
    print(f"\nComparison summary saved to '{summary_path}'")
    
    create_comparison_chart(final_comparison_table, sorted_model_names)

if __name__ == "__main__":
    main()
