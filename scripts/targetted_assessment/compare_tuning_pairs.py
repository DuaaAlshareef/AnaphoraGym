# ==============================================================================
# SCRIPT TO COMPARE PAIRED BASE VS. INSTRUCTION-TUNED MODELS
#
# This script performs a direct, head-to-head comparison of model pairs
# from the same family (e.g., Llama-2-7b base vs. chat).
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

# --- 1. DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

def main():
    """Finds results for specific model pairs and generates a comparison chart."""
    
    # ================== DEFINE THE PAIRS TO COMPARE ==================
    # This dictionary defines the controlled experiments you want to run.
    # Each key is a "family name" for the chart, and the value is a
    # tuple of the (Base Model ID, Tuned Model ID).
    model_pairs = {
        "Llama-2 (7B)": (
            "gpt2-large",
            "meta-llama/Llama-2-7b-chat-hf"
        ),
        # "Llama-3.2 (1B)": (
        #     "meta-llama/Meta-Llama-3.2-1B",
        #     "meta-llama/Meta-Llama-3.2-8B-Instruct"
        # )
        # Add other pairs here if you test them, e.g.:
        # "Llama-2 (13B)": ("meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-chat-hf")
    }
    # ===============================================================

    all_pairs_data = []
    
    print("--- Analyzing Paired Models ---")
    for family_name, (base_model, tuned_model) in model_pairs.items():
        print(f"\nProcessing family: {family_name}")
        
        # --- Find and load data for the pair ---
        base_file = os.path.join(RESULTS_DIR, f"AnaphoraGym_Results_{base_model.replace('/', '_')}.csv")
        tuned_file = os.path.join(RESULTS_DIR, f"AnaphoraGym_Results_{tuned_model.replace('/', '_')}.csv")

        try:
            df_base = pd.read_csv(base_file)
            df_tuned = pd.read_csv(tuned_file)
            
            # Add a 'tuning_type' column for plotting
            df_base['tuning_type'] = 'Base'
            df_tuned['tuning_type'] = 'Instruction-Tuned'
            
            # Add a 'family' column to group the plots later
            df_base['family'] = family_name
            df_tuned['family'] = family_name
            
            # Combine the data for this pair
            all_pairs_data.append(df_base)
            all_pairs_data.append(df_tuned)
            print(f"  - Successfully loaded data for both models.")

        except FileNotFoundError as e:
            print(f"  - WARNING: Could not find result file for this pair. Skipping. Missing file: {e.filename}")
            continue

    if not all_pairs_data:
        print("\n[ERROR] No data found for any of the defined model pairs. Please run the experiments first.")
        return

    # Consolidate all found pair data into one DataFrame
    master_df = pd.concat(all_pairs_data, ignore_index=True)

    # --- Calculate Average Accuracy ---
    summary = master_df.groupby(['family', 'condition', 'tuning_type'])['test_passed'].mean().reset_index()
    summary['accuracy'] = (summary['test_passed'] * 100).round(2)

    print("\n--- PAIRED COMPARISON SUMMARY ---")
    print(summary)

    # --- Generate the Faceted Comparison Chart ---
    print("\nGenerating paired comparison chart...")
    
    # Use catplot to create a faceted chart, with one facet per 'family'
    g = sns.catplot(
        data=summary,
        x='condition',
        y='accuracy',
        hue='tuning_type',
        col='family', # This creates the separate charts for each family
        kind='bar',
        height=6,
        aspect=1.2,
        palette={"Base": "skyblue", "Instruction-Tuned": "salmon"},
        legend_out=False
    )

    # --- Customization ---
    g.fig.suptitle('Direct Comparison of Base vs. Instruction-Tuned Models', y=1.05, fontsize=16, weight='bold')
    g.set_axis_labels("Linguistic Condition", "Accuracy (%)")
    g.set_titles("Model Family: {col_name}")
    g.set(ylim=(0, 105))
    
    for ax in g.axes.flatten():
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.7)

    # Add a shared legend
    g.add_legend(title="Tuning Type")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # --- Save the Chart ---
    output_path = os.path.join(IMAGES_DIR, "paired_tuning_comparison_chart.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nPaired comparison chart saved to '{output_path}'")


if __name__ == "__main__":
    main()