# ==============================================================================
# SCRIPT TO GENERATE A RADAR CHART OF SKILL PROFILES
#
# Reads the summary results and creates a radar chart showing the
# performance "fingerprint" of each model.
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def create_radar_chart(summary_filename):
    """
    Reads summary data and generates a model performance radar chart.
    """
    try:
        df = pd.read_csv(summary_filename)
    except FileNotFoundError:
        print(f"Error: The summary file '{summary_filename}' was not found.")
        return

    # --- Data Preparation ---
    labels = df['condition'].values
    num_vars = len(labels)
    
    # Calculate the angle for each axis in the radar chart
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the circle for a complete shape

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    plt.rcParams['font.family'] = 'serif'

    # --- Plot each model as a separate line ---
    model_cols = [col for col in df.columns if 'accuracy_' in col]
    for col in model_cols:
        model_name = col.replace('accuracy_', '').replace('_', '/')
        values = df[col].values.tolist()
        values += values[:1] # Close the circle
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.2)

    # --- Customization ---
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=11)
    
    # Add circular gridlines and labels
    ax.set_rlabel_position(0)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color="grey", size=9)
    ax.set_ylim(0,105)

    ax.set_title('Model Skill Profiles on AnaphoraGym', size=16, color='black', y=1.12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))

    # --- Save the Chart ---
    output_path = os.path.join(IMAGES_DIR, "model_comparison_radar.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nRadar chart saved as '{output_path}'")


if __name__ == "__main__":
    summary_file = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    create_radar_chart(summary_filename=summary_file)