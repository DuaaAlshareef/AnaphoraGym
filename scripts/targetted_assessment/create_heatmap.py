

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def create_heatmap_chart(summary_filename="./results/targetted_assessment/model_comparison_summary.csv"):
    
    df = pd.read_csv(summary_filename)
    
    # Set 'condition' as the index and clean up column names for the axes
    df = df.set_index('condition')
    df.columns = df.columns.str.replace('accuracy_', '').str.replace('_', '/')
    
    # Transpose the DataFrame so models are on the Y-axis and conditions are on the X-axis
    df_transposed = df.transpose()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 6))

    # Use a high-contrast, perceptual palette like 'viridis' or 'plasma'
    sns.heatmap(
        df_transposed, 
        annot=True,      # Add the numbers to the cells
        fmt=".1f",       # Format numbers to one decimal place
        cmap="viridis",  # A good colorblind-friendly palette
        linewidths=.5,
        ax=ax
    )
    
    ax.set_title('Model Performance Heatmap on AnaphoraGym', fontsize=16, pad=20, weight='bold')
    ax.set_xlabel('Linguistic Condition', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    plt.xticks(rotation=40, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    output_filename = "./results/targetted_assessment/model_comparison_heatmap.png"
    # output_filename = "model_comparison_heatmap.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap chart saved as '{output_filename}'")

if __name__ == "__main__":
    create_heatmap_chart()
    