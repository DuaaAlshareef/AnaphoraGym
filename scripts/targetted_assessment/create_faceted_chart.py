# # # ==============================================================================
# # # SCRIPT TO GENERATE A FACETED (SMALL MULTIPLES) BAR CHART
# # #
# # # Reads the summary results and creates a grid of small charts,
# # # one for each linguistic condition.
# # # ==============================================================================
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # import os

# # # --- DEFINE PROJECT PATHS ---
# # try:
# #     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# #     PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# # except NameError:
# #     PROJECT_ROOT = os.path.abspath('.')

# # RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
# # IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
# # os.makedirs(IMAGES_DIR, exist_ok=True)


# # def create_faceted_chart(summary_filename):
# #     """
# #     Reads summary data and generates a faceted bar chart.
# #     """
# #     try:
# #         df = pd.read_csv(summary_filename)
# #     except FileNotFoundError:
# #         print(f"Error: The summary file '{summary_filename}' was not found.")
# #         return

# #     # Melt the data into a long format suitable for faceting
# #     id_vars = ['condition']
# #     value_vars = [col for col in df.columns if 'accuracy_' in col]
# #     df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
# #                       var_name='model_name', value_name='accuracy')
# #     df_long['model_name'] = df_long['model_name'].str.replace('accuracy_', '').str.replace('_', '/')
    
# #     # --- Create the Faceted Plot ---
# #     # `catplot` is seaborn's tool for creating faceted categorical plots
# #     g = sns.catplot(
# #         data=df_long,
# #         x='model_name',
# #         y='accuracy',
# #         col='condition',  # Create a new column of plots for each condition
# #         kind='bar',       # Use a bar chart
# #         height=5,
# #         aspect=0.8,
# #         palette='deep',
# #         col_wrap=3        # Wrap to the next row after 3 plots
# #     )

# #     # --- Customization ---
# #     g.fig.suptitle('Model Performance by Linguistic Condition', y=1.03, fontsize=16, weight='bold')
# #     g.set_axis_labels("Model", "Accuracy (%)")
# #     g.set_titles("Condition: {col_name}")
# #     g.set(ylim=(0, 105))
# #     g.despine(left=True)
    
# #     # Rotate x-axis labels on all subplots for readability
# #     for ax in g.axes.flatten():
# #         ax.tick_params(axis='x', rotation=60)

# #     plt.tight_layout(rect=[0, 0, 1, 0.97])

# #     # --- Save the Chart ---
# #     output_path = os.path.join(IMAGES_DIR, "model_comparison_faceted.png")
# #     plt.savefig(output_path, dpi=300)
# #     print(f"\nFaceted chart saved as '{output_path}'")


# # if __name__ == "__main__":
# #     summary_file = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
# #     create_faceted_chart(summary_filename=summary_file)




# # ==============================================================================
# # FINAL SCRIPT: FACETED BAR CHART (DEFINITIVE, READABLE VERSION)
# #
# # This script uses manual layout control to guarantee that the x-axis
# # model names are visible and readable. This is the definitive version.
# # ==============================================================================
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import numpy as np
# # --- DEFINE PROJECT PATHS ---
# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# except NameError:
#     PROJECT_ROOT = os.path.abspath('.')

# RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
# IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
# os.makedirs(IMAGES_DIR, exist_ok=True)


# def create_faceted_chart(summary_filename, model_order):
#     """
#     Reads summary data and generates a faceted bar chart with readable labels.
#     """
#     try:
#         df = pd.read_csv(summary_filename)
#     except FileNotFoundError:
#         print(f"Error: The summary file '{summary_filename}' was not found.")
#         return

#     # --- Data Preparation ---
#     id_vars = ['condition']
#     value_vars = [f'accuracy_{model}' for model in model_order if f'accuracy_{model}' in df.columns]
#     df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
#                       var_name='model_name', value_name='accuracy')
    
#     df_long['model_name'] = df_long['model_name'].str.replace('accuracy_', '').str.replace('_', '/')
    
#     cleaned_model_order = [name.replace('_', '/') for name in model_order]
    
#     # --- Create the Faceted Plot using the more controllable FacetGrid ---
#     g = sns.FacetGrid(
#         df_long, 
#         col="condition", 
#         col_wrap=3, 
#         height=5, 
#         aspect=1.2,
#         sharex=False, # This is crucial
#         sharey=True
#     )
    
#     # --- Map the plotting function to the grid ---
#     # We map the barplot, telling it to color by model name for consistency
#     g.map_dataframe(
#         sns.barplot,
#         x="model_name",
#         y="accuracy",
#         hue="model_name",
#         order=cleaned_model_order, # Ensure models are sorted by size
#         palette="deep",
#         dodge=False
#     )

#     # --- Customization ---
#     g.fig.suptitle('Model Performance by Linguistic Condition', y=1.03, fontsize=16, weight='bold')
#     g.set_axis_labels("Model", "Accuracy (%)")
#     g.set_titles("Condition: {col_name}")
#     g.set(ylim=(0, 105))
    
#     # ================== THE DEFINITIVE FIX FOR READABILITY ==================
#     # 1. Iterate through each subplot axis to apply settings individually.
#     # ================== THE DEFINITIVE FIX FOR READABILITY ==================
#     # ================== READABILITY + SINGLE SHARED X-LABEL ==================
#     # ================== READABILITY: labels only on bottom row + shared xlabel ==================
#     # Make axes 2D so we can reliably get rows/cols regardless of wrapping
#     axes_2d = np.atleast_2d(g.axes)
#     n_rows, n_cols = axes_2d.shape

#     for r, ax_row in enumerate(axes_2d):
#         for ax in ax_row:
#             if ax is None:
#                 continue

#             if r == n_rows - 1:
#                 # Bottom row: keep tick labels, rotate for readability
#                 ax.tick_params(axis='x', labelrotation=90, labelsize=8)
#             else:
#                 # Other rows: hide tick labels and per-axis x-label
#                 for tick in ax.get_xticklabels():
#                     tick.set_visible(False)
#                 ax.set_xlabel("")

#     # Use a single shared x-axis label for the whole figure
#     g.fig.text(0.5, 0.02, "Model", ha='center', va='center', fontsize=11)

#     # Leave room for the bottom-row tick labels and the shared xlabel
#     plt.subplots_adjust(bottom=0.35, top=0.9, hspace=0.4)
#     # =====================================================================


# if __name__ == "__main__":
#     summary_file = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    
#     # Logic to get the model order for plotting
#     model_sizes = {
#         'gpt2': 124, 'gpt2-medium': 355, 'EleutherAI_pythia-410m-deduped': 410,
#         'gpt2-large': 774, 'meta-llama_Llama-2-7b-hf': 7000,
#         'lmsys_vicuna-7b-v1.5': 7000, 'meta-llama_Llama-2-7b-chat-hf': 7000,
#         'lmsys_vicuna-13b-v1.3': 13000, 'meta-llama_Llama-2-13b-hf': 13000,
#         'meta-llama_Llama-3.2-1B': 1000, 'meta-llama_Meta-Llama-3.1-8B-Instruct': 8000
#     }
    
#     try:
#         temp_df = pd.read_csv(summary_file)
#         found_models = [col.replace('accuracy_', '') for col in temp_df.columns if 'accuracy_' in col]
#         sorted_model_names = sorted(found_models, key=lambda name: model_sizes.get(name, float('inf')))
#         create_faceted_chart(summary_filename=summary_file, model_order=sorted_model_names)
#     except FileNotFoundError:
#         print(f"[ERROR] Main summary file not found at '{summary_file}'. Please run the main analysis first.")



# ==============================================================================
# FINAL SCRIPT: FACETED BAR CHART (DEFINITIVE, READABLE VERSION)
#
# This script uses manual layout control to guarantee that the x-axis
# model names are visible and readable, appearing only on the bottom row.
# ==============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)


def create_faceted_chart(summary_filename, model_order):
    """
    Reads summary data and generates a faceted bar chart with readable labels,
    showing model names only on the bottom row.
    """
    try:
        df = pd.read_csv(summary_filename)
    except FileNotFoundError:
        print(f"Error: The summary file '{summary_filename}' was not found.")
        return

    # --- Data Preparation ---
    id_vars = ['condition']
    value_vars = [f'accuracy_{model}' for model in model_order if f'accuracy_{model}' in df.columns]
    df_long = pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                      var_name='model_name', value_name='accuracy')
    
    df_long['model_name'] = df_long['model_name'].str.replace('accuracy_', '').str.replace('_', '/')
    
    cleaned_model_order = [name.replace('_', '/') for name in model_order]
    
    # --- Create the Faceted Plot using the more controllable FacetGrid ---
    g = sns.FacetGrid(
        df_long, 
        col="condition", 
        col_wrap=3, 
        height=5, 
        aspect=1.2,
        sharex=False, # Crucial for individual x-axis control
        sharey=True
    )
    
    # --- Map the plotting function to the grid ---
    g.map_dataframe(
        sns.barplot,
        x="model_name",
        y="accuracy",
        hue="model_name",
        order=cleaned_model_order, 
        palette="deep",
        dodge=False
    )

    # --- Customization ---
    g.fig.suptitle('Model Performance by Linguistic Condition', y=1.03, fontsize=16, weight='bold')
    g.set_axis_labels("", "Accuracy (%)") # Remove individual x-axis labels
    g.set_titles("Condition: {col_name}")
    g.set(ylim=(0, 105))
    
    # ================== FIX FOR READABILITY: Labels only on bottom row + shared xlabel ==================
    # Make axes 2D so we can reliably get rows/cols regardless of wrapping
    axes_2d = np.atleast_2d(g.axes)
    n_rows, n_cols = axes_2d.shape

    for r, ax_row in enumerate(axes_2d):
        for ax in ax_row:
            if ax is None:
                continue

            # Hide legend from each subplot
            if ax.get_legend():
                ax.get_legend().remove()

            if r == n_rows - 1:
                # Bottom row: keep tick labels, rotate for readability
                ax.tick_params(axis='x', labelrotation=90, labelsize=8)
            else:
                # Other rows: hide tick labels
                for tick in ax.get_xticklabels():
                    tick.set_visible(False)
                ax.set_xlabel("") # Ensure no individual x-label remains

    # Create a single legend at the bottom of the figure
    handles, labels = ax.get_legend_handles_labels()
    # Ensure correct order for legend based on model_order
    ordered_handles = []
    ordered_labels = []
    for model_name in cleaned_model_order:
        try:
            idx = labels.index(model_name)
            ordered_handles.append(handles[idx])
            ordered_labels.append(labels[idx])
        except ValueError:
            pass # Model might not be in the current subplot's legend if it had no data
    
    g.fig.legend(ordered_handles, ordered_labels, loc='lower center', ncol=min(len(ordered_labels), 6), 
                 bbox_to_anchor=(0.5, 0.05), title="Model", fontsize=8, title_fontsize=9)


    # Use a single shared x-axis label for the whole figure, below the legend
    # g.fig.text(0.5, 0.02, "Model", ha='center', va='center', fontsize=11)
    
    # Adjust layout to make room for rotated labels and the shared legend
    plt.subplots_adjust(bottom=0.35, top=0.9, hspace=0.4) # Increased bottom to make space

    # --- Save the Chart ---
    output_path = os.path.join(IMAGES_DIR, "model_comparison_faceted_readable.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nFaceted chart saved as '{output_path}'")
    plt.close(g.fig) # Close the figure to free memory
    # =====================================================================


if __name__ == "__main__":
    summary_file = os.path.join(RESULTS_DIR, "model_comparison_summary.csv")
    
    # Logic to get the model order for plotting
    model_sizes = {
        'gpt2': 124, 'gpt2-medium': 355, 'EleutherAI_pythia-410m-deduped': 410,
        'gpt2-large': 774, 'meta-llama_Llama-2-7b-hf': 7000,
        'lmsys_vicuna-7b-v1.5': 7000, 'meta-llama_Llama-2-7b-chat-hf': 7000,
        'lmsys_vicuna-13b-v1.3': 13000, 'meta-llama_Llama-2-13b-hf': 13000,
        'meta-llama_Llama-3.2-1B': 1000, 'meta-llama_Meta-Llama-3.1-8B-Instruct': 8000
    }
    
    try:
        temp_df = pd.read_csv(summary_file)
        found_models = [col.replace('accuracy_', '') for col in temp_df.columns if 'accuracy_' in col]
        # Sort models based on their size for consistent ordering across plots
        sorted_model_names = sorted(found_models, key=lambda name: model_sizes.get(name, float('inf')))
        create_faceted_chart(summary_filename=summary_file, model_order=sorted_model_names)
    except FileNotFoundError:
        print(f"[ERROR] Main summary file not found at '{summary_file}'. Please run the main analysis first.")