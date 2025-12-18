import pandas as pd
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg') # This line must be *before* 'import matplotlib.pyplot as plt'
import matplotlib.pyplot as plt


# --- Configuration ---
# IMPORTANT: Replace this with the actual path to your CSV file
CSV_FILE_PATH = 'dataset/AnaphoraGym.csv' # Or the full path, e.g., '/path/to/your/AnaphoraGym-ProofOfConcept-2025 - input_data (1).csv'
OUTPUT_IMAGE_PATH = 'images/dataset_overview_for_poster.png' # Name of the output image file

# --- Load Data ---
try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Successfully loaded data from: {CSV_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: The file '{CSV_FILE_PATH}' was not found.")
    print("Please check the path and filename.")
    exit() # Exit the script if the file isn't found
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- Data Preprocessing and Aggregation ---

# Ensure 'n_tests' is numeric, coercing errors to NaN
# This is crucial because your sample data showed an empty string for n_tests for 'stripping_equals_VPE',1
df['n_tests'] = pd.to_numeric(df['n_tests'], errors='coerce')

# Drop rows where 'n_tests' is NaN after conversion (if they represent invalid test counts)
# Or, you might want to fill them with 0 if an empty 'n_tests' means zero tests.
# For a poster, dropping might be clearer unless 0 tests is a meaningful category.
df.dropna(subset=['n_tests'], inplace=True)

# Calculate Number of Samples (unique items) per Condition
samples_per_condition = df.groupby('condition')['item'].nunique().reset_index()
samples_per_condition.rename(columns={'item': 'num_samples'}, inplace=True)

# Calculate Total Number of Tests per Condition
tests_per_condition = df.groupby('condition')['n_tests'].sum().reset_index()
tests_per_condition.rename(columns={'n_tests': 'total_tests'}, inplace=True)

# Merge the two aggregations
summary_df = pd.merge(samples_per_condition, tests_per_condition, on='condition')

# Sort by condition name for consistent plotting
summary_df.sort_values('condition', inplace=True)

print("\nAggregated Data for Plotting:")
print(summary_df)

# --- Plotting for Poster ---

# Set a stylish plotting aesthetic for a poster
sns.set_theme(style="whitegrid", palette="viridis", font_scale=1.4) # Increased font_scale

# Create the figure and a set of subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=False) # Two columns, separate Y-axes

# Plot 1: Number of Samples per Condition
sns.barplot(ax=axes[0], x='num_samples', y='condition', data=summary_df, orient='h', hue='num_samples', dodge=False, palette='Blues_r')
axes[0].set_title('Number of Samples (Items) per Condition', fontsize=18, fontweight='bold')
axes[0].set_xlabel('Number of Samples', fontsize=16)
axes[0].set_ylabel('Condition', fontsize=16)
axes[0].tick_params(axis='x', labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(axis='x', linestyle='--', alpha=0.7)
# Add value labels to the bars
for index, value in enumerate(summary_df['num_samples']):
    axes[0].text(value + 0.1, index, str(int(value)), color='black', va='center', fontsize=12)
axes[0].legend_.remove() # Remove legend if hue is just for color variation


# Plot 2: Total Number of Tests per Condition
sns.barplot(ax=axes[1], x='total_tests', y='condition', data=summary_df, orient='h', hue='total_tests', dodge=False, palette='Reds_r')
axes[1].set_title('Total Number of Tests per Condition', fontsize=18, fontweight='bold')
axes[1].set_xlabel('Total Number of Tests', fontsize=16)
axes[1].set_ylabel('', fontsize=16) # Keep y-label empty as it's shared implicitly
axes[1].tick_params(axis='x', labelsize=14)
axes[1].tick_params(axis='y', labelsize=14)
axes[1].grid(axis='x', linestyle='--', alpha=0.7)
# Add value labels to the bars
for index, value in enumerate(summary_df['total_tests']):
    axes[1].text(value + 0.1, index, str(int(value)), color='black', va='center', fontsize=12)
axes[1].legend_.remove() # Remove legend if hue is just for color variation


plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.suptitle('AnaphoraGym Dataset Overview', fontsize=22, fontweight='bold', y=0.98) # Main title for the figure

# Save the figure
plt.savefig(OUTPUT_IMAGE_PATH, dpi=300, bbox_inches='tight')
print(f"\nFigure saved to: {OUTPUT_IMAGE_PATH}")

# Display the plot
plt.show()


