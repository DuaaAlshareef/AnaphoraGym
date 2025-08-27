# import pandas as pd
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
# import torch.nn.functional as F
# import re
# import os
# import yaml

# # --- 1. DEFINE PROJECT PATHS ---
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'AnaphoraGym.csv')
# RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
# CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config.yaml')
# os.makedirs(RESULTS_DIR, exist_ok=True)

# # --- The Core Log-Likelihood Function (no changes needed) ---
# def calculate_llh(model, tokenizer, input_text, continuation_text):
#     # ... (This function is unchanged)
#     if not isinstance(input_text, str) or not isinstance(continuation_text, str): return float('nan')
#     device = model.device
#     full_text = input_text + continuation_text
#     input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
#     input_only_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
#     with torch.no_grad():
#         outputs = model(input_ids)
#         logits = outputs.logits
#     continuation_start_index = input_only_ids.shape[1]
#     logits_for_continuation = logits[:, continuation_start_index - 1:-1, :]
#     continuation_token_ids = input_ids[:, continuation_start_index:]
#     if continuation_token_ids.shape[1] == 0: return 0.0
#     log_probs = F.log_softmax(logits_for_continuation, dim=2)
#     true_token_log_probs = torch.gather(log_probs, 2, continuation_token_ids.unsqueeze(-1)).squeeze(-1)
#     average_llh = true_token_log_probs.sum() / continuation_token_ids.shape[1]
#     return average_llh.item()

# def main(model_name):
#     print(f"--- Running AnaphoraGym Assessment for: {model_name} ---")
#     try: df = pd.read_csv(DATASET_PATH)
#     except FileNotFoundError: print(f"[ERROR] Dataset not found at '{DATASET_PATH}'."); return
    
#     # --- Automatic Device Selection ---
#     if torch.backends.mps.is_available(): device = torch.device("mps")
#     elif torch.cuda.is_available(): device = torch.device("cuda")
#     else: device = torch.device("cpu")
#     print(f"Using device: {device}")
    
#     # --- Automatic Quantization for Large Models ---
#     model_kwargs = {}
#     # If the model is large and we have a CUDA GPU, use 8-bit mode.
#     # (Adjust the size threshold as needed)
#     large_model_keywords = ['6b', '7b', '8b', '13b']
#     if any(keyword in model_name.lower() for keyword in large_model_keywords) and device.type == 'cuda':
#         model_kwargs['load_in_8bit'] = True
#         model_kwargs['device_map'] = 'auto'
#         print("Large model detected on CUDA. Loading in 8-bit mode.")

#     try:
#         model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
#         if not model_kwargs: # If not using device_map, manually move the model
#             model.to(device)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#     except Exception as e: print(f"[ERROR] Could not load model. Reason: {e}"); return

#     results = []
#     # --- Main Processing Loop (No Changes Needed) ---
#     for index, row in df.iterrows():
#         print(f"=> Processing Condition: {row['condition']}, Item: {row['item']}")
#         for i in range(1, row['n_tests'] + 1):
#             test_col_name, test_definition = f'test_{i}', row.get(f'test_{i}')
#             if pd.isna(test_definition): continue
#             try:
#                 match = re.match(r'(\d+)\|(\d+)>(\d+)\|(\d+)', test_definition.strip())
#                 left_cont_idx, left_input_idx, right_cont_idx, right_input_idx = map(int, match.groups())
#             except (ValueError, TypeError, AttributeError): continue
#             left_input, left_continuation = row[f'input_{left_input_idx}'], row[f'continuation_{left_cont_idx}']
#             right_input, right_continuation = row[f'input_{right_input_idx}'], row[f'continuation_{right_cont_idx}']
#             if not all(isinstance(s, str) for s in [left_input, left_continuation, right_input, right_continuation]): continue
#             llh_left = calculate_llh(model, tokenizer, left_input, left_continuation)
#             llh_right = calculate_llh(model, tokenizer, right_input, right_continuation)
#             log_odds = llh_left - llh_right
#             test_passed = log_odds > 0
#             results.append({'model_source': model_name, 'condition': row['condition'], 'item': row['item'], 'test_name': test_col_name, 'test_definition': test_definition, 'LLH_left': llh_left, 'LLH_right': llh_right, 'logOdds': log_odds, 'test_passed': test_passed})
#     # --- End of Loop ---

#     results_df = pd.DataFrame(results)
#     safe_model_name = model_name.replace('/', '_')
#     output_filename = f"AnaphoraGym_Results_{safe_model_name}.csv"
#     output_path = os.path.join(RESULTS_DIR, output_filename)
#     results_df.to_csv(output_path, index=False)
#     print(f"\nResults for {model_name} saved to '{output_path}'")

# if __name__ == "__main__":
#     # --- Load Configuration from YAML ---
#     try:
#         with open(CONFIG_PATH, 'r') as f:
#             config = yaml.safe_load(f)
#         model_to_run = config['model_to_run']
#     except (FileNotFoundError, KeyError):
#         print(f"[ERROR] Could not find 'model_to_run' key in '{CONFIG_PATH}'.")
#         exit()

#     # --- Run the main function with the loaded model name ---
#     main(model_to_run)





# ==============================================================================
# SCRIPT 1: BEHAVIORAL ASSESSMENT ENGINE

# This script is designed to be called from a shell script.
# It takes one argument--the model name--and runs the full targeted
# assessment for that single model, saving the results.
# ==============================================================================



# ==============================================================================
# FINAL SCRIPT: TARGETED ASSESSMENT ENGINE (CORRECT PATHS)
#
# This script is the core engine for the behavioral assessment. It takes a
# --model argument and uses the correct, robust pathing to find the dataset
# and save the results to the appropriate directory.
# ==============================================================================

import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import re
import argparse
import os

# --- 1. DEFINE PROJECT PATHS ---
# This block uses the provided logic to make the script robust.
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # Fallback for interactive environments
    PROJECT_ROOT = os.path.abspath('.')

DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'AnaphoraGym.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')

# Ensure the results directory exists before saving
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- The Core Log-Likelihood Function (Unchanged) ---
def calculate_llh(model, tokenizer, input_text, continuation_text):
    if not isinstance(input_text, str) or not isinstance(continuation_text, str): return float('nan')
    device = model.device
    full_text = input_text + continuation_text
    input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    input_only_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits
    continuation_start_index = input_only_ids.shape[1]
    logits_for_continuation = logits[:, continuation_start_index - 1:-1, :]
    continuation_token_ids = input_ids[:, continuation_start_index:]
    if continuation_token_ids.shape[1] == 0: return 0.0
    log_probs = F.log_softmax(logits_for_continuation, dim=2)
    true_token_log_probs = torch.gather(log_probs, 2, continuation_token_ids.unsqueeze(-1)).squeeze(-1)
    average_llh = true_token_log_probs.sum() / continuation_token_ids.shape[1]
    return average_llh.item()

def main(model_name):
    """
    Main function that takes model_name as an argument.
    """
    print(f"--- Running AnaphoraGym Assessment for: {model_name} ---")
    
    # --- Load the full dataset using the correct DATASET_PATH variable ---
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Successfully loaded dataset from '{DATASET_PATH}'")
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at the specified path: '{DATASET_PATH}'.")
        return
    
    # --- Load Model ---
    try:
        print(f"Loading model: {model_name}...")
        if torch.backends.mps.is_available(): device = torch.device("mps")
        elif torch.cuda.is_available(): device = torch.device("cuda")
        else: device = torch.device("cpu")
        print(f"Using device: {device}")
        
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Model successfully loaded on {device}.")
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_name}'. Reason: {e}")
        return
        
    results = []
    # --- Main experiment loop ---
    for index, row in df.iterrows():
        print(f"=> Processing Condition: {row['condition']}, Item: {row['item']}")
        for i in range(1, row['n_tests'] + 1):
            test_col_name, test_definition = f'test_{i}', row.get(f'test_{i}')
            if pd.isna(test_definition): continue
            try:
                match = re.match(r'(\d+)\|(\d+)>(\d+)\|(\d+)', test_definition.strip())
                left_cont_idx, left_input_idx, right_cont_idx, right_input_idx = map(int, match.groups())
            except (ValueError, TypeError, AttributeError): continue
            
            left_input = row.get(f'input_{left_input_idx}')
            left_continuation = row.get(f'continuation_{left_cont_idx}')
            right_input = row.get(f'input_{right_input_idx}')
            right_continuation = row.get(f'continuation_{right_cont_idx}')

            if not all(isinstance(s, str) for s in [left_input, left_continuation, right_input, right_continuation]): continue

            llh_left = calculate_llh(model, tokenizer, left_input, left_continuation)
            llh_right = calculate_llh(model, tokenizer, right_input, right_continuation)
            log_odds = llh_left - llh_right
            test_passed = log_odds > 0
            results.append({'model_source': model_name, 'condition': row['condition'], 'item': row['item'], 'test_name': test_col_name, 'test_definition': test_definition, 'LLH_left': llh_left, 'LLH_right': llh_right, 'logOdds': log_odds, 'test_passed': test_passed})

    # --- Save the results to the correct RESULTS_DIR folder ---
    if not results:
        print("No valid results were generated.")
        return
        
    results_df = pd.DataFrame(results)
    
    # Sanitize the model name for use in the filename
    safe_model_name = model_name.replace('/', '_')
    output_filename = f"AnaphoraGym_Results_{safe_model_name}.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nResults for {model_name} saved to '{output_path}'")


if __name__ == "__main__":
    # This part correctly parses the --model argument from the command line
    parser = argparse.ArgumentParser(description="Run targeted assessment for a specific language model.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the Hugging Face model to test (e.g., 'gpt2')."
    )
    args = parser.parse_args()
    main(model_name=args.model)