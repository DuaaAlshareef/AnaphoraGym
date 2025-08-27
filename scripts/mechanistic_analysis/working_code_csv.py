# ==============================================================================
# FINAL PIPELINE: AUTOMATED PATCHSCOPES ON THE FULL ANAPHORAGYM CSV
#
# This version assumes the CSV is in the root directory, iterates through all
# items, runs a layer-by-layer patching experiment, and saves a clean,
# readable report.
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os

# --- 1. SETUP ---
MODEL_NAME = "gpt2"
# We will test a middle and a late layer for every item in the dataset.
LAYERS_TO_TEST = [6, 11]
# The script will look for this file in the SAME directory.
DATASET_FILENAME = "AnaphoraGym.csv"

print(f"Loading model: {MODEL_NAME}...")
# --- Correctly set up the device for your M2 Mac ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (GPU) device.")
else:
    device = torch.device("cpu")
    print("No GPU found, using CPU.")

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded.")

# --- 2. THE PATCHING MECHANISM ---
activation_storage = {}

def patch_hook(module, args, output):
    hidden_states = output[0]
    if activation_storage['mode'] == 'copy':
        source_idx = activation_storage['source_idx']
        vector_to_copy = hidden_states[0, source_idx, :].clone()
        activation_storage['vector'] = vector_to_copy
    elif activation_storage['mode'] == 'patch':
        target_idx = activation_storage['target_idx']
        if hidden_states.shape[1] == activation_storage['target_len']:
            hidden_states[0, target_idx, :] = activation_storage['vector']
    return (hidden_states,) + output[1:]

# --- 3. MAIN EXECUTION LOGIC ---
def main():
    # --- Load the full dataset ---
    try:
        df = pd.read_csv(DATASET_FILENAME)
        print(f"\nSuccessfully loaded '{DATASET_FILENAME}' with {len(df)} items.")
    except FileNotFoundError:
        print(f"\n[ERROR] Dataset not found. Please make sure '{DATASET_FILENAME}' is in the same directory as this script.")
        return

    all_results = []
    stop_token_id = tokenizer.encode("\n")[0]

    # --- Main Loop: Iterate through each row of the CSV ---
    for index, row in df.iterrows():
        condition = row['condition']
        item_num = row['item']
        
        # We will use input_1 as the source and patching_prompt_1 as the target
        source_sentence = row.get('input_1')
        patching_prompt = row.get('patching_prompt_1')

        # Skip rows that don't have the required data
        if pd.isna(source_sentence) or pd.isna(patching_prompt):
            continue

        print(f"\n=====================================================================")
        print(f"=> Processing Item: {condition} / {item_num}")
        print(f"=====================================================================")

        # --- Tokenize and locate indices FOR THIS SPECIFIC ITEM ---
        source_ids = tokenizer.encode(source_sentence, return_tensors='pt').to(device)
        target_ids = tokenizer.encode(patching_prompt, return_tensors='pt').to(device)
        
        activation_storage['source_idx'] = -2  # The last word
        activation_storage['target_idx'] = -1  # The very last token of the prompt
        activation_storage['target_len'] = target_ids.shape[1]

        # --- Inner Loop: Iterate through the layers to test ---
        for layer_num in LAYERS_TO_TEST:
            print(f"  -> Testing patch from Layer {layer_num}...")
            
            layer_to_hook = model.transformer.h[layer_num]

            # --- Run 1: Source Run (Copy) ---
            activation_storage['mode'] = 'copy'
            hook_handle = layer_to_hook.register_forward_hook(patch_hook)
            with torch.no_grad():
                model(source_ids, output_hidden_states=True)
            hook_handle.remove()

            # --- Run 2: Patched Generation ---
            activation_storage['mode'] = 'patch'
            activation_storage['layer_num'] = layer_num # Pass layer info to hook for debugging
            hook_handle = layer_to_hook.register_forward_hook(patch_hook)
            with torch.no_grad():
                patched_generated_ids = model.generate(
                    target_ids,
                    max_new_tokens=15,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
                    eos_token_id=stop_token_id
                )
            hook_handle.remove()

            # --- Analyze and store the result ---
            readout = tokenizer.decode(patched_generated_ids[0][target_ids.shape[1]:], skip_special_tokens=True).strip()
            
            print(f"     Readout: '{readout}'")

            all_results.append({
                'condition': condition,
                'item': item_num,
                'source_sentence': source_sentence,
                'patching_prompt': patching_prompt,
                'layer': layer_num,
                'readout': readout if readout else "[NO NEW TEXT GENERATED]"
            })

    # --- 4. FINAL SAVE ---
    print("\n\n--- FULL ANALYSIS COMPLETE ---")
    if not all_results:
        print("No experiments were run (check if your CSV has input_1 and patching_prompt_1 columns).")
        return

    # Define a clean, readable column order
    column_order = [
        'condition', 'item', 'layer', 
        'source_sentence', 'patching_prompt', 'readout'
    ]
    results_df = pd.DataFrame(all_results, columns=column_order)
    
    output_filename = f"AnaphoraGym_Full_Patchscope_Results_{MODEL_NAME}.csv"
    
    try:
        results_df.to_csv(output_filename, index=False)
        print(f"Final report with {len(results_df)} results saved to '{output_filename}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save final report. Reason: {e}")


if __name__ == "__main__":
    main()