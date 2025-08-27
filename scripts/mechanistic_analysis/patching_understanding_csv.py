# ==============================================================================
# SCRIPT FOR PATCHSCOPES - MANUAL HOOK METHOD ON CSV DATA
#
# This script adapts the manual hook-based patching to iterate through
# the AnaphoraGym.csv file and save the results.
# ==============================================================================
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# --- 1. SETUP: Load Model ---
MODEL_NAME = "gpt2"
PATCH_LAYER = 10 # The layer we will consistently patch at

print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded.")

# --- 2. THE PATCHING MECHANISM (Unchanged) ---
activation_storage = {}
# This will be populated inside the loop now
source_tokens = [] 

def patch_hook(module, args, output):
    hidden_states = output[0]
    if activation_storage['mode'] == 'copy':
        vector_to_copy = hidden_states[0, activation_storage['source_idx'], :].clone()
        activation_storage['vector'] = vector_to_copy
        print(f"  -> Copied activation from token '{source_tokens[activation_storage['source_idx']]}'")
    elif activation_storage['mode'] == 'patch':
        hidden_states[0, activation_storage['target_idx'], :] = activation_storage['vector']
    return (hidden_states,) + output[1:]

layer_to_hook = model.transformer.h[PATCH_LAYER]

def get_top_k_predictions(logits, k=5):
    last_token_logits = logits[0, -1, :]
    probabilities = torch.softmax(last_token_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probabilities, k)
    top_k_tokens = [tokenizer.decode(i.item()) for i in top_k_indices]
    return list(zip(top_k_tokens, top_k_probs.tolist()))

# --- 3. MAIN LOOP: Read CSV and run experiments ---
try:
    df = pd.read_csv('AnaphoraGym.csv')
    print(f"\nSuccessfully loaded 'AnaphoraGym.csv' with {len(df)} items.")
except FileNotFoundError:
    print("\n[ERROR] 'AnaphoraGym.csv' not found.")
    exit()

results = []
for index, row in df.iterrows():
    # --- Get data for this row ---
    source_sentence = row.get('input_1')
    target_sentence = row.get('patching_prompt_1')

    if pd.isna(source_sentence) or pd.isna(target_sentence):
        continue

    print(f"\n=============================================================")
    print(f"=> Processing Item: {row['condition']} / {row['item']}")
    print(f"   Source: '{source_sentence[:70]}...'")
    print(f"   Target: '{target_sentence[:70]}...'")
    print(f"=============================================================")

    # --- Locate tokens for THIS SPECIFIC row ---
    source_tokens = tokenizer.tokenize(source_sentence)
    source_patch_token_index = len(source_tokens) - 1 # Last token

    target_tokens = tokenizer.tokenize(target_sentence)
    target_patch_token_index = len(target_tokens) - 1 # Last token
    
    # Pass indices to the hook via the storage dictionary
    activation_storage['source_idx'] = source_patch_token_index
    activation_storage['target_idx'] = target_patch_token_index

    source_ids = tokenizer.encode(source_sentence, return_tensors='pt')
    target_ids = tokenizer.encode(target_sentence, return_tensors='pt')

    # --- Execute the 3-run pipeline ---
    # 1. Source Run (Copy)
    activation_storage['mode'] = 'copy'
    hook_handle = layer_to_hook.register_forward_hook(patch_hook)
    with torch.no_grad():
        model(source_ids)
    hook_handle.remove()

    # 2. Patched Run
    activation_storage['mode'] = 'patch'
    hook_handle = layer_to_hook.register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_outputs = model(target_ids)
    hook_handle.remove()
    
    # 3. Control Run
    with torch.no_grad():
        unpatched_outputs = model(target_ids)

    # --- Analyze and Store Results ---
    patched_preds = get_top_k_predictions(patched_outputs.logits)
    unpatched_preds = get_top_k_predictions(unpatched_outputs.logits)

    # For this example, we'll just store the top prediction
    results.append({
        'condition': row['condition'],
        'item': row['item'],
        'source_sentence': source_sentence,
        'top_unpatched_prediction': unpatched_preds[0][0],
        'top_patched_prediction': patched_preds[0][0],
        'patched_pred_prob': patched_preds[0][1]
    })
    
    # Print a summary for this item
    print(f"  Top Unpatched Prediction: '{unpatched_preds[0][0]}'")
    print(f"  Top Patched Prediction:   '{patched_preds[0][0]}'")

# --- 4. Save Final Report ---
results_df = pd.DataFrame(results)
output_filename = f"AnaphoraGym_ManualPatch_Results_{MODEL_NAME}.csv"
results_df.to_csv(output_filename, index=False)

print("\n--- ANALYSIS COMPLETE ---")
print(results_df.head())
print(f"\nFull report saved to '{output_filename}'")