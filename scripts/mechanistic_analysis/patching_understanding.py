# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # ==============================================================================
# # 1. SETUP: Load Model and Define Inputs
# # ==============================================================================
# MODEL_NAME = "gpt2"
# print(f"Loading model: {MODEL_NAME}...")
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# # Set the model to evaluation mode (disables dropout, etc.)
# model.eval()
# print("Model loaded.")

# # Define our source and target sentences
# source_sentence = "Alex passed Bo, but not Charlie."
# target_sentence = "Sam didn’t pass Ricky; Cory didn’t pass Harvey; Kim didn’t pass Taylor; "

# # ==============================================================================
# # 2. LOCATING TOKENS: Find the exact index of the tokens we'll use
# # ==============================================================================
# # We need to find the integer index of our source and target tokens in the tokenized list.

# source_tokens = tokenizer.tokenize(source_sentence)
# # The source token is the last one in the sentence
# source_patch_token_index = len(source_tokens) - 1

# target_tokens = tokenizer.tokenize(target_sentence)
# # The target token is also the last one
# target_patch_token_index = len(target_tokens) - 1

# print("\n--- Token Analysis ---")
# print(f"Source tokens: {source_tokens}")
# print(f"Token to copy from: '{source_tokens[source_patch_token_index]}' at index {source_patch_token_index}")
# print(f"Target tokens: {target_tokens}")
# print(f"Token to patch over: '{target_tokens[target_patch_token_index]}' at index {target_patch_token_index}")

# # Encode the sentences into tensor IDs for the model
# source_ids = tokenizer.encode(source_sentence, return_tensors='pt')
# target_ids = tokenizer.encode(target_sentence, return_tensors='pt')

# # ==============================================================================
# # 3. THE PATCHING MECHANISM: Using a PyTorch Hook
# # ==============================================================================
# # This dictionary will hold the activation we copy
# activation_storage = {}

# # The hook function is the core of the patch. It runs during the forward pass.
# def patch_hook(module, args, output):
#     # GPT-2's layer output is a tuple; the hidden states are the first element.
#     hidden_states = output[0]
    
#     if activation_storage['mode'] == 'copy':
#         # In copy mode, we grab the activation vector at the specified token index.
#         vector_to_copy = hidden_states[0, source_patch_token_index, :].clone()
#         activation_storage['vector'] = vector_to_copy
#         print(f"  -> Copied activation from token '{source_tokens[source_patch_token_index]}'")
    
#     elif activation_storage['mode'] == 'patch':
#         # In patch mode, we overwrite the activation at the target token index.
#         hidden_states[0, target_patch_token_index, :] = activation_storage['vector']
#         print(f"  -> Patched activation onto token '{target_tokens[target_patch_token_index]}'")
    
#     # Return the modified output to continue the forward pass
#     return (hidden_states,) + output[1:]

# # We will patch at a middle layer. GPT-2 small has 12 layers (0-11).
# PATCH_LAYER = 10
# # The specific module to hook into
# layer_to_hook = model.transformer.h[PATCH_LAYER]

# # ==============================================================================
# # 4. THE PIPELINE: Execute the copy, patch, and control runs
# # ==============================================================================

# # --- Run 1: Source Run (to copy the activation) ---
# print("\n--- Running Source Sentence to Copy Activation ---")
# activation_storage['mode'] = 'copy'
# hook_handle = layer_to_hook.register_forward_hook(patch_hook)
# with torch.no_grad():
#     model(source_ids)
# hook_handle.remove() # Always remove hooks after use

# # --- Run 2: Patched Run (to get patched predictions) ---
# print("\n--- Running Target Sentence with Patch ---")
# activation_storage['mode'] = 'patch'
# hook_handle = layer_to_hook.register_forward_hook(patch_hook)
# with torch.no_grad():
#     patched_outputs = model(target_ids)
# hook_handle.remove()

# # --- Run 3: Control Run (to see normal behavior) ---
# print("\n--- Running Target Sentence without Patch (Control) ---")
# with torch.no_grad():
#     unpatched_outputs = model(target_ids)

# # ==============================================================================
# # 5. ANALYSIS: Compare the predictions
# # ==============================================================================

# def get_top_k_predictions(logits, k=5):
#     # We only care about the predictions for the VERY LAST token in the sequence.
#     last_token_logits = logits[0, -1, :]
#     probabilities = torch.softmax(last_token_logits, dim=-1)
#     top_k_probs, top_k_indices = torch.topk(probabilities, k)
#     top_k_tokens = [tokenizer.decode(i.item()) for i in top_k_indices]
#     return list(zip(top_k_tokens, top_k_probs.tolist()))

# # Extract the final logits from the model outputs
# patched_logits = patched_outputs.logits
# unpatched_logits = unpatched_outputs.logits

# print("\n\n--- PREDICTION ANALYSIS ---")
# print("\nOriginal (Unpatched) Next Token Predictions for '...Alex didn’t see':")
# unpatched_preds = get_top_k_predictions(unpatched_logits)
# for token, prob in unpatched_preds:
#     print(f"  '{token}' (Probability: {prob:.2%})")
    
# print("\nPatched Next Token Predictions for '...Alex didn’t see':")
# patched_preds = get_top_k_predictions(patched_logits)
# for token, prob in patched_preds:
#     print(f"  '{token}' (Probability: {prob:.2%})")




# ==============================================================================
# SCRIPT FOR GENERATIVE PATCHSCOPES (MANUAL HOOK METHOD)
#
# This script correctly implements free generation after a patch to get a
# multi-token readout, and iterates through multiple source layers.
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# --- 1. SETUP ---
MODEL_NAME = "gpt2"
print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# CRITICAL FOR GENERATION: Set pad token to eos token
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded.")

source_sentence = "Alex passed Bo, but not Charlie."
target_sentence = "Sam didn’t pass Ricky; Cory didn’t pass Harvey; Kim didn’t pass Taylor; who didn't pass Charlie ? "

LAYERS_TO_TEST = [6, 11] # Test a middle and a late layer

# --- 2. LOCATE TOKENS ---
source_tokens = tokenizer.tokenize(source_sentence)
source_patch_token_index = len(source_tokens) - 1 # Index of the period '.'

target_tokens = tokenizer.tokenize(target_sentence)
target_patch_token_index = len(target_tokens) - 1 # Index of the final token "that"

source_ids = tokenizer.encode(source_sentence, return_tensors='pt')
target_ids = tokenizer.encode(target_sentence, return_tensors='pt')

# --- 3. THE PATCHING MECHANISM ---
activation_storage = {}

def patch_hook(module, args, output):
    hidden_states = output[0]
    if activation_storage['mode'] == 'copy':
        vector_to_copy = hidden_states[0, source_patch_token_index, :].clone()
        activation_storage['vector'] = vector_to_copy
    elif activation_storage['mode'] == 'patch':
        # This hook will fire multiple times during generation.
        # We only want to patch the *prompt* part, not the newly generated tokens.
        # So, we only patch if the sequence length is the same as our target prompt's length.
        if hidden_states.shape[1] == target_ids.shape[1]:
            hidden_states[0, target_patch_token_index, :] = activation_storage['vector']
            print(f"  -> Patched activation onto token '{target_tokens[target_patch_token_index]}'")
    return (hidden_states,) + output[1:]


# --- 4. THE PIPELINE ---

# First, get the unpatched control generation for comparison
print("\n--- Running Target Sentence without Patch (Control) ---")
with torch.no_grad():
    unpatched_generated_ids = model.generate(target_ids, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id)
unpatched_readout = tokenizer.decode(unpatched_generated_ids[0][target_ids.shape[1]:], skip_special_tokens=True)
print(f"  > Normal Generation: '{target_sentence}{unpatched_readout}'")


# Now, loop through the layers to test
results = []
for layer_num in LAYERS_TO_TEST:
    print(f"\n=============================================================")
    print(f"=> TESTING PATCH FROM LAYER: {layer_num}")
    print(f"=============================================================")

    # Define the specific layer to hook for this iteration
    layer_to_hook = model.transformer.h[layer_num]

    # --- Run 1: Source Run (Copy from the current layer) ---
    print(f"  Copying activation from Layer {layer_num}...")
    activation_storage['mode'] = 'copy'
    hook_handle = layer_to_hook.register_forward_hook(patch_hook)
    with torch.no_grad():
        model(source_ids)
    hook_handle.remove()

    # --- Run 2: Patched Generation ---
    print(f"  Generating from target, patching into Layer {layer_num}...")
    activation_storage['mode'] = 'patch'
    hook_handle = layer_to_hook.register_forward_hook(patch_hook)
    with torch.no_grad():
        # We now use model.generate() to get a multi-token output
        patched_generated_ids = model.generate(
            target_ids,
            max_new_tokens=10, # Generate up to 15 new tokens
            pad_token_id=tokenizer.eos_token_id
        )
    hook_handle.remove()

    # --- Analyze the result for this layer ---
    # Decode only the newly generated tokens, skipping the prompt
    readout = tokenizer.decode(patched_generated_ids[0][target_ids.shape[1]:], skip_special_tokens=True)
    
    print(f"\n  Readout from Layer {layer_num}:")
    print(f"  > '{readout.strip()}'")

    results.append({
        'layer': layer_num,
        'readout': readout.strip() if readout.strip() else "[NO NEW TEXT GENERATED]"
    })

# --- 5. FINAL SUMMARY ---
print("\n\n--- FINAL SUMMARY OF GENERATIVE READOUTS ---")
results_df = pd.DataFrame(results)
results_df['unpatched_baseline'] = unpatched_readout.strip()
print(results_df)