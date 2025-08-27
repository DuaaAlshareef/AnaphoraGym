# ==============================================================================
# FINAL SCRIPT - GENERATIVE PATCHING (GPT-J-6B in 8-bit Mode)
#
# This version uses the powerful, open EleutherAI/gpt-j-6b model.
# NOTE: Requires `pip install accelerate bitsandbytes`.
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# --- 1. SETUP ---
# Changed to the powerful and fully open GPT-J-6B
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."
PATCHING_PROMPT = "Sam didn’t pass Ricky; Cory didn’t pass Harvey; Kim didn’t pass Taylor; ?"

# GPT-J-6B has 28 layers (0-27). Let's pick a middle and a late layer.
LAYERS_TO_TEST = [14, 27]

print(f"Loading model: {MODEL_NAME}...")

# --- Device setup ---

device = "CUDA"
print("WARNING: No CUDA GPU found. Running on CPU will be extremely slow.")

# --- Load model in 8-bit mode to save memory ---
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True, # Enable 8-bit quantization
        device_map="auto"  # Automatically maps the model across available devices
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print("Model loaded successfully in 8-bit mode.")
except Exception as e:
    print(f"\n[ERROR] Could not load model. Reason: {e}")
    exit()

# --- 2. THE PATCHING MECHANISM ---
activation_storage = {}
def patch_hook(module, args, output):
    hidden_states = output[0]
    if activation_storage['mode'] == 'copy':
        vector_to_copy = hidden_states[0, activation_storage['source_idx'], :].clone()
        activation_storage['vector'] = vector_to_copy
    elif activation_storage['mode'] == 'patch':
        if hidden_states.shape[1] == activation_storage['target_len']:
            # The vector must be moved to the correct GPU if using multiple
            hidden_states[0, activation_storage['target_idx'], :] = activation_storage['vector'].to(hidden_states.device)
    return (hidden_states,) + output[1:]

# --- 3. THE PIPELINE ---
results = []
stop_token_id = tokenizer.encode("\n")[0]
source_ids = tokenizer.encode(SOURCE_SENTENCE, return_tensors='pt').to(device)
activation_storage['source_idx'] = -2 # "Charlie"
target_ids = tokenizer.encode(PATCHING_PROMPT, return_tensors='pt').to(device)
activation_storage['target_idx'] = -1 # "?"
activation_storage['target_len'] = target_ids.shape[1]

for layer_num in LAYERS_TO_TEST:
    print(f"\n=============================================================")
    print(f"=> TESTING PATCH FROM LAYER: {layer_num}")
    print(f"=============================================================")

    # CRITICAL: GPT-J's layers are in `model.transformer.h`, just like GPT-2
    layer_to_hook = model.transformer.h[layer_num]
    activation_storage['layer_num'] = layer_num

    # --- Run 1: Source Run (Copy) ---
    activation_storage['mode'] = 'copy'
    hook_handle = layer_to_hook.register_forward_hook(patch_hook)
    with torch.no_grad():
        model(source_ids, output_hidden_states=True)
    hook_handle.remove()

    # --- Run 2: Patched Generation ---
    activation_storage['mode'] = 'patch'
    hook_handle = layer_to_hook.register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_generated_ids = model.generate(
            target_ids[:, :-1], # Pass prompt EXCEPT for the "?"
            max_new_tokens=15,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=stop_token_id
        )
    hook_handle.remove()

    readout = tokenizer.decode(patched_generated_ids[0][target_ids.shape[1]-1:], skip_special_tokens=True)
    print(f"\n  Readout from Layer {layer_num}: > '{readout.strip()}'")
    results.append({'layer': layer_num, 'readout': readout.strip()})

# --- 4. FINAL SUMMARY ---
print("\n\n--- FINAL SUMMARY ---")
results_df = pd.DataFrame(results)
print(results_df)