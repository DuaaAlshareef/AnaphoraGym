# # ==============================================================================
# # FINAL SCRIPT - GENERATIVE PATCHING (CORRECTED WITH PRE-HOOK)

# # This version uses a `register_forward_pre_hook` to ensure the patch
# # is applied correctly *before* the layer's computation, fixing the issue
# # of the patch having no effect during generation.
# # ==============================================================================








import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# --- 1. SETUP ---
MODEL_NAME = "meta-llama/Llama-3.2-1B"
# SOURCE_SENTENCE = "I am so sad, I am so sad, I am so sad."

PRIMING_PREFIX = ""
EVENT_SENTENCE = "Alex passed Bo, but not Charlie. Alex passed Bo, but not Charlie. Alex passed Bo, but not Charlie."
SOURCE_SENTENCE = PRIMING_PREFIX + EVENT_SENTENCE
# SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."
# SOURCE_SENTENCE = "Alex passed Bo, but not Charlie. Alex passed Bo, but not Charlie. Alex passed Bo, but not Charlie."
# SOURCE_SENTENCE = "Alex and Charlie. Alex and Charlie. Alex and Charlie. "
# PATCHING_PROMPT = "Cory and Harvey; Kim and Taylor; Alex and "
PATCHING_PROMPT = "word:word ; door:door; 1:1; first:first; cat:cat; ?"
# PATCHING_PROMPT = "Sam didn’t pass Ricky that is to say Cory didn’t pass Harvey that is to say Kim didn’t pass Taylor that is to say "
# --- Dynamic Layer Selection ---
print(f"Loading model: {MODEL_NAME}...")
if torch.backends.mps.is_available(): device = torch.device("mps")
else: device = torch.device("cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model.eval()
print("Model loaded.")

num_layers = model.config.num_hidden_layers
print(f"Detected {num_layers} layers in this model.")
middle_layer = num_layers // 2
late_layer = max(0, num_layers - 2)
LAYERS_TO_TEST = [middle_layer, late_layer]
print(f"Will be testing layers: {LAYERS_TO_TEST}")

# --- 2. THE PATCHING MECHANISM (REVISED WITH PRE-HOOK) ---
activation_storage = {}

# This is a standard forward hook, used only for copying. It works fine.
def copy_hook(module, args, output):
    hidden_states = output[0]
    vector_to_copy = hidden_states[0, activation_storage['source_idx'], :].clone().cpu()
    activation_storage['vector'] = vector_to_copy

# This is our new PRE-HOOK for patching. It modifies the INPUT `args` to the layer.
def patch_pre_hook(module, args):
    # The hidden states are the first element in the input tuple `args`.
    hidden_states = args[0]
    
    # We only want to patch the initial prompt, not subsequent generated tokens.
    if hidden_states.shape[1] == activation_storage['target_len']:
        print(f"  -> PRE-HOOK: Patching activation at Layer {activation_storage['layer_num']}")
        hidden_states[0, activation_storage['target_idx'], :] = activation_storage['vector'].to(hidden_states.device)
    
    # Return the modified arguments as a tuple
    return (hidden_states,) + args[1:]


# --- 3. THE PIPELINE ---
results = []
stop_token_id = tokenizer.encode("\n")[0]
source_ids = tokenizer.encode(SOURCE_SENTENCE, return_tensors='pt').to(device)
target_ids = tokenizer.encode(PATCHING_PROMPT, return_tensors='pt').to(device)
activation_storage['source_idx'] = -3
activation_storage['target_idx'] = -1 
activation_storage['target_len'] = target_ids.shape[1]

# ==============================================================================
# TOKENIZATION ANALYSIS BLOCK
# This section will print a clear breakdown of the tokenization for verification.
# ==============================================================================
print("\n--- Tokenization Analysis ---")

# --- 1. Analyze the Source Sentence ---
source_tokens = tokenizer.tokenize(SOURCE_SENTENCE)
source_idx_to_check = -3 # The index we intend to copy from

print(f"Source Sentence: '{SOURCE_SENTENCE}'")
print(f"Tokenized Source ({len(source_tokens)} tokens): {source_tokens}")

# Use a try-except block to safely access the index
try:
    selected_source_token = source_tokens[source_idx_to_check]
    print(f" -> Index {source_idx_to_check} corresponds to token: '{selected_source_token}'")
except IndexError:
    print(f"[ERROR] Source sentence is too short. Index {source_idx_to_check} is out of bounds.")
    # You might want to exit() here if this is critical
    
# --- 2. Analyze the Target (Patching) Prompt ---
target_tokens = tokenizer.tokenize(PATCHING_PROMPT)
target_idx_to_check = -1 # The index we intend to patch at

print(f"\nTarget Prompt: '{PATCHING_PROMPT}'")
print(f"Tokenized Target ({len(target_tokens)} tokens): {target_tokens}")

try:
    selected_target_token = target_tokens[target_idx_to_check]
    print(f" -> Index {target_idx_to_check} corresponds to token: '{selected_target_token}'")
except IndexError:
    print(f"[ERROR] Target prompt is too short. Index {target_idx_to_check} is out of bounds.")
    # You might want to exit() here

# Add a separator before the main experiment starts
print("\n" + "="*70)
# ==============================================================================


for layer_num in LAYERS_TO_TEST:
    print(f"\n=============================================================")
    print(f"=> TESTING PATCH FROM LAYER: {layer_num}")
    print(f"=============================================================")

    # Determine the correct path to the layers
    if "Llama" in model.config.architectures[0]:
        layer_to_hook = model.model.layers[layer_num]
    else:
        layer_to_hook = model.transformer.h[layer_num]

    activation_storage['layer_num'] = layer_num

    # --- Run 1: Source Run (Copy) ---
    copy_hook_handle = layer_to_hook.register_forward_hook(copy_hook)
    with torch.no_grad():
        model(source_ids, output_hidden_states=True)
    copy_hook_handle.remove()

    # --- Run 2: Patched Generation ---
    # We now register the PRE-HOOK for patching.
    patch_hook_handle = layer_to_hook.register_forward_pre_hook(patch_pre_hook)
    with torch.no_grad():
        patched_generated_ids = model.generate(
            target_ids, # We pass the full prompt now
            max_new_tokens=15,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            eos_token_id=stop_token_id
        )
    patch_hook_handle.remove()

    readout = tokenizer.decode(patched_generated_ids[0][target_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n  Readout from Layer {layer_num}: > '{readout.strip()}'")
    results.append({'layer': layer_num, 'readout': readout.strip()})

# --- 4. FINAL SUMMARY ---
print("\n\n--- FINAL SUMMARY ---")
results_df = pd.DataFrame(results)
print(results_df)



