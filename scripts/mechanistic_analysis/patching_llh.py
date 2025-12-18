# ==============================================================================
# FINAL SCRIPT: ADAPTIVE GENERATIVE PATCHING ENGINE (LOG-LIKELIHOOD MODE)
#               Designed for AnaphoraGym with specific CSV structure.
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
import argparse
import torch.nn.functional as F
import re # For parsing test definitions

# --- 1. DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

# Assuming your input_data.csv is in the 'dataset' folder
DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'input_data.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'patchscopes_llh_anaphoragym') # New results directory
os.makedirs(RESULTS_DIR, exist_ok=True)
print(f"Project Root: {PROJECT_ROOT}")
print(f"Dataset Path: {DATASET_PATH}")
print(f"Results Directory: {RESULTS_DIR}")


# --- 2. THE PATCHING MECHANISM ---
# Global storage for activation vector and relevant indices/lengths
# This is used by the hooks to pass information between the source run and target run
activation_storage = {}

def copy_hook(module, args, output):
    """
    Forward hook to copy the hidden state vector from a specific token in the source run.
    Stores it in the global 'activation_storage'.
    """
    hidden_states = output[0] # Typically (batch_size, seq_len, hidden_dim)

    # Ensure batch size is 1 for consistent indexing
    if hidden_states.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 for patching. Got {hidden_states.shape[0]}")

    # Determine how to index based on dimensions
    if hidden_states.dim() == 3: # (batch, seq_len, hidden_dim)
        vector_to_copy = hidden_states[0, activation_storage['source_idx'], :].clone().cpu()
    elif hidden_states.dim() == 2 and 'source_idx' in activation_storage: # Handle (seq_len, hidden_dim) if batch dim is squeezed
        vector_to_copy = hidden_states[activation_storage['source_idx'], :].clone().cpu()
    else:
        raise ValueError(f"Unexpected hidden_state dimension or missing source_idx for copy_hook: {hidden_states.dim()}")
    
    activation_storage['vector'] = vector_to_copy
    # print(f"  [Hook] Copied vector from layer {activation_storage.get('layer_num', 'N/A')} at source_idx {activation_storage['source_idx']}.")


def patch_pre_hook(module, args):
    """
    Pre-forward hook to patch the hidden state vector at a specific token in the target run.
    Replaces it with the vector stored in 'activation_storage'.
    """
    hidden_states = args[0] # Input to the layer, typically (batch_size, seq_len, hidden_dim)

    # Ensure hidden_states has a batch dimension for consistent indexing, if it was squeezed
    if hidden_states.dim() == 2:
        hidden_states = hidden_states.unsqueeze(0) # Add batch dimension if missing
    
    # Ensure batch size is 1
    if hidden_states.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 for patching. Got {hidden_states.shape[0]} during pre-hook.")

    # Check if target sequence length matches expectation. This is important:
    # the target_len in activation_storage should correspond to the input_text's length
    # during LLH calculation.
    if hidden_states.shape[1] == activation_storage['target_len']:
        hidden_states[0, activation_storage['target_idx'], :] = activation_storage['vector'].to(hidden_states.device)
        # print(f"  [Hook] Patched layer {activation_storage.get('layer_num', 'N/A')} at target_idx {activation_storage['target_idx']} with vector from source.")
    else:
        # This is a critical warning. If lengths don't match, the patching logic might be off
        # or applied to the wrong sequence part.
        print(f"  [WARNING] Patching sequence length mismatch in layer {activation_storage.get('layer_num', 'N/A')}. Expected {activation_storage['target_len']}, got {hidden_states.shape[1]}. Patching skipped for this specific token.")
        # We still return the original hidden states so the model doesn't crash
    
    # Return the potentially modified hidden_states (remove batch dim if it was added for consistency)
    return (hidden_states.squeeze(0) if args[0].dim() == 2 else hidden_states,) + args[1:]


# --- 3. Log-Likelihood Calculation Function (from Phase 1, adapted) ---
def calculate_llh(model, tokenizer, input_text, continuation_text):
    """
    Calculates the average log-likelihood of a continuation_text given an input_text.
    This function will automatically use any active pre-hooks (like patch_pre_hook).
    """
    if not isinstance(input_text, str) or not isinstance(continuation_text, str):
        # print(f"[LLH Calc] Skipping due to non-string input. Input: '{input_text}', Cont: '{continuation_text}'")
        return float('nan')
    
    device = model.device
    full_text = input_text + continuation_text
    
    # Tokenize the full text for the model's forward pass
    full_input_ids = tokenizer.encode(full_text, return_tensors="pt").to(device)
    
    # Tokenize only the input_text to determine where the continuation begins
    input_only_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # Ensure consistent batch dimension (models typically expect batch_size=1, seq_len)
    if full_input_ids.dim() == 1: full_input_ids = full_input_ids.unsqueeze(0)
    if input_only_ids.dim() == 1: input_only_ids = input_only_ids.unsqueeze(0)

    # Perform forward pass with no gradient tracking
    with torch.no_grad():
        outputs = model(full_input_ids)
        logits = outputs.logits # (batch_size, seq_len, vocab_size)
    
    # Determine the starting index of the continuation in the full_input_ids
    continuation_start_index = input_only_ids.shape[1]
    
    # If the continuation is empty or the input_text already covers the full_text,
    # there's no continuation to predict.
    if continuation_start_index >= full_input_ids.shape[1]:
        # print(f"[LLH Calc] Continuation is empty or fully covered by input. Returning 0.0 for: '{continuation_text}'")
        return 0.0

    # Logits for predicting the continuation tokens
    # For causal LMs, logits[..., i-1, :] predict token[..., i]
    # So, we need logits from `continuation_start_index - 1` up to `end-1` to predict
    # tokens from `continuation_start_index` up to `end`.
    logits_for_continuation = logits[:, continuation_start_index - 1:-1, :]
    continuation_token_ids = full_input_ids[:, continuation_start_index:]

    # If the slicing somehow resulted in an empty tensor (e.g., due to edge cases)
    if continuation_token_ids.shape[1] == 0 or logits_for_continuation.shape[1] == 0:
        return 0.0 

    # Calculate log probabilities
    log_probs = F.log_softmax(logits_for_continuation, dim=2)
    
    # Get the log probability of the true next token at each step
    # Use torch.gather to pick the log-prob for the actual token that appeared
    true_token_log_probs = torch.gather(log_probs, 2, continuation_token_ids.unsqueeze(-1)).squeeze(-1)
    
    # Sum and average to get the average log-likelihood
    average_llh = true_token_log_probs.sum() / continuation_token_ids.shape[1]
    return average_llh.item()


# --- 4. MAIN EXECUTION LOGIC ---
def main(model_name):
    print(f"\n--- Running Patchscopes LLH Analysis for: {model_name} ---")

    # --- Load Model ---
    try:
        print(f"Loading model: {model_name}...")
        
        # Detect device and model size for optimized loading
        device = torch.device("cpu") # Default to CPU
        load_in_8bit = False
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Heuristic for large models to use 8-bit, adjust as necessary
            if any(k in model_name.lower() for k in ["7b", "8b", "13b", "70b"]):
                load_in_8bit = True
                print(f"CUDA found, and model '{model_name}' appears large. Loading in 8-bit mode with device_map='auto'.")
            else:
                print(f"CUDA found, loading model in full precision.")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"MPS found, loading model in full precision.")
        else:
            print(f"No CUDA/MPS. Using CPU (will be slow for large models).")

        if load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set pad_token for generation, useful even if not directly generating in this LLH mode
        # It's good practice for compatibility with some tokenizers/models.
        if tokenizer.pad_token is None:
             tokenizer.pad_token = tokenizer.eos_token 

        model.eval() # Set model to evaluation mode
        print(f"Model '{model_name}' loaded successfully on {device} (8-bit: {load_in_8bit}).")
    except Exception as e:
        print(f"\n[ERROR] Could not load model '{model_name}'. Reason: {e}")
        return

    # --- Dynamic Layer Selection ---
    num_layers = model.config.num_hidden_layers
    # Select representative layers: early, mid-early, middle, mid-late, late
    layers_to_test = [0] # Always include the first layer
    if num_layers > 1:
        layers_to_test.extend([
            max(0, num_layers // 4 -1), # Adjusted to avoid index -1 if num_layers < 4
            num_layers // 2,
            max(0, num_layers * 3 // 4 -1),
            max(0, num_layers - 1) # Always include the last layer
        ])
    layers_to_test = sorted(list(set([l for l in layers_to_test if 0 <= l < num_layers])))
    print(f"Detected {num_layers} layers. Will test layers: {layers_to_test}")

    # --- Load Dataset ---
    try:
        df = pd.read_csv(DATASET_PATH)
        # Define expected columns for input text parts.
        # This dataset structure is flexible; we need to extract input/continuation based on 'test_X' definition.
        # Ensure 'input_1' and 'patching_prompt_1' are always present for source/target in patching.
        required_basic_cols = ['condition', 'item', 'n_tests', 'input_1', 'patching_prompt_1']
        if not all(col in df.columns for col in required_basic_cols):
            raise ValueError(f"Dataset must contain basic columns: {required_basic_cols}")
        print(f"Successfully loaded '{DATASET_PATH}' with required columns for patching & LLH tests.")
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at '{DATASET_PATH}'. Please ensure 'input_data.csv' is in the 'dataset' folder.")
        return
    except ValueError as e:
        print(f"[ERROR] Dataset validation failed: {e}")
        return

    all_results = []

    # --- Main Loop over CSV for each AnaphoraGym Item ---
    for index, row in df.iterrows():
        condition, item_num = row['condition'], row['item']
        num_tests_for_item = row['n_tests']

        # Get the base source sentence and patching prompt from the current row
        # These will be used across all tests for this item
        source_sentence = str(row.get('input_1', ''))
        patching_prompt_base = str(row.get('patching_prompt_1', ''))

        if not source_sentence.strip() or not patching_prompt_base.strip():
            print(f"Skipping row {index} ('{condition}' / '{item_num}') due to missing source_sentence or patching_prompt_base.")
            continue

        print(f"\n=> Processing Item: {condition} / {item_num}")
        
        try:
            # --- Prepare for Patching ---
            # Tokenize source sentence ONCE per item
            source_ids = tokenizer.encode(source_sentence, return_tensors='pt').to(model.device)
            
            # The patching_prompt_base will be the 'input_text' when calling calculate_llh.
            # We need its tokenized length for target_len in activation_storage.
            target_ids_for_patching_len_check = tokenizer.encode(patching_prompt_base, return_tensors='pt').to(model.device)
            
            # Set global storage variables for copying (source) and patching (target)
            activation_storage['source_idx'] = -1 # Patch from the LAST token of source_sentence
                                                # (typically where the "anaphoric information" might be resolved)
            activation_storage['target_idx'] = -1 # Patch TO the LAST token of patching_prompt_base
            activation_storage['target_len'] = target_ids_for_patching_len_check.shape[1]


            # --- Loop through each defined 'test_X' for the current item ---
            for test_idx in range(1, int(num_tests_for_item) + 1):
                test_col_name = f'test_{test_idx}'
                test_definition_str = row.get(test_col_name)

                if pd.isna(test_definition_str):
                    print(f"  -> Skipping {test_col_name} for item {item_num}: Definition missing.")
                    continue

                # Parse test_definition to get indices for left/right input/continuation
                try:
                    match = re.match(r'(\d+)\|(\d+)>(\d+)\|(\d+)', test_definition_str.strip())
                    if not match: raise ValueError("Test definition regex mismatch.")
                    left_cont_idx, left_input_idx, right_cont_idx, right_input_idx = map(int, match.groups())
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"  -> Skipping {test_col_name} for item {item_num}: Error parsing definition '{test_definition_str}'. Reason: {e}")
                    continue

                # Retrieve the specific input and continuation texts based on parsed indices
                # Note: `input_X` for LLH comparison is still `patching_prompt_base` (the patched text).
                # `input_left_text` and `input_right_text` below refer to the inputs *from the CSV*
                # that would have been used in Phase 1's LLH calculation *without* patching.
                # Here, they help retrieve the correct continuations.

                # We assume 'left' is the plausible/correct and 'right' is the implausible/contrast.
                # The LLH calculation uses patching_prompt_base as the input prefix.
                continuation_left_text = str(row.get(f'continuation_{left_cont_idx}', ''))
                continuation_right_text = str(row.get(f'continuation_{right_cont_idx}', ''))

                if not continuation_left_text.strip() or not continuation_right_text.strip():
                    print(f"  -> Skipping {test_col_name} for item {item_num}: Missing left or right continuation text.")
                    continue

                # --- Loop through each selected layer for patching ---
                for layer_num in layers_to_test:
                    # Identify the specific layer module to hook into based on model architecture
                    model_arch = model.config.architectures[0]
                    if "LlamaForCausalLM" in model_arch: layer_to_hook = model.model.layers[layer_num]
                    elif "GPTJForCausalLM" in model_arch or "GPT2ForCausalLM" in model_arch: layer_to_hook = model.transformer.h[layer_num]
                    elif "GPTNeoXForCausalLM" in model_arch: layer_to_hook = model.gpt_neox.layers[layer_num]
                    else: raise ValueError(f"Unsupported model architecture: {model_arch}")
                    
                    activation_storage['layer_num'] = layer_num # Store for hook debugging messages

                    # --- Step 1: COPY activations from source_sentence ---
                    # This hook is temporary and only active during this model(source_ids) call
                    copy_hook_handle = layer_to_hook.register_forward_hook(copy_hook)
                    with torch.no_grad():
                        model(source_ids, output_hidden_states=True) # Run model on source to populate activation_storage['vector']
                    copy_hook_handle.remove() # Always remove the hook after use

                    # --- Step 2: Calculate LLH for plausible continuation with patching ---
                    # This hook is temporary and only active during this calculate_llh call
                    patch_hook_handle_plausible = layer_to_hook.register_forward_pre_hook(patch_pre_hook)
                    llh_plausible_patched = calculate_llh(
                        model, tokenizer, patching_prompt_base, continuation_left_text
                    )
                    patch_hook_handle_plausible.remove() # Always remove the hook

                    # --- Step 3: Calculate LLH for implausible continuation with patching ---
                    # Crucial: Re-register the hook. Hooks are one-shot or apply on each forward pass,
                    # but it's safer to ensure it's fresh for each distinct LLH calculation.
                    patch_hook_handle_implausible = layer_to_hook.register_forward_pre_hook(patch_pre_hook)
                    llh_implausible_patched = calculate_llh(
                        model, tokenizer, patching_prompt_base, continuation_right_text
                    )
                    patch_hook_handle_implausible.remove() # Always remove the hook

                    # --- Step 4: Compute Log-Odds and Test Result ---
                    log_odds_patched = llh_plausible_patched - llh_implausible_patched
                    test_passed_patched = log_odds_patched > 0

                    # Append results for this specific (item, test, layer) combination
                    all_results.append({
                        'model_source': model_name,
                        'condition': condition,
                        'item': item_num,
                        'test_name': test_col_name,
                        'test_definition': test_definition_str,
                        'layer': layer_num, # New column for layer
                        'source_sentence': source_sentence,
                        'patching_prompt': patching_prompt_base,
                        'continuation_left': continuation_left_text, # For clarity in results
                        'continuation_right': continuation_right_text, # For clarity in results
                        'LLH_left': llh_plausible_patched, # 'left' now refers to plausible
                        'LLH_right': llh_implausible_patched, # 'right' now refers to implausible
                        'logOdds': log_odds_patched,
                        'test_passed': test_passed_patched
                    })
        except Exception as e:
            print(f"  -> [CRITICAL ERROR] Failed processing item {item_num} (condition: {condition}): {e}")
            import traceback
            traceback.print_exc() # Print full traceback for critical errors
            continue # Continue to the next item

    # --- Save Final Report ---
    if not all_results:
        print("No experiments were run successfully.")
        return
        
    results_df = pd.DataFrame(all_results)
    # Sanitize model name for filename
    safe_model_name = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    output_filename = f"Patchscopes_LLH_Results_{safe_model_name}.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"\nPatchscopes LLH results for {model_name} saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Patchscopes LLH analysis for a specific language model using AnaphoraGym-like dataset.")
    parser.add_argument("--model", type=str, required=True, help="The name of the Hugging Face model to test (e.g., 'meta-llama/Meta-Llama-3.1-8B-Instruct').")
    args = parser.parse_args()
    main(model_name=args.model)