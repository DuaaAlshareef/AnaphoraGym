# ==============================================================================
# FINAL SCRIPT: ADAPTIVE GENERATIVE PATCHING ENGINE (CORRECTED LOOP LOGIC)
#
# This definitive version correctly uses the source and target sentences
# from each row of the CSV file inside the main loop.
# ==============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import os
import argparse

# --- 1. DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'AnaphoraGym.csv')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'mechanistic_analysis')
os.makedirs(RESULTS_DIR, exist_ok=True)


# --- 2. THE PATCHING MECHANISM ---
activation_storage = {}
def copy_hook(module, args, output):
    hidden_states = output[0]
    if hidden_states.dim() == 3:
        vector_to_copy = hidden_states[0, activation_storage['source_idx'], :].clone().cpu()
    elif hidden_states.dim() == 2:
        vector_to_copy = hidden_states[activation_storage['source_idx'], :].clone().cpu()
    else:
        raise ValueError(f"Unexpected hidden_state dimension: {hidden_states.dim()}")
    activation_storage['vector'] = vector_to_copy

def patch_pre_hook(module, args):
    hidden_states = args[0]
    if hidden_states.shape[1] == activation_storage['target_len']:
        hidden_states[0, activation_storage['target_idx'], :] = activation_storage['vector'].to(hidden_states.device)
    return (hidden_states,) + args[1:]


# --- 3. MAIN EXECUTION LOGIC ---
def main(model_name):
    print(f"--- Running Patchscopes Analysis for: {model_name} ---")

    # --- Load Model ---
    try:
        print(f"Loading model: {model_name}...")
        is_large_model = any(k in model_name for k in ["6b", "7b", "8b", "12b", "13b"])
        
        if torch.cuda.is_available() and is_large_model:
            print("CUDA found. Loading large model in 8-bit mode.")
            model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")
        else:
            if torch.backends.mps.is_available(): device = torch.device("mps")
            else: device = torch.device("cpu")
            print(f"Using device: {device}. Loading model in full precision.")
            model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Could not load model '{model_name}'. Reason: {e}")
        return

    # --- Dynamic Layer Selection ---
    num_layers = model.config.num_hidden_layers
    middle_layer = num_layers // 2
    late_layer = max(0, num_layers - 2)
    layers_to_test = [middle_layer, late_layer]
    print(f"Detected {num_layers} layers. Will test layers: {layers_to_test}")

    # --- Load Dataset ---
    try:
        df = pd.read_csv(DATASET_PATH)
        print(f"Successfully loaded '{DATASET_PATH}'")
    except FileNotFoundError:
        print(f"[ERROR] Dataset not found at '{DATASET_PATH}'.")
        return

    all_results = []
    stop_token_id = tokenizer.encode("\n")[0]

    # --- Main Loop over CSV ---
    for index, row in df.iterrows():
        condition, item_num = row['condition'], row['item']
        
        # ================== THIS IS THE CRITICAL PART ==================
        # Get the source sentence (the input for the patch) from the current row
        source_sentence = str(row.get('input_1', '')) # Ensure it's a string, default to empty
        
        # Get the patching prompt (the sentence to be patched) from the current row
        # Ensure it's a string before concatenation
        patching_prompt_base = str(row.get('patching_prompt_1', '')) 
        patching_prompt = patching_prompt_base + " ?"
        # ===============================================================

        # Now, check for emptiness after ensuring they are strings
        if not source_sentence.strip() or not patching_prompt_base.strip(): # Check if effectively empty
            print(f"Skipping row {index} due to missing source or patching prompt.")
            continue

        print(f"\n=> Processing Item: {condition} / {item_num}")
        try:
            # ================== MOVED INSIDE THE LOOP ==================
            # Encode the sentences for the CURRENT item
            source_ids = tokenizer.encode(source_sentence, return_tensors='pt').to(model.device)
            target_ids = tokenizer.encode(patching_prompt, return_tensors='pt').to(model.device)
            
            # Set the storage variables for the CURRENT item
            activation_storage['source_idx'] = -2
            activation_storage['target_idx'] = -1
            activation_storage['target_len'] = target_ids.shape[1]
            # ===========================================================

            for layer_num in layers_to_test:
                print(f"  -> Testing patch from Layer {layer_num}...")
                
                model_arch = model.config.architectures[0]
                if "Llama" in model_arch: layer_to_hook = model.model.layers[layer_num]
                elif "GPTJ" in model_arch or "GPT2" in model_arch: layer_to_hook = model.transformer.h[layer_num]
                elif "GPTNeoX" in model_arch: layer_to_hook = model.gpt_neox.layers[layer_num]
                else: raise ValueError(f"Unsupported model architecture: {model_arch}")
                
                activation_storage['layer_num'] = layer_num

                copy_hook_handle = layer_to_hook.register_forward_hook(copy_hook)
                with torch.no_grad(): model(source_ids, output_hidden_states=True)
                copy_hook_handle.remove()

                patch_hook_handle = layer_to_hook.register_forward_pre_hook(patch_pre_hook)
                with torch.no_grad():
                    patched_generated_ids = model.generate(
                        target_ids, max_new_tokens=15, pad_token_id=tokenizer.eos_token_id,
                        do_sample=False, eos_token_id=stop_token_id
                    )
                patch_hook_handle.remove()

                readout = tokenizer.decode(patched_generated_ids[0][target_ids.shape[1]:], skip_special_tokens=True).strip()
                all_results.append({
                    'condition': condition, 'item': item_num, 'layer': layer_num,
                    'source_sentence': source_sentence, 'patching_prompt': patching_prompt,
                    'readout': readout if readout else "[NO NEW TEXT GENERATED]"
                })
        except Exception as e:
            print(f"  -> [ERROR] Failed on this item: {e}")
            continue

    # --- Save Final Report ---
    if not all_results:
        print("No experiments were run successfully.")
        return
        
    results_df = pd.DataFrame(all_results)
    safe_model_name = model_name.replace('/', '_')
    output_filename = f"AnaphoraGym_Patchscope_Results_{safe_model_name}.csv"
    output_path = os.path.join(RESULTS_DIR, output_filename)
    results_df.to_csv(output_path, index=False)
    print(f"\nFull report saved to '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Patchscopes analysis for a specific language model.")
    parser.add_argument("--model", type=str, required=True, help="The name of the Hugging Face model to test.")
    args = parser.parse_args()
    main(model_name=args.model)