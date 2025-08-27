# ==============================================================================
# FINAL SCRIPT: ANAPHORAGYM PATCHING (OFFICIAL TOOLKIT + TOKEN ANALYSIS)
#
# This version uses the robust `inspect` function and adds a clear
# tokenization analysis block for verification and debugging.
# ==============================================================================

import torch
import pandas as pd

# --- 1. Import from the local utility files ---
from scripts.mechanistic_analysis.general_utils import ModelAndTokenizer
from scripts.mechanistic_analysis.patchscopes_utils import inspect, set_hs_patch_hooks_gptj, set_hs_patch_hooks_llama

# --- 2. Configuration ---
MODEL_NAME = "meta-llama/Llama-3.2-1B"
# SOURCE_SENTENCE = "Alex passed Bo, but not Charlie. Alex passed Bo, but not Charlie."
# SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."
SOURCE_SENTENCE = "I am so sad, I am so sad, I am so sad."
# PATCHING_PROMPT = "Sam didn’t pass Ricky; Cory didn’t pass Harvey; Kim didn’t pass Taylor; ?"
PATCHING_PROMPT = "word:word ; door:door; 1:1; first:first; cat:cat; emotion:?"

LAYERS_TO_TEST = [6, 15]

# This dictionary maps model architecture names to the correct hook-setting function
MODEL_FAMILY_TO_HOOK_SETTER = {
    'GPT2LMHeadModel': set_hs_patch_hooks_gptj,
    'LlamaForCausalLM': set_hs_patch_hooks_llama,
}

# --- 3. Main Execution Logic ---
def main():
    print("--- Running Patchscopes Analysis (Official Toolkit Method) ---")

    # --- Load Model ---
    try:
        print(f"Loading model: {MODEL_NAME}...")
        if torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
        print(f"Using device: {device}")
        
        mt = ModelAndTokenizer(model_name=MODEL_NAME, device=device)
        
        model_architecture = mt.model.__class__.__name__
        if model_architecture in MODEL_FAMILY_TO_HOOK_SETTER:
            mt.set_hs_patch_hooks = MODEL_FAMILY_TO_HOOK_SETTER[model_architecture]
            print(f"Assigned correct hook setter for architecture: {model_architecture}")
        else:
            print(f"[ERROR] No hook setter found for model architecture '{model_architecture}'.")
            return
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Could not load model. Reason: {e}")
        return

    # ==============================================================================
    # NEW: TOKENIZATION ANALYSIS BLOCK
    # ==============================================================================
    print("\n--- Tokenization Analysis ---")
    
    # Define the indices we will be using
    source_position_to_check = -2  # The last word of the source
    target_position_to_check = -1  # The last token of the target

    # --- Analyze Source Sentence ---
    source_tokens = mt.tokenizer.tokenize(SOURCE_SENTENCE)
    try:
        selected_source_token = source_tokens[source_position_to_check]
        print(f"Source: '{SOURCE_SENTENCE}'")
        print(f"Tokenized ({len(source_tokens)}): {source_tokens}")
        print(f" -> Will COPY from index {source_position_to_check}: '{selected_source_token}'")
    except IndexError:
        print(f"[ERROR] Source sentence is too short for index {source_position_to_check}.")
        return
        
    # --- Analyze Target Prompt ---
    target_tokens = mt.tokenizer.tokenize(PATCHING_PROMPT)
    try:
        selected_target_token = target_tokens[target_position_to_check]
        print(f"\nTarget: '{PATCHING_PROMPT}'")
        print(f"Tokenized ({len(target_tokens)}): {target_tokens}")
        print(f" -> Will PATCH at index {target_position_to_check}: '{selected_target_token}'")
    except IndexError:
        print(f"[ERROR] Target prompt is too short for index {target_position_to_check}.")
        return

    print("\n" + "="*70)
    # ==============================================================================
    
    results = []
    # --- Run the experiment for each layer ---
    for layer in LAYERS_TO_TEST:
        print(f"\n==============================================")
        print(f"=> Running experiment for Layer {layer}...")
        print(f"==============================================")

        readout_text = inspect(
            mt=mt,
            prompt_source=SOURCE_SENTENCE,
            prompt_target=PATCHING_PROMPT,
            layer_source=layer,
            layer_target=layer,
            position_source=source_position_to_check, # Use the verified index
            position_target=target_position_to_check, # Use the verified index
            generation_mode=True,
            max_gen_len=15
        )
        
        print(f"\n  Readout from Layer {layer}:")
        print(f"  > '{readout_text.strip()}'")
        
        results.append({
            'model_name': MODEL_NAME,
            'layer': layer,
            'readout': readout_text.strip() if readout_text.strip() else "[NO NEW TEXT GENERATED]"
        })

    # --- 4. FINAL SUMMARY AND SAVE ---
    print("\n\n--- FINAL SUMMARY ---")
    results_df = pd.DataFrame(results)
    print(results_df)

    output_filename = f"Official_Toolkit_Readouts_{MODEL_NAME.replace('/', '_')}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to '{output_filename}'")

if __name__ == "__main__":
    main()