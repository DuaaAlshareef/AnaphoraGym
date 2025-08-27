# ==============================================================================
# SCRIPT FOR PATCHSCOPES - OFFICIAL TOOLKIT METHOD (V3 - FINAL)
#
# This version passes raw strings to `inspect` and relies on the modified
# utility files to handle the attention_mask warning correctly.
# ==============================================================================

import torch
from scripts.patchscopes.general_utils import ModelAndTokenizer
from scripts.patchscopes.patchscopes_utils import inspect, set_hs_patch_hooks_gptj

# --- Configuration ---
MODEL_NAME = "gpt2"
SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."
PATCHING_PROMPT = "Sam didn’t pass Ricky; Cory didn’t pass Harvey; Kim didn’t pass Taylor; "
LAYERS_TO_TEST = [6, 11]

# --- Main Execution Logic ---
def main():
    print("--- Running Patchscopes Analysis (Official Toolkit Method) ---")

    try:
        print(f"Loading model: {MODEL_NAME}...")
        if torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"
        print(f"Using device: {device}")
        
        mt = ModelAndTokenizer(model_name=MODEL_NAME, device=device)
        
        # Explicitly set the pad token to the eos token for gpt2
        if mt.tokenizer.pad_token is None:
            mt.tokenizer.pad_token = mt.tokenizer.eos_token

        mt.set_hs_patch_hooks = set_hs_patch_hooks_gptj
        
        print("Model loaded successfully.")
    except Exception as e:
        print(f"\n[ERROR] Could not load model. Reason: {e}")
        return

    print(f"\nSource Sentence: '{SOURCE_SENTENCE}'")
    print(f"Patching Prompt: '{PATCHING_PROMPT}'")

    for layer in LAYERS_TO_TEST:
        print(f"\n==============================================")
        print(f"=> Running experiment for Layer {layer}...")
        print(f"==============================================")

        # We now pass the raw strings directly to inspect, as it was designed.
        readout_text = inspect(
            mt=mt,
            prompt_source=SOURCE_SENTENCE, # <-- Pass the raw STRING here
            prompt_target=PATCHING_PROMPT, # <-- Pass the raw STRING here
            
            layer_source=layer,
            layer_target=layer,
            position_source=-1,
            position_target=-1,
            generation_mode=True,
            max_gen_len=10
        )
        
        print(f"\n  Readout from Layer {layer}:")
        print(f"  > '{readout_text.strip()}'")

if __name__ == "__main__":
    main()