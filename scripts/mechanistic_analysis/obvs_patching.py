# # # ==============================================================================
# # # SCRIPT FOR PATCHSCOPES - V3 (FINAL, ROBUST METHOD)
# # #
# # # This version trusts the `.full_output()` method as the only way to get
# # # results and uses improved source/target prompts to get a clearer signal.
# # # ==============================================================================

# # from obvs.patchscope import Patchscope, SourceContext, TargetContext

# # # --- 1. Configuration ---
# # MODEL_NAME = "gpt2"

# # SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."

# # # A better target prompt that directly asks a question.
# # PATCHING_PROMPT = "Sam didn’t pass Ricky; Cory didn’t pass Harvey; Kim didn’t pass Taylor; ? "

# # # We will test a middle and a late layer.
# # LAYERS_TO_TEST = [6, 11]

# # # --- 2. Main Execution Logic ---
# # def main():
# #     print("--- Running Patchscopes Analysis (Robust Method) ---")
# #     print(f"\nSource Sentence: '{SOURCE_SENTENCE}'")
# #     print(f"Patching Prompt: '{PATCHING_PROMPT}'")

# #     for layer in LAYERS_TO_TEST:
# #         print(f"\n==============================================")
# #         print(f"=> Running experiment for Layer {layer}...")
# #         print(f"==============================================")

# #         try:
# #             # Step 1: Define the Source Context.
# #             # We are now taking the representation from the last *word* ("Charlie"),
# #             # which is at token position -2 (since '.' is at -1).
# #             source = SourceContext(
# #                 model_name=MODEL_NAME,
# #                 prompt=SOURCE_SENTENCE,
# #                 layer=layer,
# #                 position=-4  # The token for "Charlie"
# #             )

# #             # Step 2: Define the Target Context.
# #             # We patch at the end of the question.
# #             target = TargetContext(
# #                 model_name=MODEL_NAME,
# #                 prompt=PATCHING_PROMPT,
# #                 layer=layer,
# #                 position=-1, # Patch at the end of the prompt
# #                 max_new_tokens=10
# #             )

# #             # Step 3: Create and run the Patchscope experiment.
# #             patchscope = Patchscope(source, target)
# #             patchscope.run()

# #             # =================== THIS IS THE CORRECT METHOD ===================
# #             # We use .full_output() as shown in the original example.
# #             full_generated_text = patchscope.full_output()
# #             readout_text = full_generated_text.removeprefix(PATCHING_PROMPT)
# #             # ==================================================================
            
# #             print(f"\n  Readout from Layer {layer}:")
# #             # If the readout is empty, it means the model generated nothing new.
# #             # This is a valid, meaningful result.
# #             if not readout_text.strip():
# #                 print("  > [Model generated no new text. The patched state did not override the context.]")
# #             else:
# #                 print(f"  > '{readout_text.strip()}'")

# #         except Exception as e:
# #             print(f"\n[ERROR] An error occurred during the experiment for layer {layer}:")
# #             import traceback
# #             traceback.print_exc()

# # if __name__ == "__main__":
# #     main()



# # ==============================================================================
# # SCRIPT FOR PATCHSCOPES - V4 (Corrected and Clarified)
# #
# # This version fixes the source token index and uses a more robust prompt
# # to get a clearer, multi-word signal.
# # ==============================================================================

# from obvs.patchscope import Patchscope, SourceContext, TargetContext

# # --- 1. Configuration ---
# MODEL_NAME = "gpt2"

# SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."

# # A more robust target prompt that frames the task as question answering.
# PATCHING_PROMPT = "Q: What happened with Sam and Ricky? A: Sam didn’t pass Ricky. Q: What happened with Cory and Harvey? A: Cory didn’t pass Harvey. Q: What happened with Alex and Charlie? A:"

# # We will test a middle and a late layer.
# LAYERS_TO_TEST = [6, 11]

# # --- 2. Main Execution Logic ---
# def main():
#     print("--- Running Patchscopes Analysis (Robust Method) ---")
#     print(f"\nSource Sentence: '{SOURCE_SENTENCE}'")
#     print(f"Patching Prompt: '{PATCHING_PROMPT}'")

#     for layer in LAYERS_TO_TEST:
#         print(f"\n==============================================")
#         print(f"=> Running experiment for Layer {layer}...")
#         print(f"==============================================")

#         try:
#             # Step 1: Define the Source Context.
#             # We are taking the representation from the last *word* ("Charlie"),
#             # which is at token position -2 (since '.' is at -1).
#             source = SourceContext(
#                 model_name=MODEL_NAME,
#                 prompt=SOURCE_SENTENCE,
#                 layer=layer,
#                 # CORRECTED: This now correctly targets the token for "Charlie".
#                 position=-2
#             )

#             # Step 2: Define the Target Context.
#             # We patch at the end of the question prompt.
#             target = TargetContext(
#                 model_name=MODEL_NAME,
#                 prompt=PATCHING_PROMPT,
#                 layer=layer,
#                 position=-1, # Patch at the final token of the prompt
#                 max_new_tokens=10
#             )

#             # Step 3: Create and run the Patchscope experiment.
#             patchscope = Patchscope(source, target)
#             patchscope.run()

#             # We use .full_output() to get the generated text.
#             full_generated_text = patchscope.full_output()
#             readout_text = full_generated_text.removeprefix(PATCHING_PROMPT)
            
#             print(f"\n  Readout from Layer {layer}:")
#             if not readout_text.strip():
#                 print("  > [Model generated no new text. The patched state did not override the context.]")
#             else:
#                 print(f"  > '{readout_text.strip()}'")

#         except Exception as e:
#             print(f"\n[ERROR] An error occurred during the experiment for layer {layer}:")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":
#     main()

# ==============================================================================
# SCRIPT FOR PATCHSCOPES - USING gpt2-medium
#
# This version is configured to run the experiment with the gpt2-medium model.
# ==============================================================================

from obvs.patchscope import Patchscope, SourceContext, TargetContext

# --- 1. Configuration ---
# The model name is now changed to "gpt2-medium"
MODEL_NAME = "gpt2-medium"

SOURCE_SENTENCE = "Alex passed Bo, but not Charlie."

# A more robust target prompt that frames the task as question answering.
PATCHING_PROMPT = "Q: What happened with Sam and Ricky? A: Sam didn’t pass Ricky. Q: What happened with Cory and Harvey? A: Cory didn’t pass Harvey. Q: What happened with Alex and Charlie? A:"

# We will test a middle and a late layer. For gpt2-medium (24 layers),
# layers 12 and 22 are good choices.
LAYERS_TO_TEST = [12, 22]

# --- 2. Main Execution Logic ---
def main():
    print("--- Running Patchscopes Analysis (Robust Method) ---")
    print(f"\nSource Sentence: '{SOURCE_SENTENCE}'")
    print(f"Patching Prompt: '{PATCHING_PROMPT}'")

    for layer in LAYERS_TO_TEST:
        print(f"\n==============================================")
        print(f"=> Running experiment for Layer {layer}...")
        print(f"==============================================")

        try:
            # Step 1: Define the Source Context.
            # We are taking the representation from the last *word* ("Charlie"),
            # which is at token position -2 (since '.' is at -1).
            source = SourceContext(
                model_name=MODEL_NAME,
                prompt=SOURCE_SENTENCE,
                layer=layer,
                position=-2  # The token for "Charlie"
            )

            # Step 2: Define the Target Context.
            # We patch at the end of the question prompt.
            target = TargetContext(
                model_name=MODEL_NAME,
                prompt=PATCHING_PROMPT,
                layer=layer,
                position=-1, # Patch at the final token of the prompt
                max_new_tokens=10
            )

            # Step 3: Create and run the Patchscope experiment.
            patchscope = Patchscope(source, target)
            patchscope.run()

            # We use .full_output() to get the generated text.
            full_generated_text = patchscope.full_output()
            readout_text = full_generated_text.removeprefix(PATCHING_PROMPT)
            
            print(f"\n  Readout from Layer {layer}:")
            if not readout_text.strip():
                print("  > [Model generated no new text. The patched state did not override the context.]")
            else:
                print(f"  > '{readout_text.strip()}'")

        except Exception as e:
            print(f"\n[ERROR] An error occurred during the experiment for layer {layer}:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()