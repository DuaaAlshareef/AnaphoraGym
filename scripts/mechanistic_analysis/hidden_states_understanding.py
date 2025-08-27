import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==============================================================================
# 1. SETUP: Load Model and Define Inputs
# ==============================================================================
MODEL_NAME = "gpt2"
print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.eval() # Set model to evaluation mode
print("Model loaded.")

# --- 💡 EDIT THESE LINES TO TEST DIFFERENT WORDS/CONTEXTS 💡 ---
sentence = "The cat quickly jumped onto the high table."
target_word = "cat"
# ----------------------------------------------------------------

# ==============================================================================
# 2. Get the "BEFORE" Vector (Static Embedding)
# ==============================================================================

# First, get the token ID for our target word.
# Note: This simple approach works best for words that are a single token.
# Words like "transformer" might be multiple tokens ("transform", "er").
target_word_ids = tokenizer.encode(target_word, add_special_tokens=False)
if len(target_word_ids) > 1:
    print(f"Warning: '{target_word}' is tokenized into multiple IDs: {target_word_ids}. Using the first ID.")
target_word_id = target_word_ids[0]

# Access the model's word embedding lookup table
embedding_layer = model.transformer.wte
# Retrieve the static vector for our word's ID
with torch.no_grad():
    static_embedding_vector = embedding_layer.weight[target_word_id]

print(f"\n--- Analysis for the word '{target_word}' ---")
print(f"Sentence: '{sentence}'")
print(f"Token ID for '{target_word}': {target_word_id}")

# ==============================================================================
# 3. Get the "AFTER" Vector (Contextual Embedding)
# ==============================================================================

# Find the position of our target word in the full sentence
tokenized_sentence = tokenizer.tokenize(sentence)
try:
    target_word_token_index = tokenized_sentence.index(target_word)
except ValueError:
    # Handle cases where the tokenizer might alter the word (e.g., add a space prefix)
    target_word_token_index = tokenized_sentence.index('Ġ' + target_word)


print(f"Tokenized sentence: {tokenized_sentence}")
print(f"Index of '{target_word}' in the sentence: {target_word_token_index}")

# Encode the full sentence and run it through the model
input_ids = tokenizer.encode(sentence, return_tensors='pt')
with torch.no_grad():
    # We need output_hidden_states=True to access the final layer's output
    outputs = model(input_ids, output_hidden_states=True)

# The final hidden states are the last item in the hidden_states tuple
final_hidden_states = outputs.hidden_states[-1]

# Get the vector at our target word's position
contextual_embedding_vector = final_hidden_states[0, target_word_token_index, :]

# ==============================================================================
# 4. Compare the Vectors
# ==============================================================================

# Calculate Dot Product
dot_product = torch.dot(static_embedding_vector, contextual_embedding_vector).item()

# Calculate Cosine Similarity (this is usually more interpretable)
cosine_sim = F.cosine_similarity(static_embedding_vector.unsqueeze(0), contextual_embedding_vector.unsqueeze(0)).item()

# Calculate the magnitude (L2 norm) of each vector to help with interpretation
static_magnitude = torch.linalg.norm(static_embedding_vector).item()
contextual_magnitude = torch.linalg.norm(contextual_embedding_vector).item()

print("\n--- RESULTS ---")
print(f"Magnitude of 'BEFORE' (Static) Vector:     {static_magnitude:.2f}")
print(f"Magnitude of 'AFTER' (Contextual) Vector:  {contextual_magnitude:.2f}")
print(f"\nDot Product: {dot_product:.2f}")
print(f"Cosine Similarity: {cosine_sim:.4f}")