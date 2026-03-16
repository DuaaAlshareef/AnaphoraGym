"""
Shared model loading utilities for the AnaphoraGym project.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables (for Hugging Face token if needed)
load_dotenv()

# Any model whose name contains one of these substrings is treated as "large".
# Float16 is used on MPS/CPU; 8-bit is used on CUDA when load_in_8bit=True.
_LARGE_MODEL_KEYWORDS = [
    "0.5b", "1b", "1.5b", "2b", "3b", "4b", "6b", "6.7b",
    "7b", "8b", "9b", "11b", "12b", "13b", "14b", "20b",
    "30b", "34b", "40b", "70b", "72b",
]


def get_device():
    """
    Determine the best available device.
    
    Returns:
        torch.device: The device to use
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def _is_large_model(model_name: str) -> bool:
    """Return True if the model name contains a known large-model size keyword."""
    name_lower = model_name.lower()
    return any(k in name_lower for k in _LARGE_MODEL_KEYWORDS)


def load_model_and_tokenizer(
    model_name: str,
    use_fast_tokenizer: bool = True,
    load_in_8bit: bool = False
):
    """
    Load a model and tokenizer from Hugging Face.

    Large models (≥1 B parameters) are loaded in float16 on MPS/CPU to avoid
    out-of-memory errors, and in 8-bit on CUDA when load_in_8bit=True.
    
    Args:
        model_name: Hugging Face model identifier
        use_fast_tokenizer: Whether to use the fast tokenizer
        load_in_8bit: Whether to load large models in 8-bit mode (CUDA only)
    
    Returns:
        tuple: (model, tokenizer, device)
    """
    hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
    
    device = get_device()
    large = _is_large_model(model_name)
    
    print(f"Loading model: {model_name}...")
    print(f"Using device: {device}")
    
    try:
        if torch.cuda.is_available() and large and load_in_8bit:
            print("Loading large model in 8-bit mode (CUDA).")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                token=hf_token
            )
        elif large:
            # Use float16 on MPS/CPU to halve memory consumption vs float32.
            # A 7B model in float32 needs ~28 GB; float16 needs ~14 GB.
            print("Loading large model in float16 to reduce memory usage.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                token=hf_token
            ).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token
            ).to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer,
            token=hf_token
        )
        
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print(f"Model successfully loaded on {device}.")
        return model, tokenizer, device
        
    except Exception as e:
        raise RuntimeError(f"Could not load model '{model_name}'. Reason: {e}") from e

