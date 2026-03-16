#!/usr/bin/env python3
"""
Layer-wise probing for AnaphoraGym using a small HF model (default: gpt2).

What it does:
- Reads dataset/AnaphoraGym.csv
- Builds prompt from input_1..input_4 + context_prompt
- Computes behavioral signal: ΔlogP = logP(cont1|prompt) - logP(cont2|prompt)
- Extracts hidden states at a chosen token position (default: last prompt token)
- Trains a ridge probe per layer to predict ΔlogP
- Saves plots + CSV to results/layer_probing/

Run:
  python scripts/mechanistic_analysis/layer_probing.py --csv dataset/AnaphoraGym.csv --model gpt2
"""

import os
import argparse
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# -----------------------
# Helpers
# -----------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Example:
    prompt: str
    cont1: str
    cont2: str

def build_prompt_from_row(row: pd.Series) -> str:
    """
    AnaphoraGym-style prompt builder.
    Concatenate input_1..input_4 (if present) + context_prompt with newlines.
    """
    parts = []
    for col in ["input_1", "input_2", "input_3", "input_4"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            parts.append(str(row[col]).strip())
    if "context_prompt" in row and pd.notna(row["context_prompt"]) and str(row["context_prompt"]).strip():
        parts.append(str(row["context_prompt"]).strip())

    prompt = "\n".join(parts).strip()
    if not prompt.endswith(" "):
        prompt += " "
    return prompt

def load_examples(csv_path: str, max_rows: Optional[int] = None) -> List[Example]:
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.head(max_rows)

    if "continuation_1" not in df.columns or "continuation_2" not in df.columns:
        raise ValueError("CSV must contain continuation_1 and continuation_2 columns.")

    exs = []
    for _, row in df.iterrows():
        prompt = build_prompt_from_row(row)
        c1 = str(row["continuation_1"]).strip()
        c2 = str(row["continuation_2"]).strip()
        exs.append(Example(prompt=prompt, cont1=c1, cont2=c2))
    return exs

def logprob_of_continuation(model, tokenizer, prompt: str, continuation: str, device: torch.device) -> float:
    """
    Sum log p(tokens in continuation | prompt).
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full_ids = tokenizer(prompt + continuation, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]
    if full_len <= prompt_len:
        return float("-inf")

    with torch.no_grad():
        out = model(full_ids)
        logits = out.logits  # [1, T, V]

    log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    # continuation tokens are positions prompt_len ... full_len-1
    for pos in range(prompt_len, full_len):
        tok = full_ids[0, pos].item()
        pred_pos = pos - 1
        total += log_probs[0, pred_pos, tok].item()
    return total

def extract_layerwise_state(model, tokenizer, text: str, device: torch.device, which_token: str = "last") -> np.ndarray:
    """
    Returns [n_layers+1, hidden_dim] hidden state for a selected token position.
    (Layer 0 = embeddings)
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)

    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple length = n_layers+1

    if which_token == "last":
        tidx = input_ids.shape[1] - 1
    elif which_token == "first":
        tidx = 0
    else:
        raise ValueError("which_token must be 'last' or 'first'")

    layers = [h[0, tidx, :].float().cpu().numpy() for h in hs]
    return np.stack(layers, axis=0)


# -----------------------
# Main
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="dataset/AnaphoraGym.csv")
    ap.add_argument("--model", type=str, default="gpt2")
    ap.add_argument("--max_rows", type=int, default=500)
    ap.add_argument("--which_token", type=str, default="last", choices=["last", "first"])
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=10.0, help="Ridge regularization strength")
    ap.add_argument("--outdir", type=str, default="results/layer_probing")
    args = ap.parse_args()

    set_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[info] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # GPT2 has no pad token by default; set to eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    examples = load_examples(args.csv, max_rows=args.max_rows)
    print(f"[info] loaded {len(examples)} rows from {args.csv}")

    # 1) behavioral signal ΔlogP
    prompts = []
    deltas = []
    for ex in tqdm(examples, desc="ΔlogP"):
        lp1 = logprob_of_continuation(model, tokenizer, ex.prompt, ex.cont1, device)
        lp2 = logprob_of_continuation(model, tokenizer, ex.prompt, ex.cont2, device)
        deltas.append(lp1 - lp2)
        prompts.append(ex.prompt)

    deltas = np.array(deltas, dtype=np.float32)

    # 2) hidden states per example: [N, L+1, D]
    H_list = []
    for p in tqdm(prompts, desc="hidden states"):
        H_list.append(extract_layerwise_state(model, tokenizer, p, device, which_token=args.which_token))
    H = np.stack(H_list, axis=0)

    N, n_layers, d = H.shape
    print(f"[info] H shape: {H.shape} (N, layers, dim)")

    # 3) split once for all layers
    idx = np.arange(N)
    idx_tr, idx_te = train_test_split(idx, test_size=args.test_size, random_state=args.seed)

    y_tr = deltas[idx_tr]
    y_te = deltas[idx_te]

    # 4) probe per layer
    r2s = np.zeros(n_layers, dtype=np.float32)
    preds_te = []

    for L in tqdm(range(n_layers), desc="train probe per layer"):
        X_tr = H[idx_tr, L, :]
        X_te = H[idx_te, L, :]

        probe = Ridge(alpha=args.alpha, random_state=args.seed)
        probe.fit(X_tr, y_tr)
        y_hat = probe.predict(X_te)
        r2s[L] = r2_score(y_te, y_hat)
        preds_te.append(y_hat)

    best_L = int(np.argmax(r2s))
    print(f"[result] best layer = {best_L}, R^2 = {r2s[best_L]:.4f}")

    # 5) save CSV
    df_out = pd.DataFrame({"layer": np.arange(n_layers), "r2": r2s})
    csv_out = os.path.join(args.outdir, "r2_by_layer.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"[save] {csv_out}")

    # 6) plot R^2 by layer
    plt.figure()
    plt.plot(np.arange(n_layers), r2s)
    plt.axvline(best_L, linestyle="--")
    plt.xlabel("Layer (0=embeddings)")
    plt.ylabel("R² (predict ΔlogP)")
    plt.title(f"Layer probing ({args.model}, token={args.which_token})")
    plt.tight_layout()
    fig1 = os.path.join(args.outdir, "r2_by_layer.png")
    plt.savefig(fig1, dpi=200)
    print(f"[save] {fig1}")

    # 7) scatter best layer
    y_hat_best = preds_te[best_L]
    plt.figure()
    plt.scatter(y_te, y_hat_best, s=10)
    lo = float(min(y_te.min(), y_hat_best.min()))
    hi = float(max(y_te.max(), y_hat_best.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("True ΔlogP")
    plt.ylabel("Predicted ΔlogP")
    plt.title(f"Best layer {best_L} scatter (R²={r2s[best_L]:.3f})")
    plt.tight_layout()
    fig2 = os.path.join(args.outdir, "scatter_best_layer.png")
    plt.savefig(fig2, dpi=200)
    print(f"[save] {fig2}")

    print("[done]")


if __name__ == "__main__":
    main()
