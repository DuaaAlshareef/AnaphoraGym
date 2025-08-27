# AnaphoraGym: a Benchmark for Evaluating Anaphora Resolution in Language Models

This repository contains the code and resources for the AnaphoraGym project, a computational framework for analyzing the linguistic competence of transformer-based language models. The project focuses on the complex task of anaphora resolution, using a dual approach that combines behavioral assessment and mechanistic interpretability.

---

##  Project Overview

While modern language models (LLMs) display remarkable fluency, it remains unclear if they acquire a consistent, human-like understanding of complex linguistic phenomena. This project investigates this question through the lens of **anaphora resolution**, a core linguistic challenge that requires a deep integration of syntax, context, and world knowledge.

Our methodology is composed of two main phases:

1.  **Phase 1: Behavioral Assessment**
    We use a rigorous **targeted assessment** framework to quantify model performance on the `AnaphoraGym` benchmark. By calculating the conditional log-likelihood (`log P(continuation | input)`), we measure a model's preference between competing linguistic interpretations, allowing us to identify systematic successes and failures across different model families and scales.

2.  **Phase 2: Mechanistic Analysis**
    To explain the behavioral findings, we employ **Patchscopes** [[1]](#1), a causal intervention technique. By surgically extracting and patching hidden state representations from one context to another, we can decode the model's internal "thoughts" into natural language. This allows us to trace the model's reasoning process layer-by-layer and find direct, mechanistic explanations for its behavior.

---
##  Data Availability

Please note that the `AnaphoraGym.csv` dataset is not publicly available in this repository. For access to the dataset for academic research purposes, please contact the us directly via email.

---

##  Getting Started

### Prerequisites

This project uses Python 3.11+. The required libraries are listed in `requirements.txt`.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/DuaaAlshareef/AnaphoraGym.git
    cd AnaphoraGym
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

### Running the Behavioral Assessment

The entire behavioral assessment pipeline can be run with a single command from the project's root directory.

1.  **Configure the experiment:** Open the `run_all.sh` script and edit the `MODELS_TO_TEST` array to include the Hugging Face model identifiers you wish to evaluate.

2.  **Make the script executable (only once):**
    ```bash
    chmod +x run_all.sh
    ```

3.  **Run the full pipeline:**
    ```bash
    ./run_all.sh
    ```

This will execute the `test_anaphoragym.py` script for each model, saving the results in the `results/targetted_assessment/` directory. It will then automatically run `analyze_results.py` to generate a summary table and a comparative performance chart in the `images/` directory.

### Running the Mechanistic (Patchscopes) Analysis

The Patchscopes scripts are designed for more targeted, single-model investigations.

1.  **Configure the script:** Open `scripts/2_mechanistic_analysis/run_patchscopes.py`.
2.  **Set the `MODEL_NAME`** and other experimental parameters (e.g., `SOURCE_SENTENCE`, `PATCHING_PROMPT`).
3.  **Run the script:**
    ```bash
    python3 scripts/2_mechanistic_analysis/run_patchscopes.py
    ```
This will print the readout results to the console and save a corresponding CSV file in the `results/mechanistic_analysis/` folder.



---

## <a id="references"></a>📜 Citations
<a id="1">[1]</a> 
Ghandeharioun, A., Caciularu, A., Pearce, A., Dixon, L., & Geva, M. (2024). 
[*Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models*](https://arxiv.org/abs/2401.06102). 
In *Proceedings of the 41st International Conference on Machine Learning (ICML)*.

---