# AnaphoraGym: A Benchmark for Evaluating Anaphora Resolution in Language Models

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A computational framework for analyzing the linguistic competence of transformer-based language models through the lens of anaphora resolution.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Behavioral Assessment](#behavioral-assessment)
  - [Mechanistic Analysis](#mechanistic-analysis)
- [Data Availability](#data-availability)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Overview

While modern language models (LLMs) display remarkable fluency, it remains unclear whether they acquire a consistent, human-like understanding of complex linguistic phenomena. **AnaphoraGym** investigates this question through the lens of **anaphora resolution**—a core linguistic challenge that requires deep integration of syntax, context, and world knowledge.

### Research Approach

Our methodology employs a dual-phase approach:

1. **Behavioral Assessment**: Quantify model performance using conditional log-likelihood comparisons between competing linguistic interpretations
2. **Mechanistic Analysis**: Explain behavioral findings using Patchscopes, a causal intervention technique that decodes model internal representations

### Key Questions

- Do language models systematically succeed or fail on specific anaphora resolution patterns?
- How do model scale, architecture, and training affect anaphora resolution capabilities?
- What are the mechanistic explanations for observed behavioral patterns?

## Features

- 🎯 **Comprehensive Benchmark**: Systematic evaluation across multiple anaphora resolution conditions
- 📊 **Behavioral Analysis**: Log-likelihood based assessment framework
- 🔬 **Mechanistic Interpretability**: Patchscopes-based analysis of model internals
- 🧠 **Layer-wise Probing**: Identify which transformer layers encode anaphoric information
- 📈 **Visualization Tools**: Publication-ready charts and comparisons
- 🧩 **Modular Architecture**: Clean separation of experiments, analysis, and visualization
- 🔄 **Reproducible**: Well-documented pipeline with shared utilities

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-capable GPU (recommended for large models)
- 20+ GB RAM (for large models)

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/DuaaAlshareef/AnaphoraGym.git
   cd AnaphoraGym
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install additional dependencies for model evaluation**
   ```bash
   pip install torch transformers accelerate
   ```

5. **Set up Hugging Face authentication** (for gated models)
   ```bash
   # Create a .env file in the project root
   echo "HUGGING_FACE_HUB_TOKEN=your_token_here" > .env
   ```

## Quick Start

### Run a Single Model Assessment

```bash
python scripts/targetted_assessment/experiments/run_experiment.py --model "gpt2"
```

### Run Full Pipeline

```bash
# 1. Configure models in scripts/targetted_assessment/run_all.sh
# 2. Run the complete pipeline
bash scripts/targetted_assessment/run_all.sh
```

## Project Structure

```
AnaphoraGym/
├── dataset/                          # AnaphoraGym benchmark dataset
│   └── AnaphoraGym.csv
├── scripts/
│   ├── targetted_assessment/         # Behavioral assessment module
│   │   ├── experiments/            # Model evaluation scripts
│   │   ├── analysis/                # Statistical analysis
│   │   ├── visualization/           # Plotting and charts
│   │   ├── data/                    # Data processing
│   │   └── utils/                   # Shared utilities
│   └── mechanistic_analysis/       # Patchscopes analysis
├── results/                         # Experimental results
│   ├── targetted_assessment/
│   └── mechanistic_analysis/
├── images/                          # Generated visualizations
├── config.yaml                      # Configuration file
└── requirements.txt                 # Python dependencies
```

For detailed module documentation, see [`scripts/targetted_assessment/README.md`](scripts/targetted_assessment/README.md).

## Usage

### Behavioral Assessment

The behavioral assessment evaluates models by comparing log-likelihoods of competing linguistic interpretations.

#### Single Model Evaluation

```bash
python scripts/targetted_assessment/experiments/run_experiment.py \
    --model "meta-llama/Llama-2-7b-hf"
```

#### Batch Evaluation

1. Edit `scripts/targetted_assessment/run_all.sh`:
   ```bash
   MODELS_TO_TEST=(
     "gpt2"
     "gpt2-medium"
     "meta-llama/Llama-2-7b-hf"
   )
   ```

2. Run the pipeline:
   ```bash
   bash scripts/targetted_assessment/run_all.sh
   ```

#### Analysis and Visualization

After running experiments, generate summaries and visualizations:

```bash
# Aggregate results across all models
python scripts/targetted_assessment/analysis/aggregate_results.py

# Create comparison charts
python scripts/targetted_assessment/visualization/create_bar_chart.py
python scripts/targetted_assessment/visualization/create_faceted_chart.py
python scripts/targetted_assessment/visualization/create_heatmap.py

# Compare model types
python scripts/targetted_assessment/analysis/compare_model_types.py
python scripts/targetted_assessment/visualization/create_comparison_charts.py
```

#### Data Processing

```bash
# Concatenate all model results
python scripts/targetted_assessment/data/concatenate_results.py

# Create enriched dataset with metrics
python scripts/targetted_assessment/data/create_enriched_dataset.py
```

### Mechanistic Analysis

The mechanistic analysis uses Patchscopes to investigate model internal representations.

#### Standard Patchscopes Analysis

```bash
# Configure parameters in scripts/mechanistic_analysis/run_patchscopes_analysis.sh
# Then run:
bash scripts/mechanistic_analysis/run_patchscopes_analysis.sh
```

#### Layer-wise Probing Analysis

**NEW**: Identify which layers of transformer models encode anaphoric information.

**Quick Start:**

```bash
# Run complete layer probing analysis
bash scripts/mechanistic_analysis/run_layer_probing.sh
```

**What it does:**
- Probes all 32 layers of Llama-2-7b-chat-hf
- Measures anaphora resolution performance at each layer
- Identifies which layers encode anaphoric information
- Creates comprehensive visualizations

**Quick Demo:**

```bash
# Test with a simple example
python scripts/mechanistic_analysis/demo_layer_probing.py
```

**Manual Execution:**

```bash
# Step 1: Run layer probing
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "dataset/AnaphoraGym.csv" \
    --max_samples 10

# Step 2: Create visualizations
python scripts/mechanistic_analysis/visualize_layer_probing.py \
    --results_dir "results/mechanistic_analysis/layer_probing" \
    --output_dir "images/layer_probing"
```

**Output:**
- **Statistics**: `results/mechanistic_analysis/layer_probing/layer_statistics.csv`
- **Detailed results**: `results/mechanistic_analysis/layer_probing/detailed_layer_results.csv`
- **Visualizations**: `images/layer_probing/`
  - `layer_performance.png` - 4-panel comprehensive analysis
  - `layer_trajectory.png` - Gradient visualization of layer evolution
  - `layer_comparison.png` - Multi-metric comparison
  - `layer_condition_heatmap.png` - Performance by anaphora condition
  - `analysis_report.txt` - Detailed text summary

**Key Insights:**
- Identifies at which depth anaphora resolution occurs (early/middle/late layers)
- Shows if different anaphora types are processed at different layers
- Provides mechanistic understanding of model linguistic competence

For detailed documentation, see [`scripts/mechanistic_analysis/LAYER_PROBING_README.md`](scripts/mechanistic_analysis/LAYER_PROBING_README.md).

## Data Availability

The `AnaphoraGym.csv` dataset is not publicly available in this repository. For access to the dataset for academic research purposes, please contact the authors directly via email.

## Results

Results are saved in the following locations:

- **Behavioral Assessment**: `results/targetted_assessment/`
  - Individual model results: `AnaphoraGym_Results_<model_name>.csv`
  - Summary table: `model_comparison_summary.csv`
  - Concatenated results: `AnaphoraGym_All_Model_Results_Concatenated.csv`

- **Visualizations**: `images/`
  - Main comparison chart: `model_comparison_chart.png`
  - Faceted charts: `model_comparison_faceted.png`
  - Heatmap: `model_comparison_heatmap.png`
  - Comparison charts: `base_vs_instruct_comparison_chart.png`

- **Mechanistic Analysis**: `results/mechanistic_analysis/`

## Citation

If you use AnaphoraGym in your research, please cite:

```bibtex
@article{anaphoragym2024,
  title={AnaphoraGym: A Benchmark for Evaluating Anaphora Resolution in Language Models},
  author={[Authors]},
  journal={[Journal/Conference]},
  year={2024}
}
```

### Related Work

This project uses Patchscopes for mechanistic analysis:

```bibtex
@inproceedings{ghandeharioun2024patchscopes,
  title={Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models},
  author={Ghandeharioun, Asma and Caciularu, Avi and Pearce, Adam and Dixon, Lucas and Geva, Mor},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The Patchscopes framework by Ghandeharioun et al. (2024)
- Hugging Face for model hosting and transformers library
- The broader NLP and interpretability research community

## Contact

For questions, dataset access requests, or collaboration inquiries, please contact:

- **Email**: dalshareif@aimsammi.org

---

**Note**: This is a research project. Results and interpretations should be considered in the context of the specific experimental setup and model versions used.


