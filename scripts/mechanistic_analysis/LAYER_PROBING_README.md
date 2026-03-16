# Layer-wise Anaphora Probing Analysis

## Overview

This module provides tools to analyze **which layers** of transformer language models (specifically Llama-2-7b-chat-hf) encode anaphoric information. The analysis uses a probing methodology inspired by Patchscopes to measure layer-wise performance on anaphora resolution tasks.

## Key Question

**At which layers does Llama-2-7b-chat-hf encode the information necessary for resolving anaphoric references?**

Understanding this helps us:
- Identify where in the model's processing pipeline anaphora resolution occurs
- Understand the mechanistic basis of linguistic competence in LLMs
- Compare anaphora processing to other linguistic phenomena
- Guide future model improvements and interpretability research

## Model Architecture

Llama-2-7b-chat-hf has:
- **32 transformer layers** (Layer 0 to Layer 31)
- 4096 hidden dimensions
- 32 attention heads per layer

## Methodology

### Probing Approach

For each layer, we:

1. **Extract representations** from that layer for text containing anaphora
2. **Patch** these representations into a target context
3. **Measure** how well the model resolves the anaphora using that layer's information
4. **Compare** log-probabilities of correct vs. incorrect anaphora resolutions

### Metrics

- **Mean Log-Probability Difference**: How much more likely is the correct resolution compared to incorrect? (Higher = better anaphora encoding)
- **Accuracy**: Percentage of examples where correct resolution has higher probability
- **Median Log-Probability Difference**: Robust measure of central tendency

## Files

### Scripts

1. **`layer_wise_probing.py`** - Main probing script
   - Extracts layer representations
   - Measures anaphora resolution performance per layer
   - Saves detailed statistics

2. **`visualize_layer_probing.py`** - Visualization script
   - Creates comprehensive plots showing layer performance
   - Generates heatmaps, trajectories, and comparisons
   - Produces summary report

3. **`run_layer_probing.sh`** - Pipeline runner
   - Runs complete analysis with one command
   - Configurable parameters
   - Displays summary results

## Usage

### Quick Start

Run the complete analysis pipeline:

```bash
cd /Users/duaaalshareif/AMMI/AnaphoraGym
bash scripts/mechanistic_analysis/run_layer_probing.sh
```

This will:
1. Probe all 32 layers of Llama-2-7b-chat-hf
2. Generate performance statistics
3. Create visualizations
4. Produce a summary report

### Custom Configuration

Edit `run_layer_probing.sh` to customize:

```bash
# Model to analyze
MODEL="meta-llama/Llama-2-7b-chat-hf"

# Number of samples per condition
MAX_SAMPLES=10  # Increase for more comprehensive analysis

# Specific conditions (optional)
CONDITIONS="stripping_VPE joins"  # Leave empty for all conditions
```

### Manual Execution

#### Step 1: Run Probing

```bash
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "dataset/AnaphoraGym.csv" \
    --output_dir "results/mechanistic_analysis/layer_probing" \
    --max_samples 10 \
    --conditions stripping_VPE joins
```

**Parameters:**
- `--model`: Model name or path
- `--dataset`: Path to AnaphoraGym dataset
- `--output_dir`: Where to save results
- `--max_samples`: Samples per condition (None = all)
- `--conditions`: Specific conditions to analyze (optional)
- `--device`: cuda or cpu

#### Step 2: Create Visualizations

```bash
python scripts/mechanistic_analysis/visualize_layer_probing.py \
    --results_dir "results/mechanistic_analysis/layer_probing" \
    --output_dir "images/layer_probing"
```

## Output

### Results Directory

`results/mechanistic_analysis/layer_probing/`

- **`layer_statistics.csv`**: Aggregate statistics per layer
  - Columns: layer, mean_logprob_diff, std_logprob_diff, median_logprob_diff, accuracy, n_examples

- **`detailed_layer_results.csv`**: Per-example results for each layer
  - Each row is one example with scores for all 32 layers

- **`summary.json`**: High-level summary
  - Best performing layer
  - Model information
  - Overall statistics

### Visualizations Directory

`images/layer_probing/`

1. **`layer_performance.png`** - Main 4-panel plot showing:
   - Mean log-probability difference across layers
   - Accuracy by layer
   - Median performance
   - Top 10 performing layers

2. **`layer_trajectory.png`** - Gradient visualization showing:
   - How anaphora encoding evolves through layers
   - Highlighting of peak performance layer
   - Color-coded layer regions (early/middle/late)

3. **`layer_comparison.png`** - Multi-metric comparison:
   - Normalized mean and median performance
   - Accuracy overlaid
   - Easy comparison of different metrics

4. **`layer_condition_heatmap.png`** - Condition-specific analysis:
   - Performance breakdown by anaphora type
   - Identifies if different layers handle different anaphora types

5. **`analysis_report.txt`** - Detailed text summary:
   - Key findings
   - Top performing layers
   - Regional analysis (early/middle/late layers)
   - Interpretation

## Interpreting Results

### What to Look For

1. **Peak Layer**: Which layer shows highest anaphora resolution performance?
   - Early layers (0-10): Suggests surface-level pattern matching
   - Middle layers (11-21): Suggests intermediate semantic processing
   - Late layers (22-31): Suggests high-level reasoning and integration

2. **Performance Trajectory**: How does performance change across layers?
   - Gradual increase: Progressive refinement
   - Sharp peak: Specific layer specialization
   - Multiple peaks: Distributed processing

3. **Condition Differences**: Do different anaphora types peak at different layers?
   - Uniform: General anaphora mechanism
   - Varied: Type-specific processing

### Example Interpretation

```
Best Layer: Layer 18
Mean Score: 0.42
Region: Middle layers

Interpretation: The model primarily encodes anaphoric information 
in the middle layers, suggesting that anaphora resolution involves 
intermediate semantic processing rather than surface patterns or 
late-stage reasoning.
```

## Expected Runtime

- **With GPU (CUDA)**: ~10-30 minutes for 10 samples/condition
- **With CPU**: ~1-2 hours for 10 samples/condition

For comprehensive analysis (all samples):
- **With GPU**: ~1-3 hours
- **With CPU**: ~5-10 hours

## Requirements

### Python Packages

```bash
torch
transformers
pandas
numpy
matplotlib
seaborn
tqdm
```

### Hardware

- **Minimum**: 16GB RAM, 8GB GPU VRAM (for float16)
- **Recommended**: 32GB RAM, 16GB GPU VRAM
- **CPU only**: Possible but slow

### Model Access

For gated models like Llama-2:
1. Request access at [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. Login: `huggingface-cli login`
3. Or set token: `export HUGGING_FACE_HUB_TOKEN=your_token`

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA OOM errors:

```bash
# Reduce batch size in probing script
# Or use CPU
python scripts/mechanistic_analysis/layer_wise_probing.py --device cpu
```

### Model Loading Issues

```bash
# Ensure you have access to the model
huggingface-cli login

# Or use a different model
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-hf"  # Base model instead
```

### Slow Processing

```bash
# Reduce samples per condition
# Edit run_layer_probing.sh:
MAX_SAMPLES=3  # Instead of 10

# Or analyze specific conditions only
CONDITIONS="stripping_VPE"
```

## Advanced Usage

### Analyzing Different Models

Compare layer-wise anaphora encoding across models:

```bash
# Llama-2 7B base
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --output_dir "results/mechanistic_analysis/layer_probing_base"

# Llama-2 7B chat (instruction-tuned)
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --output_dir "results/mechanistic_analysis/layer_probing_chat"

# Compare results
python scripts/mechanistic_analysis/compare_model_layers.py \
    --base_results "results/mechanistic_analysis/layer_probing_base" \
    --chat_results "results/mechanistic_analysis/layer_probing_chat"
```

### Focusing on Specific Conditions

Analyze only certain anaphora types:

```bash
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --conditions stripping_VPE joins \
    --max_samples 20
```

### Exporting Results

```bash
# Convert to different formats
import pandas as pd

# Load results
df = pd.read_csv('results/mechanistic_analysis/layer_probing/layer_statistics.csv')

# Export to LaTeX table
df.to_latex('layer_results.tex')

# Export to Excel
df.to_excel('layer_results.xlsx')
```

## Research Applications

### Comparative Analysis

Use this tool to:
- Compare base vs. instruction-tuned models
- Compare different model sizes (7B, 13B, 70B)
- Compare different architectures (Llama, GPT, etc.)

### Hypothesis Testing

Test hypotheses about:
- Where linguistic processing occurs in transformers
- Effects of instruction tuning on layer-wise representations
- Relationship between model size and layer specialization

### Integration with Other Analyses

Combine with:
- Attention pattern analysis
- Neuron activation studies
- Causal intervention experiments

## Citation

If you use this layer probing analysis in your research, please cite:

```bibtex
@article{anaphoragym2024,
  title={AnaphoraGym: A Benchmark for Evaluating Anaphora Resolution in Language Models},
  author={[Authors]},
  year={2024}
}
```

And the Patchscopes paper:

```bibtex
@inproceedings{ghandeharioun2024patchscopes,
  title={Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models},
  author={Ghandeharioun, Asma and Caciularu, Avi and Pearce, Adam and Dixon, Lucas and Geva, Mor},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year={2024}
}
```

## Contact

For questions or issues:
- Email: dalshareif@aimsammi.org
- GitHub Issues: [AnaphoraGym Repository](https://github.com/DuaaAlshareef/AnaphoraGym)

## License

This code is part of the AnaphoraGym project and follows the same MIT License.
