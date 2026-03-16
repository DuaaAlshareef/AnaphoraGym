# Layer-wise Anaphora Probing: Complete Implementation

## 🎯 Summary

I've created a comprehensive layer-wise probing system for analyzing **which layers of Llama-2-7b-chat-hf encode anaphoric information**. This system allows you to understand the mechanistic basis of anaphora resolution in transformer language models.

## 📦 What Was Created

### 1. Core Scripts

#### `layer_wise_probing.py`
The main probing engine that:
- Loads Llama-2-7b-chat-hf (32 layers)
- Extracts hidden representations from each layer
- Uses patchscopes methodology to measure anaphora resolution at each layer
- Calculates log-probability differences between correct/incorrect resolutions
- Saves comprehensive statistics

**Key Features:**
- Batch processing of entire AnaphoraGym dataset
- Layer-by-layer performance measurement
- Configurable sampling and condition filtering
- Detailed per-example results

#### `visualize_layer_probing.py`
Comprehensive visualization suite that creates:
- **4-panel performance plot**: Mean/std, accuracy, median, top layers
- **Layer trajectory**: Gradient visualization showing evolution through layers
- **Multi-metric comparison**: Normalized comparison of different metrics
- **Condition heatmap**: Performance breakdown by anaphora type
- **Text report**: Detailed summary with interpretations

#### `run_layer_probing.sh`
One-command pipeline that:
- Runs complete probing analysis
- Generates all visualizations
- Displays summary results
- Configurable via environment variables

#### `demo_layer_probing.py`
Interactive demo that:
- Tests the system with simple examples
- Shows layer extraction in action
- Demonstrates probing methodology
- Provides quick verification

### 2. Documentation

#### `LAYER_PROBING_README.md`
Comprehensive 400+ line documentation covering:
- Methodology and approach
- Installation and requirements
- Usage examples (basic and advanced)
- Output interpretation
- Troubleshooting
- Research applications

#### Updated Main `README.md`
Added complete section on layer probing to main project documentation with:
- Quick start commands
- Output descriptions
- Key insights section

## 🚀 Quick Start

### Option 1: Run Everything (Recommended for First Time)

```bash
cd /Users/duaaalshareif/AMMI/AnaphoraGym
bash scripts/mechanistic_analysis/run_layer_probing.sh
```

This will:
1. Probe all 32 layers of Llama-2-7b-chat-hf
2. Analyze 10 samples per anaphora condition
3. Create all visualizations
4. Display summary results

**Expected Runtime:** 15-30 minutes with GPU

### Option 2: Quick Demo First

```bash
python scripts/mechanistic_analysis/demo_layer_probing.py
```

Tests the system with a simple example (5-10 minutes).

### Option 3: Manual Step-by-Step

```bash
# Step 1: Probe layers
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "dataset/AnaphoraGym.csv" \
    --output_dir "results/mechanistic_analysis/layer_probing" \
    --max_samples 10

# Step 2: Create visualizations
python scripts/mechanistic_analysis/visualize_layer_probing.py \
    --results_dir "results/mechanistic_analysis/layer_probing" \
    --output_dir "images/layer_probing"
```

## 📊 Output Files

### Results Directory: `results/mechanistic_analysis/layer_probing/`

1. **`layer_statistics.csv`** - Main results
   ```csv
   layer,mean_logprob_diff,std_logprob_diff,median_logprob_diff,accuracy,n_examples
   0,0.1234,0.0567,0.1123,0.65,50
   1,0.2345,0.0678,0.2234,0.72,50
   ...
   ```

2. **`detailed_layer_results.csv`** - Per-example scores
   ```csv
   condition,item,source_text,0,1,2,...,31
   stripping_VPE,1,"Alex passed...",0.12,0.23,...,0.45
   ```

3. **`summary.json`** - High-level summary
   ```json
   {
     "model": "meta-llama/Llama-2-7b-chat-hf",
     "num_layers": 32,
     "best_layer": 18,
     "best_layer_score": 0.4234
   }
   ```

### Visualizations Directory: `images/layer_probing/`

1. **`layer_performance.png`** - Main 4-panel plot
   - Mean log-prob difference with std dev
   - Accuracy by layer
   - Median performance
   - Top 10 layers bar chart

2. **`layer_trajectory.png`** - Gradient visualization
   - Color-coded layer evolution
   - Peak layer highlighted
   - Region annotations (early/middle/late)

3. **`layer_comparison.png`** - Multi-metric overlay
   - Normalized mean and median
   - Accuracy comparison
   - Baseline reference

4. **`layer_condition_heatmap.png`** - Condition breakdown
   - Performance by anaphora type
   - Layer × Condition matrix

5. **`analysis_report.txt`** - Text summary
   - Key findings
   - Top layers
   - Regional analysis
   - Interpretation

## 🧪 What You'll Learn

### Research Questions Answered

1. **At which layers does Llama-2-7b-chat-hf encode anaphoric information?**
   - Early layers (0-10): Surface patterns?
   - Middle layers (11-21): Semantic processing?
   - Late layers (22-31): High-level reasoning?

2. **Is anaphora processing localized or distributed?**
   - Single peak: Specialized layer
   - Multiple peaks: Distributed processing
   - Gradual increase: Progressive refinement

3. **Do different anaphora types use different layers?**
   - Check the heatmap for condition-specific patterns

### Example Interpretation

```
Best Layer: Layer 18
Score: 0.42
Region: Middle layers

Interpretation:
The model encodes anaphoric information primarily in the middle 
layers (around layer 18 out of 32), suggesting that anaphora 
resolution involves intermediate semantic processing rather than 
surface patterns (early layers) or late-stage reasoning (late layers).
```

## 🔧 Configuration Options

### In `run_layer_probing.sh`:

```bash
# Model to analyze
MODEL="meta-llama/Llama-2-7b-chat-hf"

# Samples per condition (higher = more comprehensive, slower)
MAX_SAMPLES=10  # Default: 10, increase to 50+ for full analysis

# Specific conditions (optional)
# CONDITIONS="stripping_VPE joins"  # Uncomment to focus on specific types

# Output directories
RESULTS_DIR="results/mechanistic_analysis/layer_probing"
IMAGES_DIR="images/layer_probing"
```

### Via Command Line:

```bash
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "dataset/AnaphoraGym.csv" \
    --max_samples 20 \
    --conditions stripping_VPE joins \
    --device cuda
```

## 💡 Advanced Usage

### Compare Different Models

```bash
# Base model
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-hf" \
    --output_dir "results/layer_probing_base"

# Chat model (instruction-tuned)
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --output_dir "results/layer_probing_chat"

# Compare results manually or create comparison visualizations
```

### Focus on Specific Conditions

```bash
# Only analyze VPE and joins
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --conditions stripping_VPE joins \
    --max_samples 20
```

### Comprehensive Analysis

```bash
# Process all examples (may take hours)
python scripts/mechanistic_analysis/layer_wise_probing.py \
    --max_samples None  # No limit
```

## 📈 Expected Results

Based on typical transformer behavior, you might see:

1. **Early layers (0-10)**: Lower performance
   - Still processing surface features
   - Anaphora resolution requires deeper understanding

2. **Middle layers (11-21)**: Peak performance
   - Semantic processing active
   - Contextual understanding emerging
   - **Most likely location of anaphora encoding**

3. **Late layers (22-31)**: Variable performance
   - Task-specific adaptation
   - May show specialization from instruction tuning

## ⚠️ Important Notes

### Requirements

- **GPU**: Strongly recommended (CUDA)
- **RAM**: 16GB minimum, 32GB recommended
- **VRAM**: 8GB minimum for float16
- **Time**: 15-30 minutes (GPU), 1-2 hours (CPU)

### Model Access

Llama-2 models are gated. You need:

```bash
# Login to Hugging Face
huggingface-cli login

# Or set token
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

Request access at: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

### Troubleshooting

**Out of Memory:**
```bash
# Use CPU (slower but works)
python scripts/mechanistic_analysis/layer_wise_probing.py --device cpu
```

**Slow Processing:**
```bash
# Reduce samples
# Edit run_layer_probing.sh:
MAX_SAMPLES=3
```

**Model Loading Issues:**
```bash
# Ensure proper authentication
huggingface-cli login
```

## 📝 Files Created

```
AnaphoraGym/
├── scripts/mechanistic_analysis/
│   ├── layer_wise_probing.py           # Main probing script (340 lines)
│   ├── visualize_layer_probing.py      # Visualization suite (480 lines)
│   ├── run_layer_probing.sh            # Pipeline runner (80 lines)
│   ├── demo_layer_probing.py           # Interactive demo (120 lines)
│   └── LAYER_PROBING_README.md         # Full documentation (450 lines)
├── README.md                            # Updated with layer probing section
└── LAYER_PROBING_SUMMARY.md            # This file
```

## 🎓 Research Applications

This tool enables:

1. **Mechanistic Interpretability**
   - Understand where linguistic processing occurs
   - Compare to other phenomena (syntax, semantics, etc.)

2. **Model Comparison**
   - Base vs. instruction-tuned models
   - Different sizes (7B, 13B, 70B)
   - Different architectures (Llama, GPT, etc.)

3. **Hypothesis Testing**
   - Effects of training on layer specialization
   - Relationship between model depth and linguistic competence

4. **Architecture Design**
   - Inform future model designs
   - Optimize layer depth for specific tasks

## 📚 References

The methodology is based on:

**Patchscopes:**
- Ghandeharioun et al. (2024). "Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models." ICML 2024.

**Probing Techniques:**
- Belinkov & Glass (2019). "Analysis Methods in Neural Language Processing: A Survey."
- Rogers et al. (2020). "A Primer on Neural Network Architectures for Natural Language Processing."

## 🤝 Next Steps

1. **Run the analysis:**
   ```bash
   bash scripts/mechanistic_analysis/run_layer_probing.sh
   ```

2. **Examine visualizations:**
   ```bash
   open images/layer_probing/layer_performance.png
   ```

3. **Read the report:**
   ```bash
   cat images/layer_probing/analysis_report.txt
   ```

4. **Explore the data:**
   ```python
   import pandas as pd
   df = pd.read_csv('results/mechanistic_analysis/layer_probing/layer_statistics.csv')
   print(df.nlargest(5, 'mean_logprob_diff'))
   ```

## 📧 Support

For questions or issues:
- Check `scripts/mechanistic_analysis/LAYER_PROBING_README.md`
- Run the demo: `python scripts/mechanistic_analysis/demo_layer_probing.py`
- Contact: dalshareif@aimsammi.org

---

**Created:** 2026-01-16  
**Model Analyzed:** meta-llama/Llama-2-7b-chat-hf (32 layers)  
**Methodology:** Patchscopes-based layer probing  
**Purpose:** Mechanistic understanding of anaphora resolution in LLMs
