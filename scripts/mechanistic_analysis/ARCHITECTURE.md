# Layer Probing System Architecture

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LAYER PROBING SYSTEM                          │
│                  for Anaphora Resolution Analysis                │
└─────────────────────────────────────────────────────────────────┘

                              INPUT
                                │
                    ┌───────────▼───────────┐
                    │  AnaphoraGym.csv     │
                    │  (Dataset)           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────────────┐
                    │   layer_wise_probing.py      │
                    │                               │
                    │  ┌─────────────────────────┐ │
                    │  │ 1. Load Model (32 L)   │ │
                    │  │    Llama-2-7b-chat-hf  │ │
                    │  └──────────┬──────────────┘ │
                    │             │                 │
                    │  ┌──────────▼──────────────┐ │
                    │  │ 2. For each layer:      │ │
                    │  │    - Extract hidden rep │ │
                    │  │    - Patch to target    │ │
                    │  │    - Measure resolution │ │
                    │  └──────────┬──────────────┘ │
                    │             │                 │
                    │  ┌──────────▼──────────────┐ │
                    │  │ 3. Calculate metrics:   │ │
                    │  │    - Log-prob diff      │ │
                    │  │    - Accuracy           │ │
                    │  │    - Statistics         │ │
                    │  └──────────┬──────────────┘ │
                    └─────────────┼─────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │        RESULTS            │
                    ├───────────────────────────┤
                    │ • layer_statistics.csv    │
                    │ • detailed_results.csv    │
                    │ • summary.json            │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────▼──────────────────┐
                    │  visualize_layer_probing.py   │
                    │                                │
                    │  ┌──────────────────────────┐ │
                    │  │ 4. Create visualizations:│ │
                    │  │   - Performance plot     │ │
                    │  │   - Trajectory           │ │
                    │  │   - Heatmap              │ │
                    │  │   - Comparison           │ │
                    │  └──────────┬───────────────┘ │
                    └─────────────┼─────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │    VISUALIZATIONS         │
                    ├───────────────────────────┤
                    │ • layer_performance.png   │
                    │ • layer_trajectory.png    │
                    │ • layer_comparison.png    │
                    │ • layer_heatmap.png       │
                    │ • analysis_report.txt     │
                    └───────────────────────────┘

              OUTPUT: Understanding of which layers
                     encode anaphoric information
```

## 📦 Component Details

### 1. Data Input Layer

**AnaphoraGym.csv**
- Source: Benchmark dataset
- Structure: Conditions × Items × Inputs × Continuations
- Examples: stripping_VPE, joins, etc.

### 2. Probing Engine

**layer_wise_probing.py** (340 lines)

```python
class AnaphoraLayerProber:
    def __init__(model_name):
        # Load Llama-2-7b-chat-hf (32 layers)
        
    def extract_layer_representations(text, position):
        # Extract hidden states from all layers
        # Returns: {layer: tensor}
        
    def probe_anaphora_resolution(source, target, correct, incorrect):
        # For each layer:
        #   1. Extract representation
        #   2. Patch to target
        #   3. Measure log-prob of continuations
        # Returns: DataFrame with layer-wise scores
        
    def probe_dataset(dataset_path, output_dir):
        # Process entire dataset
        # Returns: Aggregate statistics + detailed results
```

**Key Features:**
- Patchscopes-based methodology
- Layer-by-layer analysis
- Batch processing
- Configurable sampling

### 3. Visualization Engine

**visualize_layer_probing.py** (480 lines)

```python
Functions:
├── plot_layer_performance()      # 4-panel comprehensive plot
├── plot_layer_trajectory()       # Gradient evolution visualization
├── plot_layer_comparison()       # Multi-metric comparison
├── plot_layer_heatmap()          # Condition × Layer matrix
└── create_summary_report()       # Text interpretation
```

**Output Types:**
- PNG images (high-resolution, publication-ready)
- Text reports (human-readable insights)

### 4. Pipeline Orchestration

**run_layer_probing.sh** (80 lines)

```bash
1. Setup
   ├── Configure parameters
   ├── Create directories
   └── Validate inputs

2. Execute
   ├── Run layer_wise_probing.py
   └── Run visualize_layer_probing.py

3. Report
   ├── Display summary
   ├── Show best layer
   └── List output files
```

### 5. Testing & Demo

**demo_layer_probing.py** (120 lines)

```python
Demonstrates:
├── Model loading
├── Simple anaphora example
├── Layer probing process
├── Representation extraction
└── Result interpretation
```

## 🔄 Data Flow

### Stage 1: Input Processing

```
AnaphoraGym.csv
    │
    ├─► Parse rows (condition, item, inputs, continuations)
    ├─► Filter conditions (optional)
    ├─► Sample examples (configurable)
    └─► Format for probing
```

### Stage 2: Layer Probing

```
For each example:
    For layer in range(32):
        1. Extract hidden representation at layer L
           ├─► Run model forward pass
           ├─► Get hidden_states[L]
           └─► Extract at anaphora position
        
        2. Setup patching
           ├─► Create target context
           ├─► Configure patch hooks
           └─► Patch representation from layer L
        
        3. Measure resolution
           ├─► Compute log-prob(correct continuation)
           ├─► Compute log-prob(incorrect continuation)
           ├─► Calculate difference
           └─► Determine accuracy
        
        4. Store results
           └─► Layer × Metric matrix
```

### Stage 3: Aggregation

```
Results from all examples:
    ├─► Calculate mean per layer
    ├─► Calculate std per layer
    ├─► Calculate median per layer
    ├─► Calculate accuracy per layer
    └─► Identify best layer
```

### Stage 4: Visualization

```
Statistics:
    ├─► 4-panel performance plot
    │   ├─► Mean + std by layer
    │   ├─► Accuracy by layer
    │   ├─► Median by layer
    │   └─► Top 10 layers
    │
    ├─► Trajectory plot
    │   ├─► Color-coded evolution
    │   ├─► Peak highlighting
    │   └─► Region annotation
    │
    ├─► Multi-metric comparison
    │   ├─► Normalized metrics
    │   └─► Overlay comparison
    │
    └─► Condition heatmap
        ├─► Layer × Condition matrix
        └─► Type-specific patterns
```

## 🧬 Technical Stack

### Core Technologies

```
┌─────────────────────────────────────┐
│         Python 3.11+                │
├─────────────────────────────────────┤
│ PyTorch          │ Model operations │
│ Transformers     │ Model loading    │
│ Pandas           │ Data processing  │
│ NumPy            │ Numerical ops    │
│ Matplotlib       │ Plotting         │
│ Seaborn          │ Visualization    │
│ tqdm             │ Progress bars    │
└─────────────────────────────────────┘
```

### Model Architecture

```
Llama-2-7b-chat-hf
├── 32 Transformer Layers
│   ├── Multi-head Self-Attention (32 heads)
│   ├── Feed-Forward Network
│   └── Layer Normalization
├── 4096 Hidden Dimensions
├── 32,000 Vocabulary Size
└── 4096 Context Length
```

### Probing Methodology

```
Patchscopes Framework
├── Extract: Get hidden representations
├── Patch: Intervene in forward pass
├── Measure: Evaluate output quality
└── Analyze: Quantify information encoding
```

## 📊 Metrics Computed

### Per-Layer Metrics

1. **Mean Log-Probability Difference**
   ```
   score = mean(log P(correct) - log P(incorrect))
   Higher = better anaphora encoding
   ```

2. **Accuracy**
   ```
   accuracy = proportion where log P(correct) > log P(incorrect)
   Range: [0, 1]
   ```

3. **Median Log-Probability Difference**
   ```
   Robust to outliers
   Better for skewed distributions
   ```

4. **Standard Deviation**
   ```
   Measures consistency across examples
   Lower = more consistent
   ```

### Aggregate Statistics

- **Best Layer**: Layer with highest mean score
- **Regional Performance**: Early/middle/late layer comparison
- **Condition Breakdown**: Performance by anaphora type

## 🎯 Output Schema

### layer_statistics.csv

```csv
layer,mean_logprob_diff,std_logprob_diff,median_logprob_diff,accuracy,n_examples
0,0.1234,0.0567,0.1123,0.65,50
1,0.2345,0.0678,0.2234,0.72,50
...
31,0.3456,0.0789,0.3345,0.81,50
```

### detailed_layer_results.csv

```csv
condition,item,source_text,0,1,2,...,31
stripping_VPE,1,"Alex passed Bo...",0.12,0.23,0.34,...,0.45
joins,1,"Alex introduced...",0.15,0.26,0.37,...,0.48
...
```

### summary.json

```json
{
  "model": "meta-llama/Llama-2-7b-chat-hf",
  "num_layers": 32,
  "num_examples": 80,
  "best_layer": 18,
  "best_layer_score": 0.4234,
  "layer_scores": [...]
}
```

## 🔍 Implementation Choices

### Design Decisions

1. **Patchscopes over Other Methods**
   - Causal intervention
   - Layer-specific analysis
   - Interpretable results

2. **Log-Probability Comparison**
   - Standard in NLP
   - Captures model confidence
   - Differentiable metric

3. **Batch Processing**
   - Efficiency
   - Consistent measurements
   - Scalable to full dataset

4. **Multiple Visualizations**
   - Different perspectives
   - Publication-ready
   - Comprehensive understanding

### Performance Optimizations

- **Float16 precision**: Reduce memory usage
- **Batch inference**: Speed up processing
- **Cached representations**: Avoid redundant computation
- **Configurable sampling**: Balance thoroughness vs. speed

## 🧪 Testing Strategy

### 1. Unit Testing (demo_layer_probing.py)
- Simple example
- Quick verification
- Component validation

### 2. Integration Testing (run_layer_probing.sh)
- End-to-end pipeline
- Full workflow
- Output validation

### 3. Visualization Testing
- Plot generation
- Data integrity
- Format correctness

## 📈 Scalability

### Current Capacity
- **Examples**: 10-100 per condition
- **Layers**: 32 (Llama-2-7b)
- **Time**: 15-30 min (GPU), 1-2 hrs (CPU)

### Extension Points
- **More models**: Add to model list
- **More layers**: Works with any transformer depth
- **More conditions**: Filter in configuration
- **More metrics**: Add to probing function

## 🔗 Dependencies

```
External:
├── torch >= 2.0
├── transformers >= 4.30
├── pandas >= 2.0
├── matplotlib >= 3.7
├── seaborn >= 0.12
└── tqdm >= 4.65

Internal:
├── patchscopes_utils.py (hooks)
├── general_utils.py (model wrapper)
└── AnaphoraGym.csv (data)
```

## 📝 Code Organization

```
scripts/mechanistic_analysis/
├── Core Components
│   ├── layer_wise_probing.py          # Main engine
│   ├── visualize_layer_probing.py     # Visualization
│   └── general_utils.py               # Shared utilities
│
├── Infrastructure
│   ├── patchscopes_utils.py           # Patching hooks
│   └── run_layer_probing.sh           # Pipeline
│
├── Testing & Demo
│   └── demo_layer_probing.py          # Quick demo
│
└── Documentation
    ├── LAYER_PROBING_README.md        # Full guide
    ├── ARCHITECTURE.md                # This file
    └── QUICK_START_LAYER_PROBING.md   # Quick start
```

## 🎓 Research Applications

This architecture enables:

1. **Comparative Studies**
   - Base vs. fine-tuned models
   - Different model sizes
   - Different architectures

2. **Mechanistic Understanding**
   - Where processing occurs
   - How information flows
   - Layer specialization

3. **Hypothesis Testing**
   - Effects of training
   - Architecture choices
   - Task-specific adaptation

---

**Architecture designed for:**
- ✅ Modularity
- ✅ Extensibility
- ✅ Reproducibility
- ✅ Interpretability
