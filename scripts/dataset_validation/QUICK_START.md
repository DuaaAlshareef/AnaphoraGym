# Quick Start: Dataset Validation

## What This Does

This validation suite establishes the credibility of your AnaphoraGym dataset through **objective linguistic metrics** that anyone can reproduce. Perfect for addressing "How do we know this dataset is trustworthy?" questions.

## Installation (One-Time)

```bash
# From project root
pip install textstat

# Or install all requirements
pip install -r requirements.txt
```

## Run the Analysis

### Option 1: Using the Shell Script (Recommended)

```bash
bash scripts/dataset_validation/run_validation.sh
```

### Option 2: Direct Python

```bash
python scripts/dataset_validation/validate_dataset.py
```

### Option 3: Custom Paths

```bash
python scripts/dataset_validation/validate_dataset.py \
    --dataset path/to/dataset.csv \
    --output path/to/output/
```

## What You Get

### 📊 Visualizations (5 PNG files)

1. **readability_distribution.png** - Shows your dataset spans multiple difficulty levels
2. **category_comparison.png** - Proves balanced complexity across categories
3. **input_vs_continuation.png** - Demonstrates systematic construction
4. **metrics_heatmap.png** - Comprehensive overview of all metrics
5. **complexity_vs_performance.png** - Validates metrics against actual model performance

### 📄 Report

**VALIDATION_REPORT.md** - Complete analysis with:
- Summary statistics
- Per-category breakdown
- Correlation with model performance
- Key findings for papers/presentations

### 📊 Data Files (3 CSV files)

- `readability_metrics.csv` - All readability scores
- `linguistic_features.csv` - All linguistic features
- `structural_metrics.csv` - Structural properties

## Results Location

Everything saves to: `results/dataset_validation/`

## Expected Runtime

- Small datasets (<100 items): ~5 seconds
- Your AnaphoraGym dataset: ~10 seconds
- Large datasets: ~30 seconds

## Interpreting Results

### Good Signs for Dataset Credibility

✅ **Wide range of scores** - Not artificially constrained  
✅ **FK Grade 8-16** - Appropriate difficulty  
✅ **Type-Token Ratio 0.4-0.8** - Good lexical diversity  
✅ **Correlation with model performance** - Metrics reflect real difficulty

### Use in Your Paper

Include statements like:

> "To validate dataset quality, we performed linguistic analysis using standard readability metrics. The dataset exhibits appropriate complexity (mean FK Grade: X.X) and substantial lexical diversity (TTR: 0.XX), with balanced distributions across categories. Importantly, linguistic complexity correlates with model performance (r=X.XX), demonstrating that our metrics reflect genuine task difficulty."

## Troubleshooting

### `ModuleNotFoundError: textstat`

```bash
pip install textstat
```

### Script won't run

```bash
chmod +x scripts/dataset_validation/run_validation.sh
bash scripts/dataset_validation/run_validation.sh
```

### Results look wrong

Check that your CSV has columns: `condition`, `input_1`, `continuation_1`, etc.

## Questions?

See the full `README.md` in this directory for detailed documentation.

## Quick Stats After Running

After running, you'll see output like:

```
📈 DATASET VALIDATION SUMMARY
============================================================
Total conditions: 9
Total items: 23
Total text segments: 142
Flesch Kincaid Grade: 10.52 (±3.14)
Type-Token Ratio: 0.72 (±0.18)
```

These numbers establish your dataset's credibility! 🎉
