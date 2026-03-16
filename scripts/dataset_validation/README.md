# Dataset Validation Analysis

This module performs comprehensive linguistic analysis on the AnaphoraGym dataset to establish its credibility and demonstrate objective quality metrics.

## Purpose

When presenting a curated dataset, it's important to provide objective evidence of its quality. This validation analysis addresses potential concerns by:

1. **Proving Non-triviality**: Showing the dataset isn't artificially simple
2. **Demonstrating Diversity**: Proving linguistic variety across examples
3. **Establishing Balance**: Showing systematic coverage across categories
4. **Providing Reproducibility**: Using standard metrics anyone can verify

## What It Does

### Metrics Calculated

#### 1. Readability Metrics
- **Flesch Reading Ease**: Overall readability (0-100 scale)
- **Flesch-Kincaid Grade Level**: Required education level
- **Gunning Fog Index**: Years of formal education needed
- **SMOG Index**: Grade level estimation
- **Coleman-Liau Index**: Grade level based on characters
- **Automated Readability Index**: Grade level using character-based formula
- **Dale-Chall Readability**: Difficulty based on familiar words

#### 2. Linguistic Features
- Word count and sentence count
- Average word length and syllable counts
- Type-Token Ratio (lexical diversity)
- Polysyllable and monosyllable counts
- Difficult words count

#### 3. Structural Metrics
- Input/continuation/test distribution
- Per-category statistics
- Test-worthy subset analysis

### Visualizations Generated

1. **`readability_distribution.png`**: Distribution histograms of 6 key readability metrics
2. **`category_comparison.png`**: Box plots comparing metrics across anaphora categories
3. **`input_vs_continuation.png`**: Comparison of linguistic properties between inputs and continuations
4. **`metrics_heatmap.png`**: Comprehensive heatmap showing all metrics by category
5. **`complexity_vs_performance.png`**: Dataset complexity vs model performance (if results available)

### Output Files

- **`VALIDATION_REPORT.md`**: Comprehensive markdown report with all findings
- **`readability_metrics.csv`**: Detailed readability scores for each text
- **`linguistic_features.csv`**: All linguistic features per text
- **`structural_metrics.csv`**: Structural properties per item

## Installation

Install required dependencies:

```bash
pip install textstat pandas numpy matplotlib seaborn
```

Or if using the project requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

From the project root directory:

```bash
python scripts/dataset_validation/validate_dataset.py
```

This will:
- Read `dataset/AnaphoraGym.csv`
- Generate all metrics and visualizations
- Save results to `results/dataset_validation/`

### Custom Paths

```bash
python scripts/dataset_validation/validate_dataset.py \
    --dataset path/to/your/dataset.csv \
    --output path/to/output/directory
```

### Using the Shell Script

```bash
bash scripts/dataset_validation/run_validation.sh
```

## Interpreting Results

### Readability Scores

- **Flesch-Kincaid Grade 8-12**: High school level (appropriate for testing)
- **Grade 13-16**: College level (indicates sophistication)
- **Gunning Fog > 12**: Complex, professional-level text

### Lexical Diversity

- **Type-Token Ratio (TTR)**:
  - 0.4-0.5: Good diversity for short texts
  - 0.6-0.8: High diversity
  - Higher = more unique words

### What Makes a Dataset Trustworthy?

1. **Wide Range**: Metrics show variance (not all examples are similar)
2. **Appropriate Difficulty**: Not trivially easy or impossibly hard
3. **Category Balance**: Similar metrics across different conditions
4. **Reproducibility**: Anyone can run these metrics and verify

## Example Output

After running, you'll see console output like:

```
📊 DATASET VALIDATION SUMMARY
============================================================
🔢 Overall Dataset Statistics:
   Total conditions: 10
   Total items: 23
   Total text segments: 142
   
📚 Readability Metrics (averaged across all texts):
   Flesch Kincaid Grade: 10.52 (±3.14)
   Gunning Fog: 12.83 (±3.67)
   ...

✅ VALIDATION COMPLETE!
```

And generated files in `results/dataset_validation/`:
- PNG visualizations
- CSV data files
- Markdown report

## Use in Papers/Presentations

When citing this analysis in your work:

> "To establish the dataset's quality, we performed comprehensive linguistic analysis using standard readability metrics (Flesch-Kincaid, Gunning Fog, SMOG Index). The analysis revealed [insert key finding, e.g., 'a mean Flesch-Kincaid grade level of X, indicating college-level complexity']. The dataset exhibits substantial linguistic diversity (Type-Token Ratio: X.XX) and balanced complexity across categories (see Figure X). These objective metrics demonstrate systematic construction and appropriate difficulty for testing advanced language understanding."

## Technical Details

- **Library**: Uses `textstat` Python library for standard metrics
- **Compatibility**: Works with the AnaphoraGym CSV format
- **Performance**: Processes the entire dataset in seconds
- **Extensibility**: Easy to add new metrics or visualizations

## Troubleshooting

### Import Error: textstat

```bash
pip install textstat
```

### Matplotlib Display Issues

If running on a server without display:

```python
import matplotlib
matplotlib.use('Agg')  # Add before importing pyplot
```

### CSV Format Issues

Ensure your dataset has the expected columns:
- `condition`, `item`
- `input_1`, `input_2`, etc.
- `continuation_1`, `continuation_2`, etc.

## Contact

For questions or issues with the validation analysis, please open an issue on the project repository.
