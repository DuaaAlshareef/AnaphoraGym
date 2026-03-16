# Model Calibration Analysis

## What is Calibration?

**Calibration** measures how well a model's confidence matches its actual accuracy.

- **Well-calibrated**: When confident → usually correct, When uncertain → usually wrong
- **Poorly-calibrated**: Confidence doesn't match correctness (overconfident or underconfident)

## How We Measure It

**Confidence** = `|log odds|` (absolute value of log-likelihood difference)
- High confidence = large |log odds| = model has strong preference
- Low confidence = small |log odds| = model is uncertain

## Quick Start

### Analyze Calibration for GPT2:

```bash
bash scripts/targeted_assessment/analyze_calibration.sh gpt2
```

This will:
1. Add confidence metrics to your results CSV
2. Create a 4-panel calibration visualization

### Or Run Steps Separately:

```bash
# Step 1: Add confidence metrics
python scripts/targeted_assessment/analysis/add_confidence_metrics.py --model gpt2

# Step 2: Visualize calibration
python scripts/targeted_assessment/visualization/visualize_calibration.py --model gpt2
```

## Output Files

### 1. Enhanced Results CSV
`AnaphoraGym_Results_gpt2_with_confidence.csv`

**New columns added:**
- `confidence`: Absolute log odds value
- `confidence_normalized`: Scaled 0-1
- `confidence_bin`: Category (Very Low, Low, Medium, High, Very High)
- `correct`: 1 if test passed, 0 if failed

### 2. Calibration Visualization
`calibration_analysis_gpt2.png`

**Four panels:**

#### Panel 1: Calibration Curve
- Shows if high confidence → high accuracy
- Red dashed line = perfect calibration
- Blue line = actual model behavior
- **Good calibration**: Blue line close to red line

#### Panel 2: Confidence Distribution
- Green = confidence when correct
- Red = confidence when incorrect
- **Well-calibrated**: Green shifted right (higher) than red

#### Panel 3: Accuracy by Confidence Level
- Shows accuracy for each confidence bin
- **Well-calibrated**: Bars increase left to right

#### Panel 4: Summary Statistics
- Overall calibration metrics
- Correlation score
- Calibration status

## Interpreting Results

### Correlation Score:
- **> 0.3**: Well-calibrated (confidence predicts correctness)
- **0.1 - 0.3**: Moderately calibrated
- **< 0.1**: Poorly calibrated (confidence unreliable)

### Confidence Difference:
- **Large difference** (>0.1): Model is more confident when correct
- **Small difference** (<0.1): Similar confidence for both

### Example Interpretation:

```
Correlation: 0.45
Mean confidence (correct): 2.5
Mean confidence (wrong): 1.8
Difference: 0.7

→ WELL CALIBRATED ✓
→ Model shows higher confidence when making correct predictions
→ Can use confidence to identify likely correct answers
```

## Use in Papers

### Methods Section:
> "We assessed model calibration by examining the relationship between prediction confidence (measured as the absolute log-likelihood difference) and accuracy. A well-calibrated model exhibits higher confidence when making correct predictions."

### Results Section:
> "Calibration analysis revealed [correlation coefficient] between confidence and correctness, indicating [well/moderately/poorly] calibrated predictions. The model exhibited [X%] higher average confidence for correct predictions compared to incorrect ones."

### Key Metrics to Report:
- Correlation (confidence vs correctness)
- Average confidence for correct predictions
- Average confidence for incorrect predictions
- Accuracy by confidence level

## Why This Matters

### For Research:
1. **Trust**: Well-calibrated models are more trustworthy
2. **Uncertainty**: Can identify when model is uncertain
3. **Filtering**: Can filter predictions by confidence threshold

### For Applications:
1. **Decision-making**: Use confidence to decide when to trust predictions
2. **Human-in-the-loop**: Flag low-confidence predictions for human review
3. **Ensemble**: Weight models by their calibration quality

## Example Use Cases

### 1. Confidence Threshold
```python
# Only trust high-confidence predictions
high_confidence_predictions = df[df['confidence'] > 2.0]
accuracy = high_confidence_predictions['correct'].mean()
```

### 2. Identify Uncertain Cases
```python
# Find cases where model is uncertain
uncertain = df[df['confidence'] < 0.5]
# These might need human review
```

### 3. Compare Model Calibration
```python
# Compare GPT2 vs GPT2-large calibration
# Better calibrated model is more reliable
```

## Next Steps

After running the basic calibration analysis:

1. **Compare models**: Run for multiple models, compare calibration quality
2. **Per-category**: Analyze if some anaphora types are better calibrated
3. **Threshold tuning**: Find optimal confidence threshold for your use case

## Advanced (Coming Next)

The current version uses simple |log odds| as confidence. 

**Future enhancements could include:**
- Temperature scaling for better calibration
- Platt scaling
- Isotonic regression
- Per-category calibration
- Calibration error metrics (ECE, MCE)

But start with this simple version first! It's often sufficient and easy to interpret.
