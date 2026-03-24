# Calibration Explained Simply

## What Is Calibration?

**Calibration = Does the model "know when it knows"?**

- **Well-calibrated**: High confidence → usually correct, Low confidence → usually wrong
- **Poorly-calibrated**: Confidence doesn't match correctness

## How We Measure Confidence

**Confidence = |log odds|** (absolute value of the log-likelihood difference)

```
High |log odds| (e.g., 3.0) = "I'm sure about this"
Low |log odds| (e.g., 0.5) = "I'm uncertain"
```

## The Calibration Score

**Calibration Score = Correlation between confidence and correctness**

This single number tells you everything:

| Score | Meaning | Can you trust confidence? |
|-------|---------|---------------------------|
| ≥ 0.3 | **Well calibrated** | ✅ YES - Use confidence to filter predictions |
| 0.1-0.3 | **Moderately calibrated** | ~ MAYBE - Some reliability |
| < 0.1 | **Poorly calibrated** | ❌ NO - Confidence is meaningless |

## What The Numbers Mean

### Example 1: Well-Calibrated Model
```
Calibration Score: 0.45
Mean confidence (correct): 2.8
Mean confidence (wrong): 1.5

→ When model is confident (high score), it's usually RIGHT
→ When model is uncertain (low score), it's often WRONG
→ You can TRUST the confidence scores!
```

### Example 2: Poorly-Calibrated Model
```
Calibration Score: 0.05
Mean confidence (correct): 2.1
Mean confidence (wrong): 2.0

→ Same confidence whether right or wrong
→ Confidence tells you NOTHING
→ Can't use confidence to filter predictions
```

## Why This Matters

### For Research:
- **Well-calibrated models are more trustworthy**
- Shows the model has genuine uncertainty awareness
- Can identify when model is "guessing"

### For Practice:
```python
# Example: Only trust high-confidence predictions
if confidence > 2.0:
    # Well-calibrated model → likely correct
    use_prediction()
else:
    # Model is uncertain → might be wrong
    flag_for_human_review()
```

## Visual Interpretation

In the graph you see:

```
gpt2         ████████████ 0.450 | Acc: 65%    ← GREEN = Well calibrated
pythia       ████████ 0.250 | Acc: 55%        ← ORANGE = Moderate  
llama        ████ 0.080 | Acc: 60%            ← RED = Poor
                    ↑
              0.3 threshold
```

**Green bars (≥0.3)**: Trust these models' confidence!
**Orange bars (0.1-0.3)**: Some reliability
**Red bars (<0.1)**: Don't trust confidence

## The Bottom Line

**Calibration Score = One number that tells you:**
*"Can I trust this model's confidence scores?"*

- High score (≥0.3) → YES, use confidence
- Low score (<0.1) → NO, confidence is random

That's it! 🎯
