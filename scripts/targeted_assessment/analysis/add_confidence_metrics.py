"""
Add confidence and calibration metrics to model results

Uses the magnitude of log odds as a confidence measure.
Higher |log odds| = more confident prediction.
"""
import pandas as pd
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir


def add_confidence_metrics(model_name: str):
    """
    Add confidence metrics to existing results file.
    
    Confidence is measured as |log odds|:
    - High |log odds| = confident prediction
    - Low |log odds| = uncertain prediction
    """
    results_dir = get_results_dir()
    safe_model_name = model_name.replace('/', '_')
    results_path = os.path.join(results_dir, f"AnaphoraGym_Results_{safe_model_name}.csv")
    
    if not os.path.exists(results_path):
        print(f"❌ Results not found: {results_path}")
        return
    
    print(f"📊 Adding confidence metrics to {model_name} results...")
    
    # Load results
    df = pd.read_csv(results_path)
    
    # Calculate confidence (absolute value of log odds)
    df['confidence'] = np.abs(df['logOdds'])
    
    # Normalize confidence to 0-1 scale for easier interpretation
    if df['confidence'].max() > 0:
        df['confidence_normalized'] = df['confidence'] / df['confidence'].max()
    else:
        df['confidence_normalized'] = 0
    
    # Add confidence bins for calibration analysis
    df['confidence_bin'] = pd.cut(df['confidence'], 
                                   bins=5, 
                                   labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Calculate if prediction was correct (1) or wrong (0)
    df['correct'] = df['test_passed'].astype(int)
    
    # Save enhanced results
    output_path = os.path.join(results_dir, f"AnaphoraGym_Results_{safe_model_name}_with_confidence.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved enhanced results: {output_path}")
    
    # Print summary statistics
    print(f"\n📈 Confidence Statistics:")
    print(f"   Mean confidence: {df['confidence'].mean():.3f}")
    print(f"   Median confidence: {df['confidence'].median():.3f}")
    print(f"   Std confidence: {df['confidence'].std():.3f}")
    
    print(f"\n📊 Accuracy by Confidence Level:")
    calibration = df.groupby('confidence_bin', observed=True).agg({
        'correct': ['count', 'sum', 'mean']
    }).round(3)
    calibration.columns = ['Total', 'Correct', 'Accuracy']
    print(calibration)
    
    # Calculate correlation between confidence and correctness
    correlation = df['confidence'].corr(df['correct'])
    print(f"\n🔗 Correlation (confidence vs correctness): {correlation:.3f}")
    if correlation > 0.3:
        print("   → Strong positive: Model is well-calibrated!")
    elif correlation > 0.1:
        print("   → Moderate positive: Some calibration")
    else:
        print("   → Weak/no correlation: Model may be poorly calibrated")
    
    # Average confidence for correct vs incorrect
    print(f"\n✅ Average confidence when CORRECT: {df[df['correct']==1]['confidence'].mean():.3f}")
    print(f"❌ Average confidence when WRONG: {df[df['correct']==0]['confidence'].mean():.3f}")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add confidence and calibration metrics to model results"
    )
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., gpt2)")
    args = parser.parse_args()
    
    add_confidence_metrics(args.model)
