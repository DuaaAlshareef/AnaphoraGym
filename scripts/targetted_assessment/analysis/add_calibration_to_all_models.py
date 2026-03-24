"""
Add calibration metrics to ALL existing model results
"""
import pandas as pd
import numpy as np
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_results_dir


def add_calibration_to_all():
    """Add calibration metrics to all existing model results."""
    results_dir = get_results_dir()
    
    # Find all model result files (exclude enriched/summary files)
    pattern = os.path.join(results_dir, "AnaphoraGym_Results_*.csv")
    all_files = glob.glob(pattern)
    
    # Filter to only main result files
    result_files = [f for f in all_files 
                   if 'Enriched' not in f 
                   and 'Concatenated' not in f
                   and 'Patchscope' not in f
                   and 'with_confidence' not in f]
    
    if not result_files:
        print("❌ No result files found!")
        return
    
    print(f"📊 Found {len(result_files)} model result files")
    print("="*70)
    
    all_calibration_stats = []
    
    for filepath in sorted(result_files):
        filename = os.path.basename(filepath)
        model_name = filename.replace('AnaphoraGym_Results_', '').replace('.csv', '')
        
        print(f"\n📈 Processing: {model_name}")
        
        # Load results
        df = pd.read_csv(filepath)
        
        # Add confidence metrics
        df['confidence'] = np.abs(df['logOdds'])
        df['correct'] = df['test_passed'].astype(int)
        
        # Calculate statistics
        overall_acc = df['correct'].mean()
        mean_conf = df['confidence'].mean()
        mean_conf_correct = df[df['correct']==1]['confidence'].mean()
        mean_conf_wrong = df[df['correct']==0]['confidence'].mean()
        correlation = df['confidence'].corr(df['correct'])
        
        print(f"   Accuracy: {overall_acc*100:.1f}%")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Confidence (correct): {mean_conf_correct:.3f}")
        print(f"   Confidence (wrong): {mean_conf_wrong:.3f}")
        
        # Store stats for summary
        all_calibration_stats.append({
            'model': model_name,
            'accuracy': overall_acc,
            'mean_confidence': mean_conf,
            'confidence_correct': mean_conf_correct,
            'confidence_wrong': mean_conf_wrong,
            'confidence_diff': mean_conf_correct - mean_conf_wrong,
            'correlation': correlation,
            'n_tests': len(df)
        })
        
        # Save enhanced results
        output_path = filepath.replace('.csv', '_with_confidence.csv')
        df.to_csv(output_path, index=False)
        print(f"   ✅ Saved: {os.path.basename(output_path)}")
    
    # Create calibration summary
    summary_df = pd.DataFrame(all_calibration_stats)
    summary_df = summary_df.sort_values('correlation', ascending=False)
    
    summary_path = os.path.join(results_dir, 'model_calibration_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    print("\n" + "="*70)
    print("📊 CALIBRATION SUMMARY (sorted by correlation)")
    print("="*70)
    print(summary_df.to_string(index=False))
    print("\n✅ Saved summary: model_calibration_summary.csv")
    
    # Identify best/worst calibrated
    best = summary_df.iloc[0]
    worst = summary_df.iloc[-1]
    
    print("\n" + "="*70)
    print(f"🏆 Best Calibrated: {best['model']}")
    print(f"   Correlation: {best['correlation']:.3f}")
    print(f"   Confidence diff: {best['confidence_diff']:.3f}")
    
    print(f"\n⚠️  Worst Calibrated: {worst['model']}")
    print(f"   Correlation: {worst['correlation']:.3f}")
    print(f"   Confidence diff: {worst['confidence_diff']:.3f}")
    print("="*70)


if __name__ == "__main__":
    add_calibration_to_all()
