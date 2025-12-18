"""
Create enriched dataset by merging original dataset with model results.

This script creates both WIDE and ROW-WISE (long) format enriched datasets
that combine the original AnaphoraGym dataset with model performance metrics.
"""
import pandas as pd
import glob
import os
import sys
from typing import List, Dict

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import get_dataset_path, get_results_dir, load_dataset


# Metrics we care about
METRICS = ['llh_left', 'llh_right', 'logodds', 'test_passed']

# Column variants we normalize -> lowercase canonical names
NORMALIZE_COLS = {
    'LLH_left': 'llh_left',
    'LLH_right': 'llh_right',
    'logOdds': 'logodds',
    'LogOdds': 'logodds',
    'log_odds': 'logodds',
    'test_passed': 'test_passed',
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to canonical lowercase forms."""
    rename_map = {c: NORMALIZE_COLS[c] for c in df.columns if c in NORMALIZE_COLS}
    return df.rename(columns=rename_map)


def _load_results_frames(results_dir: str) -> List[pd.DataFrame]:
    """Load all result CSV files and normalize their columns."""
    pattern = os.path.join(results_dir, "AnaphoraGym_Results_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"[ERROR] No result files found under: {results_dir}")
        return []
    
    print(f"Found {len(files)} result file(s).")
    frames = []
    
    for fp in files:
        print(f"  - Reading {os.path.basename(fp)}")
        df = pd.read_csv(fp)
        df = _normalize_columns(df)
        
        missing = [
            c for c in ['condition', 'item', 'model_source', 'test_name']
            if c not in df.columns
        ]
        if missing:
            print(f"    [WARN] Skipping {os.path.basename(fp)}; missing columns: {missing}")
            continue
        frames.append(df)
    
    return frames


def _build_keymap(master: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Build mapping: model_test_key -> {'model_sanitized':..., 'test_name':...}"""
    master['model_sanitized'] = master['model_source'].astype(str).str.replace('/', '_', regex=False)
    master['model_test_key'] = master['model_sanitized'] + "_" + master['test_name'].astype(str)
    key_map = (
        master[['model_test_key', 'model_sanitized', 'test_name']]
        .drop_duplicates()
        .set_index('model_test_key')[['model_sanitized', 'test_name']]
        .to_dict('index')
    )
    return key_map


def _rename_metric_columns(cols: List[str], metric: str, key_map: Dict[str, Dict[str, str]]) -> List[str]:
    """Rename pivoted metric columns according to specification."""
    new = []
    for c in cols:
        info = key_map.get(c)
        if not info:
            new.append(c)  # fallback
            continue
        model = info['model_sanitized']
        test = str(info['test_name'])
        if metric == 'llh_left':
            new.append(f"{model}_{test}__LLH_test_{test}_1")
        elif metric == 'llh_right':
            new.append(f"{model}_{test}__LLH_test_{test}_2")
        elif metric == 'logodds':
            new.append(f"{model}_{test}__logOdds_test_{test}")
        elif metric == 'test_passed':
            new.append(f"{model}_{test}__test_passed")
        else:
            new.append(c)  # unknown metric
    return new


def create_enriched_dataset():
    """Create enriched datasets in both WIDE and ROW-WISE formats."""
    print("--- Creating Enriched Results (WIDE + ROW-WISE) ---")
    
    results_dir = get_results_dir()
    dataset_path = get_dataset_path()
    
    # Load all results
    frames = _load_results_frames(results_dir)
    if not frames:
        print("[ABORT] No usable results.")
        return
    
    master = pd.concat(frames, ignore_index=True)
    
    # Key mapping and identifiers
    key_map = _build_keymap(master)
    
    # Build WIDE (pivoted) table
    print("\nBuilding WIDE table...")
    wide = None
    
    for metric in METRICS:
        if metric not in master.columns:
            print(f"  [INFO] Missing metric '{metric}' in results; skipping.")
            continue
        
        pivot = master.pivot_table(
            index=['condition', 'item'],
            columns='model_test_key',
            values=metric,
            aggfunc='first'
        ).reset_index()
        
        # Rename pivot columns per spec
        orig_cols = list(pivot.columns[2:])
        renamed = _rename_metric_columns(orig_cols, metric, key_map)
        pivot.columns = ['condition', 'item'] + renamed
        
        wide = pivot if wide is None else pd.merge(
            wide, pivot, on=['condition', 'item'], how='outer'
        )
    
    if wide is None:
        print("[ERROR] No metrics found to pivot. Aborting.")
        return
    
    # Reorder columns to group by <model>_<test>
    print("Reordering WIDE columns by model/test group...")
    id_cols = ['condition', 'item']
    
    # Keep order of first appearance for groups
    seen_keys = []
    for c in wide.columns:
        if '__' in c:
            key = c.split('__')[0]  # <model>_<test>
            if key not in seen_keys:
                seen_keys.append(key)
    
    # Build inverse mapping: <model>_<test> -> test
    inverse_test = {}
    for k, v in key_map.items():
        inverse_test[f"{v['model_sanitized']}_{v['test_name']}"] = str(v['test_name'])
    
    # Build ordered column list
    ordered_cols = id_cols.copy()
    for key in seen_keys:
        test = inverse_test.get(key, key.split('_', 1)[1] if '_' in key else key)
        
        left = f"{key}__LLH_test_{test}_1"
        right = f"{key}__LLH_test_{test}_2"
        odds = f"{key}__logOdds_test_{test}"
        passed = f"{key}__test_passed"
        
        for col in (left, right, odds, passed):
            if col in wide.columns:
                ordered_cols.append(col)
    
    wide = wide[ordered_cols]
    
    # Load original dataset & merge
    try:
        original = load_dataset()
        print(f"\nLoaded original dataset")
    except FileNotFoundError:
        print(f"[ERROR] Original dataset not found at '{dataset_path}'. Aborting merge.")
        return
    
    # Drop patching_prompts if present
    to_drop = [f'patching_prompt_{i}' for i in range(1, 5)]
    existing_drop = [c for c in to_drop if c in original.columns]
    original_clean = original.drop(columns=existing_drop)
    if existing_drop:
        print(f"Removed columns from original: {existing_drop}")
    
    print("Merging original dataset with WIDE metrics...")
    enriched_wide = pd.merge(original_clean, wide, on=['condition', 'item'], how='left')
    
    # Save WIDE
    wide_output_path = os.path.join(results_dir, "AnaphoraGym_Enriched_Results_WithLLH_WIDE.csv")
    enriched_wide.to_csv(wide_output_path, index=False)
    print(f"SUCCESS: WIDE file written to: {wide_output_path}")
    print(enriched_wide.head())
    
    # Build ROW-WISE (long) enriched table
    print("\nBuilding ROW-WISE table...")
    needed_cols = [
        'condition', 'item', 'model_source', 'test_name',
        'llh_left', 'llh_right', 'logodds', 'test_passed'
    ]
    missing = [c for c in needed_cols if c not in master.columns]
    if missing:
        print(f"[ERROR] Cannot build row-wise table; missing columns: {missing}")
        return
    
    long_df = (
        master[needed_cols].copy()
        .rename(columns={
            'llh_left': 'LLH_test_1',
            'llh_right': 'LLH_test_2',
            'logodds': 'logOdds_test'
        })
    )
    
    print("Merging original dataset with ROW-WISE metrics...")
    enriched_long = pd.merge(original_clean, long_df, on=['condition', 'item'], how='left')
    
    long_output_path = os.path.join(results_dir, "AnaphoraGym_Enriched_Results_Rowwise.csv")
    enriched_long.to_csv(long_output_path, index=False)
    print(f"SUCCESS: ROW-WISE file written to: {long_output_path}")
    print(enriched_long.head())


if __name__ == "__main__":
    try:
        create_enriched_dataset()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

