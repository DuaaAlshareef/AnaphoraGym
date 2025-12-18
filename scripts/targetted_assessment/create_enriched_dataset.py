# # # ==============================================================================
# # # SCRIPT TO CREATE A SINGLE, CLEAN ENRICHED CSV WITH TRUE/FALSE RESULTS
# # #
# # # This version:
# # #   - Finds all individual model result files.
# # #   - Merges only the True/False `test_passed` results back into the original data.
# # #   - Removes the original `patching_prompt` columns for a cleaner final file.
# # # ==============================================================================

# # import pandas as pd
# # import glob
# # import os

# # # --- 1. DEFINE PROJECT PATHS ---
# # try:
# #     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# #     PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# # except NameError:
# #     PROJECT_ROOT = os.path.abspath('.')

# # DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'AnaphoraGym.csv')
# # RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')

# # def main():
# #     print("--- Creating Clean Enriched Dataset with True/False Results ---")

# #     # --- Step 1: Find and Consolidate All Result Files ---
# #     search_pattern = os.path.join(RESULTS_DIR, "AnaphoraGym_Results_*.csv")
# #     result_files = glob.glob(search_pattern)

# #     if not result_files:
# #         print(f"Analysis failed: No result files found in '{RESULTS_DIR}'.")
# #         print("Please run the `run_all.sh` script first.")
# #         return

# #     print(f"Found {len(result_files)} result files to consolidate.")
    
# #     all_results_dfs = []
# #     for filepath in result_files:
# #         print(f"  - Reading {os.path.basename(filepath)}")
# #         df = pd.read_csv(filepath)
# #         all_results_dfs.append(df)

# #     master_results_df = pd.concat(all_results_dfs, ignore_index=True)

# #     # --- Step 2: Reshape Only the 'test_passed' Results ---
# #     print("\nReshaping results data for merging...")
    
# #     # Create a unique key for each model and test combination
# #     master_results_df['model_test_key'] = master_results_df['model_source'].str.replace('/', '_') + "_" + master_results_df['test_name']
    
# #     # Pivot ONLY for the 'test_passed' results
# #     results_wide_df = master_results_df.pivot_table(
# #         index=['condition', 'item'],
# #         columns='model_test_key',
# #         values='test_passed' # We only pivot this value
# #     ).reset_index()
    
# #     # Convert 1.0/0.0 from the pivot back to True/False for clarity
# #     for col in results_wide_df.columns:
# #         if col not in ['condition', 'item']:
# #             results_wide_df[col] = results_wide_df[col].apply(
# #                 lambda x: True if x == 1.0 else (False if x == 0.0 else pd.NA)
# #             )

# #     print("Reshaping complete.")

# #     # --- Step 3: Load the Original Dataset and Merge ---
# #     try:
# #         original_df = pd.read_csv(DATASET_PATH)
# #         print(f"Successfully loaded original dataset from '{DATASET_PATH}'.")
# #     except FileNotFoundError:
# #         print(f"[ERROR] Original dataset not found at '{DATASET_PATH}'.")
# #         return
        
# #     # ================== THIS IS THE CRITICAL CHANGE ==================
# #     # Drop the patching prompt columns from the original data before merging
# #     columns_to_drop = [f'patching_prompt_{i}' for i in range(1, 5)]
# #     # We use a list comprehension and check if the column exists to avoid errors
# #     existing_columns_to_drop = [col for col in columns_to_drop if col in original_df.columns]
# #     original_df_clean = original_df.drop(columns=existing_columns_to_drop)
# #     print(f"Removed columns: {existing_columns_to_drop}")
# #     # =================================================================

# #     print("Merging original data with results...")
# #     enriched_df = pd.merge(original_df_clean, results_wide_df, on=['condition', 'item'], how='left')

# #     # --- Step 4: Save the Final Enriched CSV ---
# #     output_path = os.path.join(RESULTS_DIR, "AnaphoraGym_Enriched_Results_Clean.csv")
    
# #     try:
# #         enriched_df.to_csv(output_path, index=False)
# #         print(f"\nSUCCESS: Clean, enriched dataset saved to '{output_path}'")
# #         print("\nHere's a preview of the new file structure:")
# #         print(enriched_df.head())
        
# #     except Exception as e:
# #         print(f"\n[ERROR] Could not save final report. Reason: {e}")


# # if __name__ == "__main__":
# #     main()



# #!/usr/bin/env python3
# # ==============================================================================
# # Create a single, clean enriched CSV including llh_left, llh_right, logodds,
# # and test_passed for ALL models/tests, merged back into the original dataset.
# #
# # Output columns (wide):
# #   condition, item, and for each model+test key:
# #     <model>_<test>__llh_left
# #     <model>_<test>__llh_right
# #     <model>_<test>__logodds
# #     <model>_<test>__test_passed
# #
# # Differences vs your previous script:
# #   - Pivots 4 metrics, not just test_passed.
# #   - Normalizes common column name variants (LLH_left/LLH_right/logOdds).
# #   - Drops patching_prompt_* columns from the original dataset before merge.
# # ==============================================================================

# import pandas as pd
# import glob
# import os

# # --- 1. DEFINE PROJECT PATHS ---
# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# except NameError:
#     PROJECT_ROOT = os.path.abspath('.')

# DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'AnaphoraGym.csv')
# RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
# OUTPUT_PATH  = os.path.join(RESULTS_DIR, "AnaphoraGym_Enriched_Results_WithLLH.csv")

# METRICS = ['llh_left', 'llh_right', 'logodds', 'test_passed']

# # Map common column variants -> normalized names used in METRICS
# NORMALIZE_COLS = {
#     'LLH_left': 'llh_left',
#     'LLH_right': 'llh_right',
#     'logOdds': 'logodds',
#     'LogOdds': 'logodds',
#     'log_odds': 'logodds',
#     'test_passed': 'test_passed',
# }

# def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Return a copy with known variant column names normalized to expected names."""
#     rename_map = {col: NORMALIZE_COLS[col] for col in df.columns if col in NORMALIZE_COLS}
#     out = df.rename(columns=rename_map)
#     return out

# def main():
#     print("--- Creating Clean Enriched Dataset with LLH + LogOdds + Pass/Fail ---")

#     # --- Step 1: Find and Load All Result Files ---
#     search_pattern = os.path.join(RESULTS_DIR, "AnaphoraGym_Results_*.csv")
#     result_files = glob.glob(search_pattern)

#     if not result_files:
#         print(f"Analysis failed: No result files found in '{RESULTS_DIR}'.")
#         print("Please run the `run_all.sh` script first.")
#         return

#     print(f"Found {len(result_files)} result files to consolidate.")
#     all_results = []
#     for filepath in result_files:
#         print(f"  - Reading {os.path.basename(filepath)}")
#         df = pd.read_csv(filepath)
#         df = _normalize_columns(df)
#         # Sanity: keep only rows that have the basics we need
#         missing = [c for c in ['condition','item'] if c not in df.columns]
#         if missing:
#             print(f"    [WARN] Skipping {os.path.basename(filepath)}; missing {missing}")
#             continue
#         all_results.append(df)

#     if not all_results:
#         print("[ERROR] No usable result files after validation.")
#         return

#     master = pd.concat(all_results, ignore_index=True)

#     # --- Step 2: Build model_test_key ---
#     # We need model_source + test_name to uniquely identify each metric column.
#     if 'model_source' not in master.columns or 'test_name' not in master.columns:
#         print("[ERROR] Results must include 'model_source' and 'test_name' columns.")
#         return

#     master['model_test_key'] = (
#         master['model_source'].astype(str).str.replace('/', '_', regex=False)
#         + "_" +
#         master['test_name'].astype(str)
#     )

#     # --- Step 3: Pivot EACH metric into wide format, then merge them together ---
#     print("\nReshaping metrics (llh_left, llh_right, logodds, test_passed)...")

#     # Build a mapping from the wide 'model_test_key' to (model_sanitized, test_name)
#     master['model_sanitized'] = master['model_source'].astype(str).str.replace('/', '_', regex=False)
#     key_map = (
#         master[['model_sanitized', 'test_name']]
#         .drop_duplicates()
#         .assign(model_test_key=lambda d: d['model_sanitized'] + "_" + d['test_name'])
#         .set_index('model_test_key')[['model_sanitized','test_name']].to_dict('index')
#     )

#     def rename_columns_for_metric(cols, metric):
#         """Given pivoted columns (model_test_key), return renamed list per spec."""
#         new = []
#         for c in cols:
#             if c not in key_map:
#                 # fallback: leave as-is
#                 new.append(c)
#                 continue
#             model = key_map[c]['model_sanitized']
#             test  = key_map[c]['test_name']
#             if metric == 'llh_left':
#                 new.append(f"{model}_{test}__LLH_test_{test}_1")
#             elif metric == 'llh_right':
#                 new.append(f"{model}_{test}__LLH_test_{test}_2")
#             elif metric == 'logodds':
#                 new.append(f"{model}_{test}__logOdds_test_{test}")
#             elif metric == 'test_passed':
#                 new.append(f"{model}_{test}__test_passed")
#             # If you later add patch-scoped metrics, add cases for 'ps_llh_left', etc.
#         return new

#     METRICS = ['llh_left', 'llh_right', 'logodds', 'test_passed']

#     wide = None
#     for metric in METRICS:
#         if metric not in master.columns:
#             print(f"  [INFO] Metric '{metric}' not found in data; skipping.")
#             continue

#         pivot = master.pivot_table(
#             index=['condition','item'],
#             columns='model_test_key',
#             values=metric,
#             aggfunc='first'
#         ).reset_index()

#         # rename pivoted metric columns to your desired names
#         orig_cols = list(pivot.columns[2:])
#         renamed   = rename_columns_for_metric(orig_cols, metric)
#         pivot.columns = ['condition','item'] + renamed

#         wide = pivot if wide is None else pd.merge(wide, pivot, on=['condition','item'], how='outer')

#     # --- Reorder columns to group metrics per model/test in the order L, R, logOdds, passed ---
#     id_cols = ['condition', 'item']
#     metric_order_suffixes = [
#         "_LLH_test_",   # left/right will be detected together
#         "_LLH_test_",   # we’ll handle order via the trailing _1/_2
#         "_logOdds_test_",
#         "_test_passed"
#     ]

#     # use order of first appearance for (model,test) groups
#     seen_keys = []  # stores "<model>_<test>"
#     for c in wide.columns:
#         if '__' in c:
#             key = c.split('__')[0]
#             if key not in seen_keys:
#                 seen_keys.append(key)

#     ordered_cols = id_cols.copy()
#     for key in seen_keys:
#         # exact order: left (…_1), right (…_2), logOdds, test_passed
#         left  = f"{key}_LLH_test_{key.split('_', maxsplit=1)[1]}_1"
#         right = f"{key}_LLH_test_{key.split('_', maxsplit=1)[1]}_2"
#         logodds = f"{key}_logOdds_test_{key.split('_', maxsplit=1)[1]}"
#         passed  = f"{key}_test_passed"
#         for col in (left, right, logodds, passed):
#             if col in wide.columns:
#                 ordered_cols.append(col)

#     wide = wide[ordered_cols]


#     # --- Step 4: Load Original Dataset, drop patching_prompt_* columns, and merge ---
#     try:
#         original = pd.read_csv(DATASET_PATH)
#         print(f"\nLoaded original dataset: {DATASET_PATH}")
#     except FileNotFoundError:
#         print(f"[ERROR] Original dataset not found at '{DATASET_PATH}'.")
#         return

#     # Drop patching prompt columns if present
#     to_drop = [f'patching_prompt_{i}' for i in range(1,5)]
#     existing_drop = [c for c in to_drop if c in original.columns]
#     original_clean = original.drop(columns=existing_drop)
#     if existing_drop:
#         print(f"Removed columns: {existing_drop}")

#     print("Merging original data with wide metrics...")
#     enriched = pd.merge(original_clean, wide, on=['condition','item'], how='left')

#     # --- Step 5: Save Final CSV ---
#     try:
#         enriched.to_csv(OUTPUT_PATH, index=False)
#         print(f"\nSUCCESS: Enriched dataset saved to '{OUTPUT_PATH}'")
#         print("\nPreview:")
#         print(enriched.head())
#     except Exception as e:
#         print(f"\n[ERROR] Could not save final report. Reason: {e}")

# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
# ==============================================================================
# Create enriched results for AnaphoraGym
#
# Outputs:
#   (1) WIDE  -> results/targetted_assessment/AnaphoraGym_Enriched_Results_WithLLH_WIDE.csv
#       Columns per <model>_<test> group in this order:
#         <model>_<test>__LLH_test_<test>_1
#         <model>_<test>__LLH_test_<test>_2
#         <model>_<test>__logOdds_test_<test>
#         <model>_<test>__test_passed
#
#   (2) ROW-WISE -> results/targetted_assessment/AnaphoraGym_Enriched_Results_Rowwise.csv
#       One row per (condition, item, model_source, test_name) with columns:
#         LLH_test_1, LLH_test_2, logOdds_test, test_passed
#
# Notes:
#   - Keeps test_passed as 0/1 (does NOT convert to booleans).
#   - Ignores patch-scoped (PS_*) metrics for now.
#   - Drops patching_prompt_1..4 from the original dataset before merging.
# ==============================================================================

import pandas as pd
import glob
import os
from typing import List, Dict

# --- 1) PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    PROJECT_ROOT = os.path.abspath('.')

DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'AnaphoraGym.csv')
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')

WIDE_OUTPUT_PATH = os.path.join(RESULTS_DIR, "AnaphoraGym_Enriched_Results_WithLLH_WIDE.csv")
LONG_OUTPUT_PATH = os.path.join(RESULTS_DIR, "AnaphoraGym_Enriched_Results_Rowwise.csv")

# Metrics we care about (vanilla only)
METRICS = ['llh_left', 'llh_right', 'logodds', 'test_passed']

# Column variants we normalize -> lowercase canonical names above
NORMALIZE_COLS = {
    'LLH_left': 'llh_left',
    'LLH_right': 'llh_right',
    'logOdds': 'logodds',
    'LogOdds': 'logodds',
    'log_odds': 'logodds',
    'test_passed': 'test_passed',
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {c: NORMALIZE_COLS[c] for c in df.columns if c in NORMALIZE_COLS}
    return df.rename(columns=rename_map)

def _load_results_frames(results_dir: str) -> List[pd.DataFrame]:
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
        missing = [c for c in ['condition', 'item', 'model_source', 'test_name'] if c not in df.columns]
        if missing:
            print(f"    [WARN] Skipping {os.path.basename(fp)}; missing columns: {missing}")
            continue
        frames.append(df)
    return frames

def _build_keymap(master: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Return mapping: model_test_key -> {'model_sanitized':..., 'test_name':...}"""
    master['model_sanitized'] = master['model_source'].astype(str).str.replace('/', '_', regex=False)
    master['model_test_key'] = master['model_sanitized'] + "_" + master['test_name'].astype(str)
    key_map = (
        master[['model_test_key', 'model_sanitized', 'test_name']]
        .drop_duplicates()
        .set_index('model_test_key')[['model_sanitized','test_name']].to_dict('index')
    )
    return key_map

def _rename_metric_columns(cols: List[str], metric: str, key_map: Dict[str, Dict[str, str]]) -> List[str]:
    """Rename pivoted metric columns per spec shown in the screenshot."""
    new = []
    for c in cols:
        info = key_map.get(c)
        if not info:
            new.append(c)  # fallback
            continue
        model = info['model_sanitized']
        test  = str(info['test_name'])
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

def main():
    print("--- Creating Enriched Results (WIDE + ROW-WISE) ---")

    # Load all results
    frames = _load_results_frames(RESULTS_DIR)
    if not frames:
        print("[ABORT] No usable results.")
        return
    master = pd.concat(frames, ignore_index=True)

    # Key mapping and identifiers
    key_map = _build_keymap(master)

    # ---------------------------
    # Build WIDE (pivoted) table
    # ---------------------------
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

        # rename pivot columns per our spec
        orig_cols = list(pivot.columns[2:])
        renamed   = _rename_metric_columns(orig_cols, metric, key_map)
        pivot.columns = ['condition', 'item'] + renamed

        wide = pivot if wide is None else pd.merge(wide, pivot, on=['condition','item'], how='outer')

    if wide is None:
        print("[ERROR] No metrics found to pivot. Aborting.")
        return

    # Reorder columns to group by <model>_<test>: left, right, logOdds, test_passed
    print("Reordering WIDE columns by model/test group...")
    id_cols = ['condition', 'item']

    # keep order of first appearance for groups
    seen_keys = []
    for c in wide.columns:
        if '__' in c:
            key = c.split('__')[0]  # <model>_<test>
            if key not in seen_keys:
                seen_keys.append(key)

    # Build the ordered list
    ordered_cols = id_cols.copy()
    # Need a helper from key string to test name (use key_map inverse)
    # Build inverse: <model>_<test> -> test
    inverse_test = {}
    for k, v in key_map.items():
        inverse_test[f"{v['model_sanitized']}_{v['test_name']}"] = str(v['test_name'])

    for key in seen_keys:
        test = inverse_test.get(key, None)
        if test is None:
            # best-effort: derive test from tail of key after first underscore
            test = key.split('_', 1)[1] if '_' in key else key

        left   = f"{key}__LLH_test_{test}_1"
        right  = f"{key}__LLH_test_{test}_2"
        odds   = f"{key}__logOdds_test_{test}"
        passed = f"{key}__test_passed"
        for col in (left, right, odds, passed):
            if col in wide.columns:
                ordered_cols.append(col)

    wide = wide[ordered_cols]

    # -------------------------------
    # Load original dataset & merge
    # -------------------------------
    try:
        original = pd.read_csv(DATASET_PATH)
        print(f"\nLoaded original dataset: {DATASET_PATH}")
    except FileNotFoundError:
        print(f"[ERROR] Original dataset not found at '{DATASET_PATH}'. Aborting merge.")
        return

    # drop patching_prompts if present
    to_drop = [f'patching_prompt_{i}' for i in range(1,5)]
    existing_drop = [c for c in to_drop if c in original.columns]
    original_clean = original.drop(columns=existing_drop)
    if existing_drop:
        print(f"Removed columns from original: {existing_drop}")

    print("Merging original dataset with WIDE metrics...")
    enriched_wide = pd.merge(original_clean, wide, on=['condition', 'item'], how='left')

    # Save WIDE
    os.makedirs(RESULTS_DIR, exist_ok=True)
    enriched_wide.to_csv(WIDE_OUTPUT_PATH, index=False)
    print(f"SUCCESS: WIDE file written to: {WIDE_OUTPUT_PATH}")
    print(enriched_wide.head())

    # -------------------------------------
    # Build ROW-WISE (long) enriched table
    # -------------------------------------
    print("\nBuilding ROW-WISE table...")
    needed_cols = ['condition','item','model_source','test_name',
                   'llh_left','llh_right','logodds','test_passed']
    missing = [c for c in needed_cols if c not in master.columns]
    if missing:
        print(f"[ERROR] Cannot build row-wise table; missing columns: {missing}")
        return

    long_df = (
        master[needed_cols].copy()
        .rename(columns={
            'llh_left':  'LLH_test_1',
            'llh_right': 'LLH_test_2',
            'logodds':   'logOdds_test'
        })
    )

    print("Merging original dataset with ROW-WISE metrics...")
    enriched_long = pd.merge(original_clean, long_df, on=['condition','item'], how='left')

    enriched_long.to_csv(LONG_OUTPUT_PATH, index=False)
    print(f"SUCCESS: ROW-WISE file written to: {LONG_OUTPUT_PATH}")
    print(enriched_long.head())

if __name__ == "__main__":
    main()
