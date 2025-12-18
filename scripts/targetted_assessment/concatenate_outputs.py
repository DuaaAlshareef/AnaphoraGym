# # #!/usr/bin/env python3
# # """
# # Concatenate AnaphoraGym model result files into one big CSV.

# # Matches files: results/AnaphoraGym_Results_*
# # Produces: all_results.csv (by default)

# # Usage examples:
# #   python concat_anaphora_results.py
# #   python concat_anaphora_results.py --results-folder my_results --output big.csv
# #   python concat_anaphora_results.py --results-folder results --prefix AnaphoraGym_Results_ --output all_results.csv
# # """
# # from __future__ import annotations
# # import os
# # import glob
# # import argparse
# # import sys
# # from typing import Optional
# # import pandas as pd

# # PREFIX_DEFAULT = "AnaphoraGym_Results_"

# # def try_read_csv(fname: str, **kwargs) -> pd.DataFrame:
# #     """Try multiple ways to read a CSV-like file, returning a DataFrame or raising."""
# #     # Try auto-detect sep via engine='python' and sep=None
# #     try:
# #         return pd.read_csv(fname, sep=None, engine="python", **kwargs)
# #     except TypeError:
# #         # older pandas might not accept some kwargs - fall back
# #         return pd.read_csv(fname, sep=None, engine="python")
# #     except Exception:
# #         # fallback to plain read_csv (comma)
# #         try:
# #             return pd.read_csv(fname, **kwargs)
# #         except Exception as e:
# #             raise

# # def read_file_to_df(fname: str, prefix: str) -> Optional[pd.DataFrame]:
# #     """Read a file into a DataFrame, add metadata columns; return None on fatal read error."""
# #     ext = os.path.splitext(fname)[1].lower()
# #     df = None
# #     try:
# #         if ext in (".csv", ".tsv", ".txt", ""):
# #             # common case: try CSV/TSV auto-detect
# #             df = try_read_csv(fname, encoding="utf-8", on_bad_lines="skip")
# #         elif ext in (".json", ".jsonl"):
# #             # JSON lines or JSON array
# #             try:
# #                 df = pd.read_json(fname, lines=True)
# #             except ValueError:
# #                 df = pd.read_json(fname)
# #         elif ext in (".xls", ".xlsx"):
# #             df = pd.read_excel(fname)
# #         elif ext.endswith(".gz"):
# #             # gzipped CSV
# #             try:
# #                 df = pd.read_csv(fname, compression="gzip", sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
# #             except Exception:
# #                 df = pd.read_csv(fname, compression="gzip")
# #         else:
# #             # Try CSV read first, then fallback to a single-text-column DataFrame
# #             try:
# #                 df = try_read_csv(fname, encoding="utf-8", on_bad_lines="skip")
# #             except Exception:
# #                 with open(fname, "r", encoding="utf-8", errors="replace") as f:
# #                     txt = f.read()
# #                 df = pd.DataFrame({"text": [txt]})
# #     except Exception as e:
# #         print(f"[WARN] Could not parse {fname} as structured data: {e}", file=sys.stderr)
# #         # fallback: load full file as single text column
# #         try:
# #             with open(fname, "r", encoding="utf-8", errors="replace") as f:
# #                 txt = f.read()
# #             df = pd.DataFrame({"text": [txt]})
# #         except Exception as e2:
# #             print(f"[ERROR] Failed to read {fname} even as text: {e2}", file=sys.stderr)
# #             return None

# #     # Defensive: drop rows that are accidental repeated header rows
# #     try:
# #         header_strs = tuple(str(c) for c in df.columns)
# #         # create array of rows as tuples of strings
# #         arr = df.astype(str).values
# #         repeated_header_mask = [tuple(row) == header_strs for row in arr]
# #         if any(repeated_header_mask):
# #             df = df.loc[[not m for m in repeated_header_mask]].reset_index(drop=True)
# #     except Exception:
# #         # if anything goes wrong, ignore and continue
# #         pass

# #     # Add metadata columns
# #     df["source_file"] = os.path.basename(fname)
# #     # model name: remove prefix from filename (without extension)
# #     basename_no_ext = os.path.splitext(os.path.basename(fname))[0]
# #     model = basename_no_ext.replace(prefix, "", 1)
# #     df["model"] = model

# #     return df

# # def find_files(results_folder: str, prefix: str) -> list[str]:
# #     pattern = os.path.join(results_folder, f"{prefix}*")
# #     matches = sorted(glob.glob(pattern))
# #     # ensure they are files (skip directories)
# #     matches = [m for m in matches if os.path.isfile(m)]
# #     return matches

# # def main():
# #     p = argparse.ArgumentParser(description="Concatenate AnaphoraGym_Results_* files into one CSV.")
# #     p.add_argument("--results-folder", "-r", default="results", help="Folder containing results (default: results)")
# #     p.add_argument("--prefix", "-p", default=PREFIX_DEFAULT, help=f"Filename prefix to match (default: {PREFIX_DEFAULT})")
# #     p.add_argument("--output", "-o", default="all_results.csv", help="Output CSV file (default: all_results.csv)")
# #     p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
# #     args = p.parse_args()

# #     files = find_files(args.results_folder, args.prefix)
# #     if not files:
# #         print(f"[ERROR] No files found matching {os.path.join(args.results_folder, args.prefix+'*')}", file=sys.stderr)
# #         sys.exit(2)

# #     if args.verbose:
# #         print(f"[INFO] Found {len(files)} matching files. They will be read in alphabetical order:")
# #         for f in files:
# #             print("  -", f)

# #     dfs = []
# #     for f in files:
# #         if args.verbose:
# #             print(f"[INFO] Reading {f} ...")
# #         df = read_file_to_df(f, args.prefix)
# #         if df is None:
# #             print(f"[WARN] Skipping {f} due to read errors.", file=sys.stderr)
# #             continue
# #         dfs.append(df)

# #     if not dfs:
# #         print("[ERROR] No files were successfully parsed. Exiting.", file=sys.stderr)
# #         sys.exit(3)

# #     # Concatenate with union of columns (columns that don't exist in some files become NaN)
# #     big = pd.concat(dfs, ignore_index=True, sort=False)

# #     # If output should be gzipped based on extension
# #     output_path = args.output
# #     compress = None
# #     if output_path.lower().endswith(".gz"):
# #         compress = "gzip"

# #     # Save CSV
# #     big.to_csv(output_path, index=False, compression=compress)
# #     print(f"✅ Concatenation complete: wrote {len(big)} rows from {len(dfs)} files -> {output_path}")
# #     print(f"Columns in final file: {', '.join(list(big.columns))}")

# # if __name__ == "__main__":
# #     main()


# #!/usr/bin/env python3
# """
# Consolidate ALL result CSVs under results/ into a single row-wise CSV that contains:
#   model, condition, test, llh_left, llh_right, log_odds, test_passed

# Behavior:
#  - Recursively finds *.csv under RESULTS_DIR
#  - Normalizes common column name variants
#  - If model_source is missing, infers model name from filename
#  - Keeps test_passed as numeric 0/1 (never True/False)
#  - Logs included and skipped files (with reasons)
# """

# import os
# import glob
# import re
# import pandas as pd
# from typing import List

# # -------- CONFIG ----------
# try:
#     SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#     PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
# except NameError:
#     PROJECT_ROOT = os.path.abspath('.')

# RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
# OUT_ROWWISE = os.path.join(RESULTS_DIR, "AnaphoraGym_Consolidated_Rowwise_ALL.csv")
# OUT_WIDE = os.path.join(RESULTS_DIR, "AnaphoraGym_Consolidated_WIDE_ALL.csv")
# # --------------------------

# # Known alternate column names -> canonical names used in this script
# COL_CANON = {
#     'LLH_left': 'llh_left',
#     'LLH_right': 'llh_right',
#     'LLH-left': 'llh_left',
#     'LLH_right ': 'llh_right',
#     'llhLeft': 'llh_left',
#     'llh_right': 'llh_right',
#     'logOdds': 'logodds',
#     'LogOdds': 'logodds',
#     'log_odds': 'logodds',
#     'log-odds': 'logodds',
#     'test_passed': 'test_passed',
#     'passed': 'test_passed',
#     'passed?': 'test_passed',
#     'model_source': 'model_source',
#     'model': 'model_source',
#     'test_name': 'test_name',
#     'test': 'test_name',
#     'condition': 'condition',
#     'item': 'item',
# }

# # Columns we want in the final rowwise output (in this order)
# ROW_COLS = ['model', 'condition', 'test', 'llh_left', 'llh_right', 'log_odds', 'test_passed']


# def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
#     """Normalize known column variants to canonical names (lowercase)."""
#     rename_map = {}
#     for c in df.columns:
#         c_strip = c.strip()
#         # prefer direct mapping on exact key
#         if c_strip in COL_CANON:
#             rename_map[c] = COL_CANON[c_strip]
#         else:
#             # try case-insensitive match
#             low = c_strip.lower()
#             for k, v in COL_CANON.items():
#                 if k.lower() == low:
#                     rename_map[c] = v
#                     break
#     return df.rename(columns=rename_map)


# def infer_model_from_filename(path: str) -> str:
#     """
#     Best-effort model inference from filename.
#     Examples it will catch:
#       - AnaphoraGym_Results_gpt2.csv -> gpt2
#       - results_gpt2-large.csv -> gpt2-large
#       - model=gpt2;other.csv -> gpt2
#       - pythia-410m -> pythia-410m
#     """
#     name = os.path.basename(path)
#     # remove extension
#     name_noext = os.path.splitext(name)[0]
#     # common patterns: *_<modelname> or <modelname>_results
#     # heuristics: look for tokens that look like model names (letters, digits, '-', '/', '_')
#     # prefer tokens after "results" or last token
#     # Try to extract token like "gpt2", "gpt2-medium", "pythia-410m", "meta-llama/Llama-2-7b-hf"
#     # Replace separators with single space
#     tokens = re.split(r'[_\-\s]+', name_noext)
#     # common prefixes to ignore
#     ignore = {'anaphoragym', 'results', 'result', 'run', 'eval', 'targetted', 'targeted', 'assessment'}
#     candidates = [t for t in tokens if t and t.lower() not in ignore]
#     if candidates:
#         # pick the last candidate often contains the model
#         cand = candidates[-1]
#         # try to sanitize: if it contains 'meta' or 'EleutherAI' with slash, convert / to _
#         cand = cand.replace('/', '_')
#         return cand
#     # fallback to entire filename sanitized
#     return re.sub(r'[^0-9A-Za-z_\-]+', '_', name_noext)


# def coerce_test_passed_to_01(s: pd.Series) -> pd.Series:
#     """Force test_passed to numeric 0/1 (never bool)."""
#     # handle bool dtype
#     if pd.api.types.is_bool_dtype(s):
#         return s.astype(int)
#     # map common string variants
#     s2 = s.astype(str).str.strip().str.lower()
#     mapped = s2.map({'true': 1, 'false': 0, '1': 1, '0': 0, 'yes': 1, 'no': 0})
#     # where mapping failed, try numeric coercion
#     mapped = mapped.fillna(pd.to_numeric(s, errors='coerce'))
#     # final fallback: NaN -> 0 (or you can choose to keep NaN)
#     mapped = mapped.fillna(0)
#     return mapped.astype('int64')


# def read_and_standardize(fp: str):
#     """
#     Read CSV at fp, normalize columns, try to ensure required base columns are present.
#     Returns: (df_or_None, reason_message_or_None)
#     """
#     try:
#         df = pd.read_csv(fp)
#     except Exception as e:
#         return None, f"read_error: {e}"

#     df = normalize_colnames(df)

#     # if model_source missing, infer from filename
#     if 'model_source' not in df.columns:
#         inferred = infer_model_from_filename(fp)
#         df['model_source'] = inferred

#     # if test_name missing, try to infer from filename token prior to model or similar
#     if 'test_name' not in df.columns:
#         # attempt to extract test token from filename
#         # common pattern: AnaphoraGym_Results_<model>_<test>.csv
#         name_noext = os.path.splitext(os.path.basename(fp))[0]
#         parts = name_noext.split('_')
#         # if it ends with at least two tokens, last token may be test
#         if len(parts) >= 3:
#             # assume last token is test if the token doesn't look like 'results' or model
#             candidate = parts[-1]
#             if candidate.lower() not in {'results', 'result', 'eval', 'run'}:
#                 df['test_name'] = candidate
#         # else leave missing (we'll still include rows with NaN test_name)

#     # Standardize metric column names to our desired final names
#     # Accept either llh_left / LLH_left etc.
#     # rename if present
#     # already handled some via normalize_colnames; final alias:
#     if 'logodds' in df.columns:
#         df = df.rename(columns={'logodds': 'log_odds'})
#     elif 'log_odds' in df.columns:
#         pass
#     elif 'logOdds' in df.columns:
#         df = df.rename(columns={'logOdds': 'log_odds'})

#     # coerce test_passed if present
#     if 'test_passed' in df.columns:
#         df['test_passed'] = coerce_test_passed_to_01(df['test_passed'])

#     # ensure lowercased llh columns if present
#     # allow both LLH_left or llh_left names already normalized above
#     if 'llh_left' not in df.columns and 'LLH_left' in df.columns:
#         df = df.rename(columns={'LLH_left': 'llh_left'})
#     if 'llh_right' not in df.columns and 'LLH_right' in df.columns:
#         df = df.rename(columns={'LLH_right': 'llh_right'})

#     return df, None


# def main():
#     print("=============================================================")
#     print("=> Creating the consolidated, enriched dataset CSV...")
#     print("=============================================================")

#     pattern = os.path.join(RESULTS_DIR, "**", "*.csv")
#     files = glob.glob(pattern, recursive=True)
#     if not files:
#         print(f"[ERROR] No CSV files located under: {RESULTS_DIR}")
#         return

#     included = []
#     skipped = []
#     frames = []

#     for fp in sorted(files):
#         # skip output files if they already exist in the results dir
#         if os.path.abspath(fp) in {os.path.abspath(OUT_ROWWISE), os.path.abspath(OUT_WIDE)}:
#             continue

#         df, err = read_and_standardize(fp)
#         if df is None:
#             skipped.append((fp, err))
#             print(f"[SKIP] {fp} -> {err}")
#             continue

#         # We accept files even if some metrics are missing; but require at least condition or item
#         if 'condition' not in df.columns and 'item' not in df.columns:
#             skipped.append((fp, "missing condition/item"))
#             print(f"[SKIP] {fp} -> missing condition/item")
#             continue

#         # Fill 'condition' or 'item' if one missing (not ideal but keep rows)
#         if 'condition' not in df.columns and 'item' in df.columns:
#             df['condition'] = pd.NA
#         if 'item' not in df.columns and 'condition' in df.columns:
#             df['item'] = pd.NA

#         # Ensure core columns exist (create if missing)
#         for c in ['model_source', 'test_name', 'llh_left', 'llh_right', 'log_odds', 'test_passed', 'condition', 'item']:
#             if c not in df.columns:
#                 df[c] = pd.NA

#         # Append the standardized frame
#         frames.append(df)
#         included.append(fp)
#         print(f"[INCL] {fp} (model: {df['model_source'].iat[0] if len(df)>0 else 'unknown'})")

#     if not frames:
#         print("[ERROR] No usable frames after scanning files.")
#         print("Files skipped:", skipped)
#         return

#     master = pd.concat(frames, ignore_index=True)

#     # Build row-wise output with requested columns and names
#     out = pd.DataFrame()
#     out['model'] = master['model_source']
#     # use 'condition' as asked — keep it, even if NaN
#     out['condition'] = master['condition']
#     # test (test_name)
#     out['test'] = master['test_name']
#     out['llh_left'] = master['llh_left']
#     out['llh_right'] = master['llh_right']
#     out['log_odds'] = master['log_odds']
#     # test_passed: ensure numeric 0/1 (if NaN -> leave as NaN or set 0 — we set 0 for safety)
#     out['test_passed'] = master['test_passed'].fillna(0).astype('int64')

#     # Reorder columns
#     out = out[['model', 'condition', 'test', 'llh_left', 'llh_right', 'log_odds', 'test_passed']]

#     # Optional: sort for readability
#     out = out.sort_values(by=['model', 'test', 'condition'], na_position='last').reset_index(drop=True)

#     # Write row-wise consolidated CSV
#     os.makedirs(RESULTS_DIR, exist_ok=True)
#     out.to_csv(OUT_ROWWISE, index=False)
#     print(f"\nSUCCESS: consolidated row-wise file written to:\n  {OUT_ROWWISE}")
#     print(f"Included files: {len(included)}; Skipped files: {len(skipped)}")
#     if skipped:
#         print("Sample skipped file reasons (up to 5):")
#         for s in skipped[:5]:
#             print("  ", s)

#     # -------------------------
#     # OPTIONAL: produce WIDE
#     # -------------------------
#     # If you want a WIDE table, you can uncomment the block below. It will pivot
#     # each metric per model_test_key (model + test) into columns.
#     #
#     # NOTE: wide output can be huge depending on number of models/tests.
#     #
#     do_wide = True
#     if do_wide:
#         print("\nBuilding optional WIDE table (can be large)...")
#         # ensure we have a model_test_key
#         master['model_sanitized'] = master['model_source'].astype(str).str.replace('/', '_', regex=False)
#         master['model_test_key'] = master['model_sanitized'] + "_" + master['test_name'].astype(str)

#         metrics = {
#             'llh_left': 'llh_left',
#             'llh_right': 'llh_right',
#             'log_odds': 'log_odds',
#             'test_passed': 'test_passed'
#         }

#         wide = None
#         for metric_col, outname in metrics.items():
#             if metric_col not in master.columns:
#                 print(f"  [INFO] metric '{metric_col}' not present -> skipping in wide pivot")
#                 continue
#             pivot = master.pivot_table(index=['condition', 'item'],
#                                        columns='model_test_key',
#                                        values=metric_col,
#                                        aggfunc='first').reset_index()
#             # flatten columns: change each metric column name to "<model>_<test>_<metric>"
#             new_cols = []
#             for c in pivot.columns[2:]:
#                 # c is model_test_key
#                 # build suffix from metric_col name
#                 suffix = outname
#                 # rename pattern: <model_test_key>_<suffix>
#                 new_cols.append(f"{c}_{suffix}")
#             pivot.columns = ['condition', 'item'] + new_cols
#             wide = pivot if wide is None else pd.merge(wide, pivot, on=['condition', 'item'], how='outer')

#         if wide is not None:
#             # write wide
#             wide_path = OUT_WIDE
#             wide.to_csv(wide_path, index=False)
#             print(f"SUCCESS: optional WIDE file written to:\n  {wide_path}")
#         else:
#             print("No wide output created (no metrics found).")

#     # Done
#     return


# if __name__ == "__main__":
#     main()




import pandas as pd
import glob
import os

# --- 1. DEFINE PROJECT PATHS ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
except NameError:
    # Fallback for environments where __file__ is not defined (e.g., interactive Python)
    print("Warning: __file__ not defined, defaulting PROJECT_ROOT to current directory.")
    PROJECT_ROOT = os.path.abspath('.')

# Define the directory where the individual model results are located
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'targetted_assessment')
# Define the name for the final concatenated CSV file
OUTPUT_FILENAME = "AnaphoraGym_All_Model_Results_Concatenated.csv"

def concatenate_model_results():
    print(f"--- Starting concatenation of individual model results ---")
    print(f"Searching for result files in: {RESULTS_DIR}")

    # --- Step 1: Find all individual model result files ---
    # We use a more specific pattern to only target files that are original model results
    # and exclude the 'Enriched' or 'Summary' files that are also in the directory.
    # The pattern explicitly looks for 'AnaphoraGym_Results_' followed by a model name.
    search_pattern = os.path.join(RESULTS_DIR, "AnaphoraGym_Results_*.csv")
    
    # Get all files matching the pattern
    all_files = glob.glob(search_pattern)

    # Filter out any files that are themselves aggregated results (e.g., 'Enriched', 'Rowwise')
    # This list should ideally only contain the raw results from individual models
    model_result_files = [
        f for f in all_files
        if "Enriched" not in os.path.basename(f) and "summary" not in os.path.basename(f)
    ]

    if not model_result_files:
        print(f"\nAnalysis failed: No individual model result files found in '{RESULTS_DIR}'.")
        print("Please ensure your 'AnaphoraGym_Results_modelname.csv' files are present.")
        return

    print(f"\nFound {len(model_result_files)} individual model result files to concatenate:")
    
    # --- Step 2 & 3: Read and Consolidate All Result Files ---
    all_results_dfs = []
    for filepath in model_result_files:
        filename = os.path.basename(filepath)
        print(f"  - Reading {filename}")
        try:
            df = pd.read_csv(filepath)
            all_results_dfs.append(df)
        except Exception as e:
            print(f"    [ERROR] Could not read {filename}. Reason: {e}")
            continue # Skip to the next file if there's an error

    if not all_results_dfs:
        print("\n[ERROR] No dataframes were successfully loaded. Exiting.")
        return

    # Concatenate all individual DataFrames into one big DataFrame
    concatenated_df = pd.concat(all_results_dfs, ignore_index=True)
    print(f"\nSuccessfully consolidated {len(concatenated_df)} rows from all files.")

    # --- Step 4: Save the Final Concatenated CSV ---
    output_path = os.path.join(RESULTS_DIR, OUTPUT_FILENAME)
    
    try:
        concatenated_df.to_csv(output_path, index=False)
        print(f"\nSUCCESS: All individual model results concatenated into '{output_path}'")
        print("\nHere's a preview of the new file structure (first 5 rows):")
        print(concatenated_df.head())
        print(f"\nTotal unique models in concatenated file: {concatenated_df['model_source'].nunique()}")
        
    except Exception as e:
        print(f"\n[ERROR] Could not save the concatenated results. Reason: {e}")

if __name__ == "__main__":
    concatenate_model_results()