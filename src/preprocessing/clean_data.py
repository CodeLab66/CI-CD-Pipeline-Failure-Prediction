"""
Data Cleaning Pipeline
===========================================================
  1. Drop useless columns (>= 90% missing, constant, ID-like, list/text)
  2. Impute remaining missing values
  3. Remove duplicate rows

Usage
    python src/preprocessing/clean_data.py
    python src/preprocessing/clean_data.py --input data/data.csv --output data/cleaned_data.csv
"""

import pandas as pd
import numpy as np
import os
import time
import argparse


# ────────────────────────────────────────────────────────────────────
#  COLUMNS TO DROP (from EDA analysis)
# ────────────────────────────────────────────────────────────────────

# >= 90% missing or constant
COLS_MISSING_OR_CONSTANT = [
    'gh_first_commit_created_at',    # 100% missing
    'gh_commits_in_push',            # 100% missing
    'gh_num_commits_in_push',        # 100% missing
    'gh_pushed_at',                  # 100% missing
    'tr_log_setup_time',             # ~100% missing
    'tr_log_num_test_suites_run',    # ~100% missing
    'tr_log_num_test_suites_ok',     # ~100% missing
    'tr_log_num_test_suites_failed', # ~100% missing
    'tr_log_buildduration',          # ~99% missing
    'tr_log_tests_failed',           # ~97% missing + free-text
    'gh_description_complexity',     # very high missing
]

# ID-like columns (unique ratio >= 95%)
COLS_ID_LIKE = [
    'tr_build_id',
    'tr_job_id',
    'git_trigger_commit',
    'git_prev_built_commit',
    'tr_original_commit',
    'tr_virtual_merged_into',
]

# List / free-text columns
COLS_LIST_TEXT = [
    'tr_jobs',
    'git_all_built_commits',
]

COLS_TO_DROP = list(set(
    COLS_MISSING_OR_CONSTANT + COLS_ID_LIKE + COLS_LIST_TEXT
))


# ────────────────────────────────────────────────────────────────────
#  PIPELINE FUNCTIONS
# ────────────────────────────────────────────────────────────────────

def drop_columns(df):
    """Drop columns flagged in EDA (missing, constant, ID-like, list/text)."""
    existing = [c for c in COLS_TO_DROP if c in df.columns]
    return df.drop(columns=existing)


def impute_missing(df):
    """Fill remaining nulls.
       - Numeric  → 0
       - Boolean  → False
       - Object   → 'unknown'
    """
    for col in df.select_dtypes(include='number').columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    for col in df.select_dtypes(include='bool').columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(False)

    for col in df.select_dtypes(include='object').columns:
        if df[col].isna().any():
            df[col] = df[col].fillna('unknown')

    return df


def clean_chunk(chunk):
    """Run all cleaning steps on a single chunk."""
    chunk = drop_columns(chunk)
    chunk = impute_missing(chunk)
    chunk = chunk.drop_duplicates()
    return chunk


# ────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────

def clean_data(input_path, output_path, chunk_size=250_000):
    """Process raw CSV in chunks and write cleaned CSV."""
    print(f'Input  : {input_path}')
    print(f'Output : {output_path}')
    print(f'Chunk  : {chunk_size:,} rows\n')

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    total_raw   = 0
    total_clean = 0
    first_chunk = True
    start       = time.time()

    for chunk in pd.read_csv(input_path, chunksize=chunk_size, low_memory=False):
        total_raw += len(chunk)

        chunk = clean_chunk(chunk)
        total_clean += len(chunk)

        chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False,
        )
        first_chunk = False

        elapsed = time.time() - start
        print(
            f'  Processed {total_raw:>10,} raw rows  |  '
            f'Kept {total_clean:>10,}  |  '
            f'{elapsed:.1f}s',
            flush=True,
        )

    elapsed = time.time() - start
    print(f'\nDone in {elapsed:.1f}s')
    print(f'  Raw rows read  : {total_raw:,}')
    print(f'  Clean rows kept: {total_clean:,}')
    print(f'  Rows removed   : {total_raw - total_clean:,}')
    print(f'  Columns dropped: {len(COLS_TO_DROP)}')
    print(f'  Output         : {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean raw TravisTorrent data.')
    parser.add_argument('--input',  default='data/data.csv',         help='Path to raw CSV')
    parser.add_argument('--output', default='data/cleaned_data.csv', help='Path for cleaned CSV')
    parser.add_argument('--chunk',  default=250_000, type=int,       help='Chunk size')
    args = parser.parse_args()

    clean_data(args.input, args.output, args.chunk)
