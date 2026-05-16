"""
Encoding Pipeline
===========================================================
Encodes categorical, target, and boolean values in the featured dataset.
Reads featured_data.csv -> writes encoded_data.csv

Per project docs:
  1. Encode target variable:
       passed -> 0
       failed/broken/errored -> 1
       canceled -> drop
  2. Label-encode gh_lang.
  3. Convert boolean columns to integer 0/1.

Usage
    python src/preprocessing/encode.py
    python src/preprocessing/encode.py --input data/featured_data.csv --output data/encoded_data.csv
"""

import argparse
import os
import time

import pandas as pd


TARGET_COLUMN = 'tr_status'
LANG_COLUMN = 'gh_lang'

TARGET_MAP = {
    'passed': 0,
    'failed': 1,
    'broken': 1,
    'errored': 1,
}


def encode_target(df):
    """Encode tr_status and drop canceled/unmapped rows."""
    before = len(df)
    status = df[TARGET_COLUMN].astype(str).str.lower().str.strip()
    keep_mask = status.isin(TARGET_MAP)

    df = df.loc[keep_mask].copy()
    df[TARGET_COLUMN] = status.loc[keep_mask].map(TARGET_MAP).astype(int)

    dropped = before - len(df)
    print('  [+] Target encoded: passed=0, failed/broken/errored=1')
    print(f'      Dropped canceled/unmapped rows: {dropped:,}')
    return df


def encode_language(df):
    """Label-encode gh_lang using sorted values from the current dataset."""
    languages = (
        df[LANG_COLUMN]
        .fillna('unknown')
        .astype(str)
        .str.lower()
        .sort_values()
        .unique()
    )
    language_map = {language: idx for idx, language in enumerate(languages)}

    df[LANG_COLUMN] = (
        df[LANG_COLUMN]
        .fillna('unknown')
        .astype(str)
        .str.lower()
        .map(language_map)
        .astype(int)
    )

    print(f'  [+] Label encoded gh_lang: {len(language_map)} categories')
    print(f'      Mapping: {language_map}')
    return df


def convert_booleans(df):
    """Convert boolean columns to integer 0/1."""
    bool_columns = df.select_dtypes(include='bool').columns.tolist()

    for col in bool_columns:
        df[col] = df[col].astype(int)

    print(f'  [+] Boolean columns converted to 0/1: {len(bool_columns)}')
    if bool_columns:
        print(f'      Columns: {bool_columns}')
    return df


def encode_data(input_path, output_path):
    print(f'Input  : {input_path}')
    print(f'Output : {output_path}\n')

    start = time.time()

    print('Loading featured data ...')
    df = pd.read_csv(input_path, low_memory=False)
    print(f'  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n')

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f'Missing required target column: {TARGET_COLUMN}')
    if LANG_COLUMN not in df.columns:
        raise KeyError(f'Missing required language column: {LANG_COLUMN}')

    print('Encoding data ...')
    df = encode_target(df)
    df = encode_language(df)
    df = convert_booleans(df)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start
    print(f'\nDone in {elapsed:.1f}s')
    print(f'  Final shape    : {df.shape[0]:,} rows x {df.shape[1]} columns')
    print(f'  Output saved to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode featured CI/CD data.')
    parser.add_argument('--input', default='data/featured_data.csv', help='Path to featured CSV')
    parser.add_argument('--output', default='data/encoded_data.csv', help='Path for encoded CSV')
    args = parser.parse_args()

    encode_data(args.input, args.output)
