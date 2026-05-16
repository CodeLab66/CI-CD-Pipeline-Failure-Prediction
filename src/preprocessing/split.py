"""
Dataset Splitting Pipeline
===========================================================
Splits the encoded dataset into train, validation, and test sets.
Reads encoded_data.csv -> writes train_data.csv, validation_data.csv, test_data.csv

Per project docs:
  1. Use a 70% training, 15% validation, 15% held-out test split.
  2. Preserve target class distribution with stratified splitting.
  3. Do not apply scaling, SMOTE, or model training in this phase.

Usage
    python src/preprocessing/split.py
    python src/preprocessing/split.py --input data/encoded_data.csv --output-dir data/split
"""

import argparse
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = 'tr_status'
RANDOM_STATE = 42


def split_dataset(df, train_size=0.70, val_size=0.15, test_size=0.15):
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError('train_size + val_size + test_size must equal 1.0')

    stratify = df[TARGET_COLUMN] if df[TARGET_COLUMN].nunique() > 1 else None

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    temp_stratify = temp_df[TARGET_COLUMN] if temp_df[TARGET_COLUMN].nunique() > 1 else None
    relative_test_size = test_size / (val_size + test_size)

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_stratify,
    )

    return train_df, validation_df, test_df


def print_split_summary(name, df):
    """Print row count and target distribution for a split."""
    target_counts = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    target_summary = ', '.join(
        f'{int(label)}={percent:.2f}%'
        for label, percent in target_counts.items()
    )

    print(f'  {name:<10}: {len(df):>10,} rows | target: {target_summary}')


def save_splits(train_df, validation_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train_data.csv')
    validation_path = os.path.join(output_dir, 'validation_data.csv')
    test_path = os.path.join(output_dir, 'test_data.csv')

    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f'\nSaved files:')
    print(f'  Train      : {train_path}')
    print(f'  Validation : {validation_path}')
    print(f'  Test       : {test_path}')


def split_data(input_path, output_dir):
    print(f'Input     : {input_path}')
    print(f'Output dir: {output_dir}\n')

    start = time.time()

    print('Loading encoded data ...')
    df = pd.read_csv(input_path, low_memory=False)
    print(f'  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n')

    if TARGET_COLUMN not in df.columns:
        raise KeyError(f'Missing required target column: {TARGET_COLUMN}')

    print('Splitting data ...')
    train_df, validation_df, test_df = split_dataset(df)

    print('\nSplit summary:')
    print_split_summary('Train', train_df)
    print_split_summary('Validation', validation_df)
    print_split_summary('Test', test_df)

    save_splits(train_df, validation_df, test_df, output_dir)

    elapsed = time.time() - start
    print(f'\nDone in {elapsed:.1f}s')
    print(f'  Total rows kept: {len(train_df) + len(validation_df) + len(test_df):,}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split encoded CI/CD data.')
    parser.add_argument('--input', default='data/encoded_data.csv', help='Path to encoded CSV')
    parser.add_argument('--output-dir', default='data/split', help='Directory for split CSV files')
    args = parser.parse_args()

    split_data(args.input, args.output_dir)
