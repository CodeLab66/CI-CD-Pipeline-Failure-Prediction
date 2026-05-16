"""
Feature Engineering Pipeline
=================================================================
Creates new derived features from the cleaned dataset.
Reads cleaned_data.csv → writes featured_data.csv

New features created:
  Time-based   : hour_of_day, day_of_week, is_weekend
  Commit-based : total_code_churn, test_to_src_ratio, is_large_commit, total_files_changed
  Project-level: project_failure_rate (historical rolling failure rate per project)

Usage
    python src/preprocessing/feature_engineering.py
    python src/preprocessing/feature_engineering.py --input data/cleaned_data.csv --output data/featured_data.csv
"""

import pandas as pd
import numpy as np
import os
import time
import argparse



def add_time_features(df):

    dt = pd.to_datetime(df['gh_build_started_at'], errors='coerce')

    df['hour_of_day'] = dt.dt.hour.fillna(0).astype(int)
    df['day_of_week'] = dt.dt.dayofweek.fillna(0).astype(int)
    df['is_weekend']  = df['day_of_week'].isin([5, 6])

    print(f'  [+] Time features: hour_of_day, day_of_week, is_weekend')
    return df


def add_commit_features(df):
    src  = df['git_diff_src_churn'].fillna(0)
    test = df['git_diff_test_churn'].fillna(0)

    df['total_code_churn']    = src + test
    df['test_to_src_ratio']   = np.where(src > 0, test / src, 0.0)
    df['total_files_changed'] = (
        df['gh_diff_files_added'].fillna(0) +
        df['gh_diff_files_deleted'].fillna(0) +
        df['gh_diff_files_modified'].fillna(0)
    )

    threshold = df['total_code_churn'].quantile(0.75)
    df['is_large_commit'] = df['total_code_churn'] > threshold

    print(f'  [+] Commit features: total_code_churn, test_to_src_ratio, '
          f'total_files_changed, is_large_commit (threshold={threshold})')
    return df


def add_test_features(df):
    tests_run    = df['tr_log_num_tests_run'].fillna(0)
    tests_failed = df['tr_log_num_tests_failed'].fillna(0)

    df['test_fail_ratio'] = np.where(tests_run > 0, tests_failed / tests_run, 0.0)

    print(f'  [+] Test features: test_fail_ratio')
    return df


def add_project_failure_rate(df):
    df['_is_failure'] = df['tr_status'].isin(['failed', 'errored']).astype(int)

    # Save original order, sort for rolling calc, then restore
    df['_orig_order'] = np.arange(len(df))
    df = df.sort_values(['gh_project_name', 'tr_build_number'])

    df['project_failure_rate'] = (
        df.groupby('gh_project_name')['_is_failure']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    df['project_failure_rate'] = df['project_failure_rate'].fillna(0.0)

    # Restore original row order
    df = df.sort_values('_orig_order').reset_index(drop=True)
    df.drop(columns=['_is_failure', '_orig_order'], inplace=True)

    print(f'  [+] Project features: project_failure_rate (rolling, no leakage)')
    return df



def engineer_features(input_path, output_path):
    print(f'Input  : {input_path}')
    print(f'Output : {output_path}\n')

    start = time.time()

    print('Loading cleaned data ...')
    df = pd.read_csv(input_path, low_memory=False)
    print(f'  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n')

    print('Creating features ...')
    df = add_time_features(df)
    df = add_commit_features(df)
    df = add_test_features(df)
    df = add_project_failure_rate(df)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start
    new_cols = ['hour_of_day', 'day_of_week', 'is_weekend',
                'total_code_churn', 'test_to_src_ratio',
                'total_files_changed', 'is_large_commit',
                'test_fail_ratio', 'project_failure_rate']

    print(f'\nDone in {elapsed:.1f}s')
    print(f'  Final shape    : {df.shape[0]:,} rows x {df.shape[1]} columns')
    print(f'  New features   : {len(new_cols)}')
    print(f'  Feature list   : {new_cols}')
    print(f'  Output saved to: {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature engineering for CI/CD prediction.')
    parser.add_argument('--input',  default='data/cleaned_data.csv',  help='Path to cleaned CSV')
    parser.add_argument('--output', default='data/featured_data.csv', help='Path for output CSV')
    args = parser.parse_args()

    engineer_features(args.input, args.output)
