"""
Scaling and Training-Set Balancing Pipeline

Reads:
  data/split/train_data.csv
  data/split/validation_data.csv
  data/split/test_data.csv

Writes:
  data/scaled_balanced/scaled_train_data.csv
  data/scaled_balanced/scaled_validation_data.csv
  data/scaled_balanced/scaled_test_data.csv
  data/scaled_balanced/balanced_train_data.csv

Per project docs:
  1. Use only features available before the build runs/completes.
  2. Analyze categorical columns from the training set only.
  3. One-hot encode selected low-cardinality pre-build categoricals.
  4. Fit StandardScaler only on the training set.
  5. Transform validation and test using the training-fitted scaler.
  6. Apply SMOTE on the training set only.
  7. Do not balance validation or test sets.
  8. Do not train models in this phase.

Note:
  Build-log and duration columns are dropped because this project predicts
  failure before build completion. Keeping them would cause data leakage.

Usage
    python src/preprocessing/scale_balance.py
    python src/preprocessing/scale_balance.py --input-dir data/split --output-dir data/scaled_balanced
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = 'tr_status'
RANDOM_STATE = 42

DIRECT_DROP_COLUMNS = [
    # High-cardinality identifiers/text that can overfit or are not model-ready.
    'gh_project_name',
    'gh_pr_created_at',
    'gh_build_started_at',

    # Build-log/result fields are unavailable for true pre-build prediction.
    'tr_log_lan',
    'tr_log_status',
    'tr_log_analyzer',
    'tr_log_frameworks',
    'tr_log_bool_tests_failed',
    'tr_log_bool_tests_ran',
    'tr_log_num_tests_ok',
    'tr_log_num_tests_failed',
    'tr_log_num_tests_run',
    'tr_log_num_tests_skipped',
    'tr_log_testduration',
    'tr_duration',
    'test_fail_ratio',
]

SAFE_CATEGORICAL_COLUMNS = [
    'git_merged_with',
    'git_prev_commit_resolution_status',
    'branch_group',
]

RAW_CATEGORICAL_INPUT_COLUMNS = [
    'git_merged_with',
    'git_prev_commit_resolution_status',
    'git_branch',
]


def get_required_columns(path):
    """Read only columns needed for pre-build model preparation."""
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    direct_drop = set(DIRECT_DROP_COLUMNS)

    required_columns = [
        col for col in columns
        if col not in direct_drop
    ]

    missing_raw_categoricals = [
        col for col in RAW_CATEGORICAL_INPUT_COLUMNS
        if col not in required_columns and col in columns
    ]
    required_columns.extend(missing_raw_categoricals)

    return required_columns


def load_splits(input_dir):
    """Load only needed train, validation, and test split columns."""
    paths = {
        'train': os.path.join(input_dir, 'train_data.csv'),
        'validation': os.path.join(input_dir, 'validation_data.csv'),
        'test': os.path.join(input_dir, 'test_data.csv'),
    }

    splits = {}
    for name, path in paths.items():
        usecols = get_required_columns(path)
        splits[name] = pd.read_csv(path, usecols=usecols, low_memory=False)
        print(f'  Loaded {name:<10}: {splits[name].shape[0]:,} rows x {splits[name].shape[1]} columns')

    return splits


def branch_group(value):
    """Group high-cardinality branch names into stable pre-build categories."""
    branch = str(value).lower().strip()

    if branch in ('', 'nan', 'none', 'unknown'):
        return 'unknown'
    if branch in ('master', 'main', 'trunk'):
        return 'main'
    if branch in ('develop', 'dev', 'devel', 'development'):
        return 'develop'
    if 'release' in branch or branch.startswith('rel-'):
        return 'release'
    if 'hotfix' in branch or 'fix' in branch or 'bug' in branch:
        return 'fix'
    if 'feature' in branch or branch.startswith('feat'):
        return 'feature'

    return 'other'


def add_categorical_features(df):
    """Create safe categorical features before one-hot encoding."""
    df = df.copy()

    if 'git_branch' in df.columns:
        df['branch_group'] = df['git_branch'].map(branch_group)

    return df


def analyze_categoricals(train_df):
    """Summarize non-numeric columns using the training split only."""
    categorical_columns = train_df.select_dtypes(include=['object', 'string']).columns.tolist()
    rows = []

    for col in categorical_columns:
        top_values = train_df[col].value_counts(dropna=False).head(10)
        target_by_value = (
            train_df.assign(_category=train_df[col].fillna('unknown').astype(str))
            .groupby('_category')[TARGET_COLUMN]
            .agg(count='count', failure_rate='mean')
            .sort_values('count', ascending=False)
            .head(10)
        )

        rows.append({
            'column': col,
            'unique_values': int(train_df[col].nunique(dropna=True)),
            'missing_percent': round(train_df[col].isna().mean() * 100, 4),
            'top_values': top_values.to_dict(),
            'top_value_failure_rates': {
                str(index): {
                    'count': int(row['count']),
                    'failure_rate': round(float(row['failure_rate']), 4),
                }
                for index, row in target_by_value.iterrows()
            },
        })

    return rows


def fit_one_hot_categories(train_df, categorical_columns):
    """Fit one-hot category lists from the training split only."""
    categories = {}

    for col in categorical_columns:
        if col in train_df.columns:
            categories[col] = sorted(
                train_df[col]
                .fillna('unknown')
                .astype(str)
                .str.lower()
                .unique()
                .tolist()
            )

    return categories


def apply_one_hot_encoding(df, categories):
    """Apply train-fitted one-hot categories to a split."""
    df = df.copy()

    for col, values in categories.items():
        if col not in df.columns:
            continue

        normalized = df[col].fillna('unknown').astype(str).str.lower()
        for value in values:
            new_col = f'{col}__{value}'
            df[new_col] = (normalized == value).astype(int)

    return df


def get_model_columns(train_df):
    """Keep numeric features after leakage removal and selected categorical encoding."""
    if TARGET_COLUMN not in train_df.columns:
        raise KeyError(f'Missing required target column: {TARGET_COLUMN}')

    numeric_columns = train_df.select_dtypes(include='number').columns.tolist()
    feature_columns = [col for col in numeric_columns if col != TARGET_COLUMN]
    dropped_columns = [col for col in train_df.columns if col not in feature_columns + [TARGET_COLUMN]]

    return feature_columns, dropped_columns


def prepare_model_frames(splits):
    """Drop leakage columns, encode safe categoricals, and return numeric frames."""
    splits = {
        name: add_categorical_features(df)
        for name, df in splits.items()
    }

    categorical_analysis = analyze_categoricals(splits['train'])
    one_hot_categories = fit_one_hot_categories(splits['train'], SAFE_CATEGORICAL_COLUMNS)
    encoded_splits = {
        name: apply_one_hot_encoding(df, one_hot_categories)
        for name, df in splits.items()
    }

    leakage_columns = [
        col for col in DIRECT_DROP_COLUMNS + ['git_branch', 'branch_group']
        if col in encoded_splits['train'].columns
    ]

    model_splits = {}
    for name, df in encoded_splits.items():
        model_splits[name] = df.drop(columns=[col for col in leakage_columns if col in df.columns])

    feature_columns, dropped_columns = get_model_columns(model_splits['train'])

    model_splits = {
        name: df[feature_columns + [TARGET_COLUMN]].copy()
        for name, df in model_splits.items()
    }

    encoding_columns = [
        col for col in feature_columns
        if any(col.startswith(f'{cat_col}__') for cat_col in SAFE_CATEGORICAL_COLUMNS)
    ]

    return model_splits, feature_columns, dropped_columns, leakage_columns, categorical_analysis, one_hot_categories, encoding_columns


def clean_numeric_values(train_df, validation_df, test_df, feature_columns):
    """Replace infinite values and fill missing numeric values using train medians."""
    train_df = train_df.copy()
    validation_df = validation_df.copy()
    test_df = test_df.copy()

    for df in [train_df, validation_df, test_df]:
        df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    medians = train_df[feature_columns].median()
    medians = medians.fillna(0)

    train_df[feature_columns] = train_df[feature_columns].fillna(medians)
    validation_df[feature_columns] = validation_df[feature_columns].fillna(medians)
    test_df[feature_columns] = test_df[feature_columns].fillna(medians)

    return train_df, validation_df, test_df, medians


def scale_splits(train_df, validation_df, test_df, feature_columns):
    """Fit StandardScaler on train and transform all splits."""
    scaler = StandardScaler()

    scaled_train = train_df.copy()
    scaled_validation = validation_df.copy()
    scaled_test = test_df.copy()

    scaled_train[feature_columns] = scaler.fit_transform(train_df[feature_columns])
    scaled_validation[feature_columns] = scaler.transform(validation_df[feature_columns])
    scaled_test[feature_columns] = scaler.transform(test_df[feature_columns])

    return scaled_train, scaled_validation, scaled_test, scaler


def balance_training_data(scaled_train, feature_columns):
    """Apply SMOTE to the scaled training data only."""
    x_train = scaled_train[feature_columns]
    y_train = scaled_train[TARGET_COLUMN]

    smote = SMOTE(random_state=RANDOM_STATE)
    x_balanced, y_balanced = smote.fit_resample(x_train, y_train)

    balanced_train = pd.DataFrame(x_balanced, columns=feature_columns)
    balanced_train[TARGET_COLUMN] = y_balanced.astype(int)

    return balanced_train


def print_target_summary(name, df):
    """Print class distribution for a dataset."""
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    summary = ', '.join(
        f'{int(label)}={counts[label]:,} ({percentages[label]:.2f}%)'
        for label in counts.index
    )

    print(f'  {name:<18}: {len(df):>10,} rows | {summary}')


def save_outputs(
    scaled_train,
    scaled_validation,
    scaled_test,
    balanced_train,
    output_dir,
    scaler,
    medians,
    metadata,
):
    """Save scaled/balanced datasets and preprocessing artifacts."""
    os.makedirs(output_dir, exist_ok=True)

    scaled_train_path = os.path.join(output_dir, 'scaled_train_data.csv')
    scaled_validation_path = os.path.join(output_dir, 'scaled_validation_data.csv')
    scaled_test_path = os.path.join(output_dir, 'scaled_test_data.csv')
    balanced_train_path = os.path.join(output_dir, 'balanced_train_data.csv')
    scaler_path = os.path.join(output_dir, 'standard_scaler.joblib')
    medians_path = os.path.join(output_dir, 'train_numeric_medians.csv')
    metadata_path = os.path.join(output_dir, 'scale_balance_metadata.json')

    scaled_train.to_csv(scaled_train_path, index=False)
    scaled_validation.to_csv(scaled_validation_path, index=False)
    scaled_test.to_csv(scaled_test_path, index=False)
    balanced_train.to_csv(balanced_train_path, index=False)

    joblib.dump(scaler, scaler_path)
    medians.to_csv(medians_path, header=['median'])

    with open(metadata_path, 'w', encoding='utf-8') as file:
        json.dump(metadata, file, indent=2)

    print('\nSaved files:')
    print(f'  Scaled train      : {scaled_train_path}')
    print(f'  Scaled validation : {scaled_validation_path}')
    print(f'  Scaled test       : {scaled_test_path}')
    print(f'  Balanced train    : {balanced_train_path}')
    print(f'  Scaler            : {scaler_path}')
    print(f'  Train medians     : {medians_path}')
    print(f'  Metadata          : {metadata_path}')


def scale_and_balance(input_dir, output_dir):
    print(f'Input dir : {input_dir}')
    print(f'Output dir: {output_dir}\n')

    start = time.time()

    print('Loading split data ...')
    splits = load_splits(input_dir)

    print('\nAnalyzing and encoding train-fitted categorical features ...')
    (
        model_splits,
        feature_columns,
        dropped_columns,
        leakage_columns,
        categorical_analysis,
        one_hot_categories,
        encoding_columns,
    ) = prepare_model_frames(splits)
    print(f'  Train categorical columns analyzed : {len(categorical_analysis)}')
    print(f'  Safe categorical columns encoded   : {list(one_hot_categories.keys())}')
    print(f'  One-hot columns added              : {len(encoding_columns)}')
    print(f'  Columns excluded before loading    : {len(DIRECT_DROP_COLUMNS)}')
    print(f'  Leakage/raw columns dropped        : {len(leakage_columns)}')
    print(f'  Final model-ready features         : {len(feature_columns)}')
    if dropped_columns:
        print(f'  Remaining non-numeric dropped      : {dropped_columns}')

    print('\nCleaning numeric values ...')
    train_df, validation_df, test_df, medians = clean_numeric_values(
        model_splits['train'],
        model_splits['validation'],
        model_splits['test'],
        feature_columns,
    )
    print('  [+] Infinite values replaced with NaN')
    print('  [+] Missing numeric values filled with training medians')

    print('\nScaling data ...')
    scaled_train, scaled_validation, scaled_test, scaler = scale_splits(
        train_df,
        validation_df,
        test_df,
        feature_columns,
    )
    print('  [+] StandardScaler fitted on training set')
    print('  [+] Validation/test transformed with training-fitted scaler')

    print('\nBalancing training data ...')
    balanced_train = balance_training_data(scaled_train, feature_columns)
    print('  [+] SMOTE applied to training set only')

    print('\nTarget summary:')
    print_target_summary('Scaled train', scaled_train)
    print_target_summary('Balanced train', balanced_train)
    print_target_summary('Scaled validation', scaled_validation)
    print_target_summary('Scaled test', scaled_test)

    metadata = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'target_column': TARGET_COLUMN,
        'random_state': RANDOM_STATE,
        'feature_columns': feature_columns,
        'project_goal': 'predict CI/CD build outcome before build completion',
        'categorical_analysis_from': 'train_data.csv only',
        'categorical_analysis': categorical_analysis,
        'one_hot_categories_fit_on_train': one_hot_categories,
        'one_hot_encoded_columns': encoding_columns,
        'columns_excluded_before_loading': DIRECT_DROP_COLUMNS,
        'dropped_leakage_or_raw_columns': leakage_columns,
        'dropped_non_numeric_columns_after_encoding': dropped_columns,
        'scaler_fit_on': 'train_data.csv only',
        'smote_applied_to': 'scaled training set only',
    }

    save_outputs(
        scaled_train,
        scaled_validation,
        scaled_test,
        balanced_train,
        output_dir,
        scaler,
        medians,
        metadata,
    )

    elapsed = time.time() - start
    print(f'\nDone in {elapsed:.1f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scale split data and balance the training set.')
    parser.add_argument('--input-dir', default='data/split', help='Directory containing split CSV files')
    parser.add_argument('--output-dir', default='data/scaled_balanced', help='Directory for output CSV files')
    args = parser.parse_args()

    scale_and_balance(args.input_dir, args.output_dir)
