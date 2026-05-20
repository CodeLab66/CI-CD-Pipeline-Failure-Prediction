"""
Train CI/CD failure prediction models directly from the raw TravisTorrent CSV.

This script is designed for large files. It samples rows in chunks, keeps only
pre-build features, builds a reusable scikit-learn preprocessing pipeline, and
saves the best model plus evaluation reports.

Usage:
    python src/training/train_from_raw.py
    python src/training/train_from_raw.py --input data/data.csv --max-rows 200000
"""

import argparse
import json
import os
import time
from importlib.util import find_spec

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = 'tr_status'
RANDOM_STATE = 42

TARGET_MAP = {
    'passed': 0,
    'failed': 1,
    'errored': 1,
    'broken': 1,
}

RAW_COLUMNS = [
    TARGET_COLUMN,
    'gh_is_pr',
    'gh_pull_req_num',
    'gh_lang',
    'git_merged_with',
    'git_branch',
    'git_prev_commit_resolution_status',
    'tr_prev_build',
    'gh_team_size',
    'git_num_all_built_commits',
    'gh_num_issue_comments',
    'gh_num_commit_comments',
    'gh_num_pr_comments',
    'git_diff_src_churn',
    'git_diff_test_churn',
    'gh_diff_files_added',
    'gh_diff_files_deleted',
    'gh_diff_files_modified',
    'gh_diff_tests_added',
    'gh_diff_tests_deleted',
    'gh_diff_src_files',
    'gh_diff_doc_files',
    'gh_diff_other_files',
    'gh_num_commits_on_files_touched',
    'gh_sloc',
    'gh_test_lines_per_kloc',
    'gh_test_cases_per_kloc',
    'gh_asserts_cases_per_kloc',
    'gh_by_core_team_member',
    'gh_build_started_at',
    'gh_repo_age',
    'gh_repo_num_commits',
    'tr_build_number',
]

CATEGORICAL_COLUMNS = [
    'gh_lang',
    'git_merged_with',
    'git_prev_commit_resolution_status',
    'branch_group',
]


def branch_group(value):
    """Group raw branch names into compact model-ready categories."""
    branch = str(value).lower().strip()

    if branch in ('', 'nan', 'none', 'na', 'unknown'):
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


def clean_chunk(chunk):
    """Clean and feature-engineer one raw CSV chunk."""
    chunk = chunk.copy()

    status = chunk[TARGET_COLUMN].astype(str).str.lower().str.strip()
    chunk = chunk.loc[status.isin(TARGET_MAP)].copy()
    chunk[TARGET_COLUMN] = status.loc[chunk.index].map(TARGET_MAP).astype(int)

    chunk['gh_is_pr'] = chunk['gh_is_pr'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
    chunk['gh_by_core_team_member'] = (
        chunk['gh_by_core_team_member'].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int)
    )

    started_at = pd.to_datetime(chunk['gh_build_started_at'], errors='coerce')
    chunk['build_hour'] = started_at.dt.hour
    chunk['build_day_of_week'] = started_at.dt.dayofweek
    chunk['is_weekend'] = started_at.dt.dayofweek.isin([5, 6]).astype(int)

    src_churn = pd.to_numeric(chunk['git_diff_src_churn'], errors='coerce').fillna(0)
    test_churn = pd.to_numeric(chunk['git_diff_test_churn'], errors='coerce').fillna(0)
    chunk['total_code_churn'] = src_churn + test_churn
    chunk['test_to_src_ratio'] = np.where(src_churn > 0, test_churn / src_churn, 0.0)
    chunk['total_files_changed'] = (
        pd.to_numeric(chunk['gh_diff_files_added'], errors='coerce').fillna(0)
        + pd.to_numeric(chunk['gh_diff_files_deleted'], errors='coerce').fillna(0)
        + pd.to_numeric(chunk['gh_diff_files_modified'], errors='coerce').fillna(0)
    )
    chunk['branch_group'] = chunk['git_branch'].map(branch_group)

    chunk = chunk.drop(columns=['gh_build_started_at', 'git_branch'])
    return chunk.drop_duplicates()


def sample_csv(path, max_rows, chunksize):
    """Keep a reproducible stratified sample from a large CSV."""
    reservoirs = {}
    per_class_limit = max(1, int(np.ceil(max_rows / 2)))
    rng = np.random.default_rng(RANDOM_STATE)
    total_seen = 0

    for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, chunksize=chunksize, low_memory=False):
        chunk = clean_chunk(chunk)
        if chunk.empty:
            continue

        total_seen += len(chunk)
        chunk['_sample_key'] = rng.random(len(chunk))

        for label, group in chunk.groupby(TARGET_COLUMN):
            current = reservoirs.get(label)
            combined = group if current is None else pd.concat([current, group], ignore_index=True)
            reservoirs[label] = combined.nsmallest(per_class_limit, '_sample_key')

        kept = sum(len(frame) for frame in reservoirs.values())
        print(f'  scanned {total_seen:>10,} usable rows | kept {kept:>8,}', flush=True)

    if not reservoirs:
        raise ValueError('No usable training rows were found in the raw CSV.')

    return (
        pd.concat(reservoirs.values(), ignore_index=True)
        .nsmallest(max_rows, '_sample_key')
        .drop(columns=['_sample_key'])
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def load_training_frame(path, max_rows, chunksize):
    """Load either a chunked sample or the whole file."""
    if max_rows:
        return sample_csv(path, max_rows, chunksize)

    chunks = []
    for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, chunksize=chunksize, low_memory=False):
        cleaned = clean_chunk(chunk)
        if not cleaned.empty:
            chunks.append(cleaned)

    if not chunks:
        raise ValueError('No usable training rows were found in the raw CSV.')

    return pd.concat(chunks, ignore_index=True).sample(frac=1.0, random_state=RANDOM_STATE)


def split_features_target(df):
    """Return X, y, numeric feature list, and categorical feature list."""
    y = df[TARGET_COLUMN].astype(int)
    x = df.drop(columns=[TARGET_COLUMN])

    categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in x.columns]
    numeric_columns = [col for col in x.columns if col not in categorical_columns]

    for col in numeric_columns:
        x[col] = pd.to_numeric(x[col], errors='coerce')

    x[numeric_columns] = x[numeric_columns].replace([np.inf, -np.inf], np.nan)
    x[numeric_columns] = x[numeric_columns].clip(lower=-1_000_000, upper=1_000_000)

    for col in categorical_columns:
        x[col] = x[col].fillna('unknown').astype(str).str.lower()

    return x, y, numeric_columns, categorical_columns


def make_preprocessor(numeric_columns, categorical_columns):
    """Create a train-fitted preprocessing transformer."""
    numeric_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('one_hot', OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipe, numeric_columns),
            ('categorical', categorical_pipe, categorical_columns),
        ],
        remainder='drop',
        verbose_feature_names_out=False,
    )


def get_model_specs(scale_pos_weight):
    """Return candidate classifiers."""
    specs = {
        'logistic_regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='liblinear',
            random_state=RANDOM_STATE,
        ),
        'hist_gradient_boosting': HistGradientBoostingClassifier(
            learning_rate=0.06,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=RANDOM_STATE,
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=220,
            max_depth=22,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if find_spec('xgboost') is not None:
        from xgboost import XGBClassifier

        specs['xgboost'] = XGBClassifier(
            n_estimators=320,
            max_depth=6,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )

    return specs


def predict_scores(model, x):
    """Return scores for threshold-based metrics and ROC-AUC."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    return model.predict(x)


def evaluate_model(model, x, y):
    """Evaluate one fitted model."""
    scores = predict_scores(model, x)
    predictions = (scores >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y, predictions),
        'precision': precision_score(y, predictions, zero_division=0),
        'recall': recall_score(y, predictions, zero_division=0),
        'f1': f1_score(y, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y, scores) if y.nunique() > 1 else np.nan,
    }


def save_plots(model, model_name, x_test, y_test, output_dir):
    """Save confusion matrix and ROC curve plots."""
    scores = predict_scores(model, x_test)
    predictions = (scores >= 0.5).astype(int)

    ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions),
        display_labels=['passed', 'failed_or_errored'],
    ).plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_confusion_matrix.png'), dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title(f'ROC Curve - {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'raw_roc_curve.png'), dpi=160)
    plt.close()


def get_feature_importance(model, feature_names):
    """Extract feature importance from the classifier inside a pipeline."""
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        values = classifier.feature_importances_
    elif hasattr(classifier, 'coef_'):
        values = np.abs(classifier.coef_[0])
    else:
        return pd.DataFrame(columns=['feature', 'importance'])

    return (
        pd.DataFrame({'feature': feature_names, 'importance': values})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


def train_from_raw(args):
    """Run the raw-data training workflow."""
    start = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Input CSV : {args.input}')
    print(f'Output dir: {args.output_dir}')
    print(f'Max rows  : {args.max_rows:,}' if args.max_rows else 'Max rows  : all rows')

    print('\nLoading raw data sample ...')
    df = load_training_frame(args.input, args.max_rows, args.chunksize)
    print(f'  final sample: {df.shape[0]:,} rows x {df.shape[1]} columns')
    print(f"  target distribution: {df[TARGET_COLUMN].value_counts(normalize=True).sort_index().round(4).to_dict()}")

    x, y, numeric_columns, categorical_columns = split_features_target(df)
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    negative = max(1, int((y_train == 0).sum()))
    positive = max(1, int((y_train == 1).sum()))
    scale_pos_weight = negative / positive

    print('\nTraining candidate models ...')
    metrics = []
    fitted = {}
    specs = get_model_specs(scale_pos_weight)

    for name, classifier in specs.items():
        print(f'  training {name} ...', flush=True)
        model_start = time.time()
        model = Pipeline(
            steps=[
                ('preprocessor', make_preprocessor(numeric_columns, categorical_columns)),
                ('classifier', classifier),
            ]
        )
        model.fit(x_train, y_train)
        validation_metrics = evaluate_model(model, x_validation, y_validation)
        validation_metrics.update({'model': name, 'fit_seconds': time.time() - model_start})
        metrics.append(validation_metrics)
        fitted[name] = model
        print(
            f"    f1={validation_metrics['f1']:.4f} "
            f"roc_auc={validation_metrics['roc_auc']:.4f} "
            f"recall={validation_metrics['recall']:.4f}"
        )

    metrics_df = (
        pd.DataFrame(metrics)
        .sort_values(['f1', 'roc_auc', 'recall'], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    best_name = metrics_df.loc[0, 'model']
    best_model = fitted[best_name]
    test_metrics = evaluate_model(best_model, x_test, y_test)

    feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    feature_importance = get_feature_importance(best_model, feature_names)

    joblib.dump(best_model, os.path.join(args.output_dir, 'best_raw_model.joblib'))
    metrics_df.to_csv(os.path.join(args.output_dir, 'raw_model_metrics.csv'), index=False)
    feature_importance.to_csv(os.path.join(args.output_dir, 'raw_feature_importance.csv'), index=False)
    save_plots(best_model, best_name, x_test, y_test, args.output_dir)

    report = {
        'best_model': best_name,
        'target_mapping': TARGET_MAP,
        'rows_used': int(len(df)),
        'numeric_columns': numeric_columns,
        'categorical_columns': categorical_columns,
        'validation_metrics': metrics_df.to_dict(orient='records'),
        'test_metrics': test_metrics,
        'runtime_seconds': time.time() - start,
        'input_csv': args.input,
        'random_state': RANDOM_STATE,
    }
    with open(os.path.join(args.output_dir, 'raw_model_report.json'), 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)

    print(f'\nBest validation model: {best_name}')
    print(
        f"Test metrics: accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )
    print('\nSaved files:')
    print(f"  {os.path.join(args.output_dir, 'best_raw_model.joblib')}")
    print(f"  {os.path.join(args.output_dir, 'raw_model_metrics.csv')}")
    print(f"  {os.path.join(args.output_dir, 'raw_model_report.json')}")
    print(f"  {os.path.join(args.output_dir, 'raw_feature_importance.csv')}")
    print(f"  {os.path.join(args.output_dir, 'raw_confusion_matrix.png')}")
    print(f"  {os.path.join(args.output_dir, 'raw_roc_curve.png')}")
    print(f'\nDone in {time.time() - start:.1f}s')


def parse_args():
    parser = argparse.ArgumentParser(description='Train models directly from raw TravisTorrent CSV.')
    parser.add_argument('--input', default='data/data.csv', help='Path to raw data.csv')
    parser.add_argument('--output-dir', default='models', help='Directory for model artifacts')
    parser.add_argument('--max-rows', type=int, default=200_000, help='Stratified sample size from raw CSV')
    parser.add_argument('--chunksize', type=int, default=100_000, help='Rows per read chunk')
    return parser.parse_args()


if __name__ == '__main__':
    train_from_raw(parse_args())
