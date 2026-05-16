"""
Model Training Pipeline
===========================================================
Trains CI/CD build failure classifiers from model-ready processed data.

Reads:
  data/scaled_balanced/balanced_train_data.csv
  data/scaled_balanced/scaled_validation_data.csv
  data/scaled_balanced/scaled_test_data.csv

Writes:
  models/best_model.joblib
  models/model_metrics.csv
  models/model_metrics.json
  models/confusion_matrix.png
  models/roc_curve.png
  models/feature_importance.csv
  models/feature_importance.png

Per project docs:
  1. Train Logistic Regression as the baseline.
  2. Train Random Forest, Gradient Boosting, XGBoost, and LightGBM when available.
  3. Use SMOTE-balanced training data only for fitting.
  4. Evaluate on unbalanced validation and held-out test data.
  5. Select the best model using validation F1, ROC-AUC, then recall.
  6. Optionally tune the top two models with RandomizedSearchCV.
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
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


TARGET_COLUMN = 'tr_status'
RANDOM_STATE = 42


def sample_csv(path, max_rows, chunksize):
    """Read a large CSV in chunks and keep a stratified random sample."""
    reservoirs = {}
    per_class_limit = max(1, int(np.ceil(max_rows / 2)))
    rng = np.random.default_rng(RANDOM_STATE)

    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if TARGET_COLUMN not in chunk.columns:
            raise KeyError(f'Missing required target column: {TARGET_COLUMN}')

        chunk = chunk.copy()
        chunk['_sample_key'] = rng.random(len(chunk))

        for label, group in chunk.groupby(TARGET_COLUMN):
            current = reservoirs.get(label)
            combined = group if current is None else pd.concat([current, group], ignore_index=True)
            reservoirs[label] = combined.nsmallest(per_class_limit, '_sample_key')

    if not reservoirs:
        return pd.DataFrame()

    return (
        pd.concat(reservoirs.values(), ignore_index=True)
        .nsmallest(max_rows, '_sample_key')
        .drop(columns=['_sample_key'])
        .sample(frac=1.0, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def load_frame(path, max_rows=None, chunksize=100_000):
    """Load a CSV, using chunked sampling for fast experiments on large files."""
    if max_rows:
        return sample_csv(path, max_rows, chunksize)

    return pd.read_csv(path, low_memory=False)


def load_model_data(input_dir, max_train_rows=None, max_eval_rows=None, chunksize=100_000):
    """Load balanced train and untouched validation/test datasets."""
    paths = {
        'train': os.path.join(input_dir, 'balanced_train_data.csv'),
        'validation': os.path.join(input_dir, 'scaled_validation_data.csv'),
        'test': os.path.join(input_dir, 'scaled_test_data.csv'),
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {name} data: {path}')

    train_df = load_frame(paths['train'], max_train_rows, chunksize)
    validation_df = load_frame(paths['validation'], max_eval_rows, chunksize)
    test_df = load_frame(paths['test'], max_eval_rows, chunksize)

    feature_columns = [col for col in train_df.columns if col != TARGET_COLUMN]
    expected_columns = feature_columns + [TARGET_COLUMN]

    for name, df in [('validation', validation_df), ('test', test_df)]:
        missing = sorted(set(expected_columns) - set(df.columns))
        if missing:
            raise ValueError(f'{name} data is missing columns: {missing[:10]}')

    train_df = train_df[expected_columns]
    validation_df = validation_df[expected_columns]
    test_df = test_df[expected_columns]

    return train_df, validation_df, test_df, feature_columns


def split_xy(df, feature_columns):
    """Split features and target."""
    return df[feature_columns], df[TARGET_COLUMN].astype(int)


def build_model_specs():
    """Return available model definitions and tuning spaces."""
    specs = {
        'logistic_regression': {
            'model': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='liblinear',
                random_state=RANDOM_STATE,
            ),
            'params': {
                'C': loguniform(0.01, 100),
                'penalty': ['l1', 'l2'],
            },
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=250,
                max_depth=18,
                min_samples_leaf=2,
                class_weight='balanced',
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            'params': {
                'n_estimators': randint(150, 450),
                'max_depth': [None, 10, 16, 22, 30],
                'min_samples_leaf': randint(1, 6),
                'max_features': ['sqrt', 'log2', None],
            },
        },
        'gradient_boosting': {
            'model': HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=180,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=RANDOM_STATE,
            ),
            'params': {
                'learning_rate': loguniform(0.02, 0.2),
                'max_iter': randint(100, 350),
                'max_leaf_nodes': randint(15, 63),
                'l2_regularization': loguniform(0.001, 1.0),
            },
        },
    }

    if find_spec('xgboost') is not None:
        from xgboost import XGBClassifier

        specs['xgboost'] = {
            'model': XGBClassifier(
                n_estimators=250,
                max_depth=5,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            'params': {
                'n_estimators': randint(150, 450),
                'max_depth': randint(3, 9),
                'learning_rate': loguniform(0.02, 0.2),
                'subsample': [0.75, 0.85, 1.0],
                'colsample_bytree': [0.75, 0.85, 1.0],
            },
        }
    else:
        print('  [!] Skipping XGBoost: package is not installed.')

    if find_spec('lightgbm') is not None:
        from lightgbm import LGBMClassifier

        specs['lightgbm'] = {
            'model': LGBMClassifier(
                n_estimators=300,
                learning_rate=0.06,
                num_leaves=31,
                class_weight='balanced',
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            'params': {
                'n_estimators': randint(150, 500),
                'learning_rate': loguniform(0.02, 0.2),
                'num_leaves': randint(15, 80),
                'min_child_samples': randint(10, 80),
            },
        }
    else:
        print('  [!] Skipping LightGBM: package is not installed.')

    return specs


def predict_scores(model, x):
    """Return probability-like scores for ROC-AUC and thresholding."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, 'decision_function'):
        return model.decision_function(x)
    return model.predict(x)


def evaluate_model(model, x, y):
    """Evaluate one fitted classifier."""
    scores = predict_scores(model, x)
    predictions = (scores >= 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y, predictions),
        'precision': precision_score(y, predictions, zero_division=0),
        'recall': recall_score(y, predictions, zero_division=0),
        'f1': f1_score(y, predictions, zero_division=0),
        'roc_auc': roc_auc_score(y, scores) if y.nunique() > 1 else np.nan,
    }


def fit_and_score_models(specs, x_train, y_train, x_validation, y_validation):
    """Train all candidate models and score them on validation data."""
    results = []
    fitted_models = {}

    for name, spec in specs.items():
        print(f'\nTraining {name} ...')
        start = time.time()
        model = spec['model']
        model.fit(x_train, y_train)
        elapsed = time.time() - start

        metrics = evaluate_model(model, x_validation, y_validation)
        metrics.update({'model': name, 'fit_seconds': elapsed, 'tuned': False})
        results.append(metrics)
        fitted_models[name] = model

        print(
            f"  validation f1={metrics['f1']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"({elapsed:.1f}s)"
        )

    return fitted_models, pd.DataFrame(results)


def rank_results(metrics_df):
    """Rank models by validation F1, ROC-AUC, then recall."""
    return metrics_df.sort_values(
        ['f1', 'roc_auc', 'recall'],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def tune_top_models(specs, metrics_df, x_train, y_train, x_validation, y_validation, n_iter):
    """Tune the top two validation models using randomized 5-fold CV."""
    tuned_models = {}
    tuned_results = []
    ranked = rank_results(metrics_df)
    top_names = ranked['model'].head(2).tolist()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name in top_names:
        print(f'\nTuning {name} with RandomizedSearchCV ...')
        start = time.time()
        search = RandomizedSearchCV(
            estimator=specs[name]['model'],
            param_distributions=specs[name]['params'],
            n_iter=n_iter,
            scoring='f1',
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        search.fit(x_train, y_train)
        elapsed = time.time() - start

        model = search.best_estimator_
        metrics = evaluate_model(model, x_validation, y_validation)
        metrics.update({
            'model': f'{name}_tuned',
            'fit_seconds': elapsed,
            'tuned': True,
            'best_params': search.best_params_,
            'cv_best_f1': search.best_score_,
        })
        tuned_models[f'{name}_tuned'] = model
        tuned_results.append(metrics)

        print(
            f"  tuned validation f1={metrics['f1']:.4f} "
            f"recall={metrics['recall']:.4f} "
            f"roc_auc={metrics['roc_auc']:.4f} "
            f"({elapsed:.1f}s)"
        )

    return tuned_models, pd.DataFrame(tuned_results)


def get_feature_importance(model, feature_columns):
    """Extract model feature importance when supported."""
    if hasattr(model, 'feature_importances_'):
        values = model.feature_importances_
    elif hasattr(model, 'coef_'):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=['feature', 'importance'])

    return (
        pd.DataFrame({'feature': feature_columns, 'importance': values})
        .sort_values('importance', ascending=False)
        .reset_index(drop=True)
    )


def save_plots(best_model, best_name, x_test, y_test, feature_importance, output_dir):
    """Save confusion matrix, ROC curve, and feature importance plots."""
    os.makedirs(output_dir, exist_ok=True)
    scores = predict_scores(best_model, x_test)
    predictions = (scores >= 0.5).astype(int)

    cm = confusion_matrix(y_test, predictions)
    ConfusionMatrixDisplay(cm, display_labels=['passed', 'failed']).plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=160)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, scores)
    plt.title(f'ROC Curve - {best_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=160)
    plt.close()

    if not feature_importance.empty:
        top_features = feature_importance.head(20).sort_values('importance')
        plt.figure(figsize=(9, 7))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.title(f'Top Feature Importance - {best_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=160)
        plt.close()


def to_jsonable(value):
    """Convert numpy/scipy values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value):
        return None
    return value


def save_outputs(
    output_dir,
    best_name,
    best_model,
    feature_columns,
    metrics_df,
    test_metrics,
    feature_importance,
    args,
):
    """Persist the selected model and training reports."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, 'model_metrics.csv')
    metrics_json_path = os.path.join(output_dir, 'model_metrics.json')
    best_model_path = os.path.join(output_dir, 'best_model.joblib')
    importance_path = os.path.join(output_dir, 'feature_importance.csv')

    metrics_df.to_csv(metrics_path, index=False)
    feature_importance.to_csv(importance_path, index=False)

    report = {
        'best_model': best_name,
        'selection_metric_order': ['validation_f1', 'validation_roc_auc', 'validation_recall'],
        'test_metrics': test_metrics,
        'all_validation_metrics': metrics_df.to_dict(orient='records'),
        'input_dir': args.input_dir,
        'target_column': TARGET_COLUMN,
        'feature_count': len(feature_columns),
        'max_train_rows': args.max_train_rows,
        'max_eval_rows': args.max_eval_rows,
        'tuning_enabled': args.tune,
    }

    with open(metrics_json_path, 'w', encoding='utf-8') as file:
        json.dump(to_jsonable(report), file, indent=2)

    joblib.dump(
        {
            'model': best_model,
            'model_name': best_name,
            'feature_columns': feature_columns,
            'target_column': TARGET_COLUMN,
            'threshold': 0.5,
            'test_metrics': test_metrics,
        },
        best_model_path,
    )

    print('\nSaved files:')
    print(f'  Best model         : {best_model_path}')
    print(f'  Metrics CSV        : {metrics_path}')
    print(f'  Metrics JSON       : {metrics_json_path}')
    print(f'  Feature importance : {importance_path}')


def train_models(args):
    """Run the complete model training workflow."""
    start = time.time()
    input_dir = args.input_dir or getattr(args, 'scaled_balanced_dir', 'data/scaled_balanced')
    args.input_dir = input_dir
    print(f'Input dir : {input_dir}')
    print(f'Output dir: {args.output_dir}\n')

    print('Loading processed model data ...')
    train_df, validation_df, test_df, feature_columns = load_model_data(
        input_dir,
        max_train_rows=args.max_train_rows,
        max_eval_rows=args.max_eval_rows,
        chunksize=args.read_chunksize,
    )

    print(f'  Train      : {train_df.shape[0]:,} rows x {train_df.shape[1]} columns')
    print(f'  Validation : {validation_df.shape[0]:,} rows x {validation_df.shape[1]} columns')
    print(f'  Test       : {test_df.shape[0]:,} rows x {test_df.shape[1]} columns')
    print(f'  Features   : {len(feature_columns)}')

    x_train, y_train = split_xy(train_df, feature_columns)
    x_validation, y_validation = split_xy(validation_df, feature_columns)
    x_test, y_test = split_xy(test_df, feature_columns)

    print('\nBuilding candidate models ...')
    specs = build_model_specs()
    print(f"  Models selected: {', '.join(specs.keys())}")

    fitted_models, metrics_df = fit_and_score_models(
        specs,
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    if args.tune:
        tuned_models, tuned_metrics_df = tune_top_models(
            specs,
            metrics_df,
            x_train,
            y_train,
            x_validation,
            y_validation,
            args.tune_iter,
        )
        fitted_models.update(tuned_models)
        metrics_df = pd.concat([metrics_df, tuned_metrics_df], ignore_index=True)

    ranked = rank_results(metrics_df)
    best_name = ranked.loc[0, 'model']
    best_model = fitted_models[best_name]

    print(f'\nBest validation model: {best_name}')
    test_metrics = evaluate_model(best_model, x_test, y_test)
    print(
        f"Test metrics: accuracy={test_metrics['accuracy']:.4f}, "
        f"precision={test_metrics['precision']:.4f}, "
        f"recall={test_metrics['recall']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, "
        f"roc_auc={test_metrics['roc_auc']:.4f}"
    )

    feature_importance = get_feature_importance(best_model, feature_columns)

    print('\nSaving artifacts ...')
    save_outputs(
        args.output_dir,
        best_name,
        best_model,
        feature_columns,
        ranked,
        test_metrics,
        feature_importance,
        args,
    )
    save_plots(best_model, best_name, x_test, y_test, feature_importance, args.output_dir)
    print(f'  Confusion matrix   : {os.path.join(args.output_dir, "confusion_matrix.png")}')
    print(f'  ROC curve          : {os.path.join(args.output_dir, "roc_curve.png")}')
    if not feature_importance.empty:
        print(f'  Feature plot       : {os.path.join(args.output_dir, "feature_importance.png")}')

    elapsed = time.time() - start
    print(f'\nTraining complete ({elapsed:.1f}s)')


def parse_args():
    parser = argparse.ArgumentParser(description='Train CI/CD failure prediction models.')
    parser.add_argument(
        '--input-dir',
        default='data/scaled_balanced',
        help='Directory containing scaled/balanced processed CSV files',
    )
    parser.add_argument('--output-dir', default='models', help='Directory for trained model artifacts')
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Tune the top two validation models with RandomizedSearchCV',
    )
    parser.add_argument('--tune-iter', type=int, default=12, help='RandomizedSearchCV iterations per tuned model')
    parser.add_argument(
        '--max-train-rows',
        type=int,
        help='Optional stratified training sample size for faster experiments',
    )
    parser.add_argument(
        '--max-eval-rows',
        type=int,
        help='Optional stratified validation/test sample size for faster experiments',
    )
    parser.add_argument(
        '--read-chunksize',
        type=int,
        default=100_000,
        help='CSV rows per chunk when sampling large processed files',
    )
    return parser.parse_args()


if __name__ == '__main__':
    train_models(parse_args())
