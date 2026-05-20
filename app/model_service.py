"""Model loading, metadata access, and prediction helpers for the web app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / 'models' / 'best_raw_model.joblib'
REPORT_PATH = PROJECT_ROOT / 'models' / 'raw_model_report.json'
METRICS_PATH = PROJECT_ROOT / 'models' / 'raw_model_metrics.csv'
IMPORTANCE_PATH = PROJECT_ROOT / 'models' / 'raw_feature_importance.csv'


DEFAULT_INPUT = {
    'gh_is_pr': 1,
    'gh_pull_req_num': 124,
    'gh_lang': 'python',
    'git_merged_with': 'merge_button',
    'git_prev_commit_resolution_status': 'build_found',
    'tr_prev_build': 1,
    'gh_team_size': 8,
    'git_num_all_built_commits': 340,
    'gh_num_issue_comments': 4,
    'gh_num_commit_comments': 2,
    'gh_num_pr_comments': 7,
    'git_diff_src_churn': 180,
    'git_diff_test_churn': 45,
    'gh_diff_files_added': 3,
    'gh_diff_files_deleted': 1,
    'gh_diff_files_modified': 9,
    'gh_diff_tests_added': 2,
    'gh_diff_tests_deleted': 0,
    'gh_diff_src_files': 8,
    'gh_diff_doc_files': 1,
    'gh_diff_other_files': 2,
    'gh_num_commits_on_files_touched': 18,
    'gh_sloc': 52000,
    'gh_test_lines_per_kloc': 120,
    'gh_test_cases_per_kloc': 34,
    'gh_asserts_cases_per_kloc': 86,
    'gh_by_core_team_member': 1,
    'gh_repo_age': 2100,
    'gh_repo_num_commits': 5800,
    'tr_build_number': 912,
    'build_hour': 13,
    'build_day_of_week': 2,
    'is_weekend': 0,
    'total_code_churn': 225,
    'test_to_src_ratio': 0.25,
    'total_files_changed': 13,
    'branch_group': 'feature',
}


FIELD_DETAILS = [
    {'name': 'gh_lang', 'label': 'Primary language', 'type': 'select', 'options': ['python', 'ruby', 'java', 'javascript', 'go', 'php', 'cpp', 'unknown']},
    {'name': 'branch_group', 'label': 'Branch group', 'type': 'select', 'options': ['feature', 'fix', 'main', 'develop', 'release', 'other', 'unknown']},
    {'name': 'git_prev_commit_resolution_status', 'label': 'Previous commit status', 'type': 'select', 'options': ['build_found', 'merge_found', 'no_previous_build', 'unknown']},
    {'name': 'git_merged_with', 'label': 'Merge method', 'type': 'select', 'options': ['merge_button', 'commits_in_master', 'unknown']},
    {'name': 'gh_is_pr', 'label': 'Pull request build', 'type': 'boolean'},
    {'name': 'gh_by_core_team_member', 'label': 'Core team member', 'type': 'boolean'},
    {'name': 'gh_team_size', 'label': 'Team size', 'type': 'number', 'min': 0, 'max': 200},
    {'name': 'gh_sloc', 'label': 'Source lines of code', 'type': 'number', 'min': 0, 'max': 5000000},
    {'name': 'gh_repo_age', 'label': 'Repository age days', 'type': 'number', 'min': 0, 'max': 10000},
    {'name': 'gh_repo_num_commits', 'label': 'Repository commits', 'type': 'number', 'min': 0, 'max': 500000},
    {'name': 'git_diff_src_churn', 'label': 'Source churn', 'type': 'number', 'min': 0, 'max': 100000},
    {'name': 'git_diff_test_churn', 'label': 'Test churn', 'type': 'number', 'min': 0, 'max': 100000},
    {'name': 'total_files_changed', 'label': 'Files changed', 'type': 'number', 'min': 0, 'max': 5000},
    {'name': 'gh_diff_files_added', 'label': 'Files added', 'type': 'number', 'min': 0, 'max': 5000},
    {'name': 'gh_diff_files_deleted', 'label': 'Files deleted', 'type': 'number', 'min': 0, 'max': 5000},
    {'name': 'gh_diff_files_modified', 'label': 'Files modified', 'type': 'number', 'min': 0, 'max': 5000},
    {'name': 'gh_test_lines_per_kloc', 'label': 'Test lines per KLOC', 'type': 'number', 'min': 0, 'max': 2000},
    {'name': 'gh_test_cases_per_kloc', 'label': 'Test cases per KLOC', 'type': 'number', 'min': 0, 'max': 1000},
    {'name': 'gh_asserts_cases_per_kloc', 'label': 'Asserts per KLOC', 'type': 'number', 'min': 0, 'max': 3000},
    {'name': 'gh_num_commits_on_files_touched', 'label': 'Commits on touched files', 'type': 'number', 'min': 0, 'max': 10000},
    {'name': 'build_hour', 'label': 'Build hour', 'type': 'number', 'min': 0, 'max': 23},
    {'name': 'build_day_of_week', 'label': 'Day of week', 'type': 'number', 'min': 0, 'max': 6},
    {'name': 'tr_build_number', 'label': 'Build number', 'type': 'number', 'min': 0, 'max': 1000000},
]


PIPELINE_STEPS = [
    {
        'title': 'Raw build signal',
        'detail': 'The interface collects repository, pull request, churn, branch, and historical build features.',
    },
    {
        'title': 'Feature repair',
        'detail': 'Missing values are imputed, numeric values are clipped for stability, and categoricals are normalized.',
    },
    {
        'title': 'Model pipeline',
        'detail': 'The saved scikit-learn pipeline applies median imputation, scaling, and one-hot encoding.',
    },
    {
        'title': 'Failure probability',
        'detail': 'The Random Forest returns a probability for failed or errored CI/CD builds.',
    },
    {
        'title': 'Decision band',
        'detail': 'The probability is translated into low, medium, or high operational risk for fast triage.',
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def _read_csv_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    if limit:
        frame = frame.head(limit)
    return frame.replace({np.nan: None}).to_dict(orient='records')


def load_model():
    """Load the trained pipeline from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Missing trained model: {MODEL_PATH}')
    return joblib.load(MODEL_PATH)


def load_dashboard_data() -> dict[str, Any]:
    """Return model metrics and metadata for the frontend."""
    report = _read_json(REPORT_PATH)
    metrics = _read_csv_records(METRICS_PATH)
    importance = _read_csv_records(IMPORTANCE_PATH, limit=12)

    return {
        'model_available': MODEL_PATH.exists(),
        'model_name': report.get('best_model', 'unknown'),
        'rows_used': report.get('rows_used', 0),
        'test_metrics': report.get('test_metrics', {}),
        'validation_metrics': report.get('validation_metrics', metrics),
        'feature_importance': importance,
        'numeric_columns': report.get('numeric_columns', []),
        'categorical_columns': report.get('categorical_columns', []),
        'pipeline_steps': PIPELINE_STEPS,
        'fields': FIELD_DETAILS,
        'defaults': DEFAULT_INPUT,
        'artifacts': [
            'models/best_raw_model.joblib',
            'models/raw_model_metrics.csv',
            'models/raw_model_report.json',
            'models/raw_feature_importance.csv',
            'models/raw_confusion_matrix.png',
            'models/raw_roc_curve.png',
        ],
    }


def coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge user input with defaults and coerce values to model-ready types."""
    values = {**DEFAULT_INPUT, **(payload or {})}
    numeric_columns = [name for name, value in DEFAULT_INPUT.items() if isinstance(value, (int, float))]

    for col in numeric_columns:
        try:
            values[col] = float(values[col])
        except (TypeError, ValueError):
            values[col] = float(DEFAULT_INPUT[col])

    for col in ['gh_is_pr', 'gh_by_core_team_member', 'is_weekend']:
        values[col] = int(bool(values[col]))

    values['total_code_churn'] = values['git_diff_src_churn'] + values['git_diff_test_churn']
    values['total_files_changed'] = (
        values['gh_diff_files_added'] + values['gh_diff_files_deleted'] + values['gh_diff_files_modified']
    )
    values['test_to_src_ratio'] = (
        values['git_diff_test_churn'] / values['git_diff_src_churn']
        if values['git_diff_src_churn'] > 0
        else 0.0
    )
    values['is_weekend'] = int(values['build_day_of_week'] in [5, 6])

    for col in ['gh_lang', 'git_merged_with', 'git_prev_commit_resolution_status', 'branch_group']:
        values[col] = str(values[col]).strip().lower() or 'unknown'

    return values


def risk_label(probability: float) -> str:
    """Convert failure probability into an easy operational label."""
    if probability >= 0.70:
        return 'high'
    if probability >= 0.45:
        return 'medium'
    return 'low'


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Run one prediction through the saved model pipeline."""
    model = load_model()
    features = coerce_payload(payload)
    frame = pd.DataFrame([features])

    if hasattr(model, 'predict_proba'):
        failure_probability = float(model.predict_proba(frame)[0][1])
    else:
        failure_probability = float(model.predict(frame)[0])

    passed_probability = 1.0 - failure_probability
    label = risk_label(failure_probability)

    return {
        'prediction': 'failed_or_errored' if failure_probability >= 0.5 else 'passed',
        'risk': label,
        'failure_probability': failure_probability,
        'passed_probability': passed_probability,
        'confidence_percent': round(max(failure_probability, passed_probability) * 100, 2),
        'features': features,
        'trace': [
            {'name': 'Input accepted', 'status': 'complete', 'value': f'{len(features)} model fields assembled'},
            {'name': 'Derived features', 'status': 'complete', 'value': f"churn={features['total_code_churn']:.0f}, files={features['total_files_changed']:.0f}"},
            {'name': 'Preprocessor', 'status': 'complete', 'value': 'median imputation, scaling, one-hot encoding'},
            {'name': 'Classifier', 'status': 'complete', 'value': f'{failure_probability * 100:.2f}% failure probability'},
            {'name': 'Risk band', 'status': 'complete', 'value': label.upper()},
        ],
    }
