"""
Main ML Preprocessing Pipeline
===========================================================
Runs the project preprocessing scripts in the correct order:

  1. clean_data.py
  2. feature_engineering.py
  3. encode.py
  4. split.py
  5. scale_balance.py
"""

import argparse
import os
import time

from src.preprocessing.clean_data import clean_data
from src.preprocessing.encode import encode_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.scale_balance import scale_and_balance
from src.preprocessing.split import split_data


STEPS = ['clean', 'feature', 'encode', 'split', 'scale_balance']


def outputs_exist(paths):
    """Return True when all expected outputs already exist."""
    return all(os.path.exists(path) for path in paths)


def run_step(name, func, output_paths, skip_existing=False):
    """Run a pipeline step with simple timing and optional skip behavior."""
    print('\n' + '=' * 70)
    print(f'RUNNING STEP: {name}')
    print('=' * 70)

    if skip_existing and outputs_exist(output_paths):
        print(f'Skipped {name}: expected output already exists.')
        for path in output_paths:
            print(f'  {path}')
        return

    start = time.time()
    func()
    elapsed = time.time() - start

    print(f'\nFinished step: {name} ({elapsed:.1f}s)')


def selected_steps(from_step=None, only_step=None):
    """Resolve which pipeline steps should run."""
    if only_step:
        return [only_step]

    if from_step:
        start_index = STEPS.index(from_step)
        return STEPS[start_index:]

    return STEPS


def run_pipeline(args):
    """Run selected preprocessing pipeline steps."""
    steps_to_run = selected_steps(args.from_step, args.only_step)

    pipeline = {
        'clean': {
            'func': lambda: clean_data(args.raw_data, args.cleaned_data, args.chunk_size),
            'outputs': [args.cleaned_data],
        },
        'feature': {
            'func': lambda: engineer_features(args.cleaned_data, args.featured_data),
            'outputs': [args.featured_data],
        },
        'encode': {
            'func': lambda: encode_data(args.featured_data, args.encoded_data),
            'outputs': [args.encoded_data],
        },
        'split': {
            'func': lambda: split_data(args.encoded_data, args.split_dir),
            'outputs': [
                os.path.join(args.split_dir, 'train_data.csv'),
                os.path.join(args.split_dir, 'validation_data.csv'),
                os.path.join(args.split_dir, 'test_data.csv'),
            ],
        },
        'scale_balance': {
            'func': lambda: scale_and_balance(args.split_dir, args.scaled_balanced_dir),
            'outputs': [
                os.path.join(args.scaled_balanced_dir, 'scaled_train_data.csv'),
                os.path.join(args.scaled_balanced_dir, 'scaled_validation_data.csv'),
                os.path.join(args.scaled_balanced_dir, 'scaled_test_data.csv'),
                os.path.join(args.scaled_balanced_dir, 'balanced_train_data.csv'),
                os.path.join(args.scaled_balanced_dir, 'standard_scaler.joblib'),
                os.path.join(args.scaled_balanced_dir, 'train_numeric_medians.csv'),
                os.path.join(args.scaled_balanced_dir, 'scale_balance_metadata.json'),
            ],
        },
    }

    print('Selected pipeline steps:')
    for step in steps_to_run:
        print(f'  - {step}')

    pipeline_start = time.time()

    for step in steps_to_run:
        run_step(
            step,
            pipeline[step]['func'],
            pipeline[step]['outputs'],
            skip_existing=args.skip_existing,
        )

    elapsed = time.time() - pipeline_start
    print('\n' + '=' * 70)
    print(f'PIPELINE COMPLETE ({elapsed:.1f}s)')
    print('=' * 70)


def parse_args():
    parser = argparse.ArgumentParser(description='Run the CI/CD failure prediction preprocessing pipeline.')

    parser.add_argument('--raw-data', default='data/data.csv', help='Path to raw TravisTorrent CSV')
    parser.add_argument('--cleaned-data', default='data/cleaned_data.csv', help='Path for cleaned CSV')
    parser.add_argument('--featured-data', default='data/featured_data.csv', help='Path for featured CSV')
    parser.add_argument('--encoded-data', default='data/encoded_data.csv', help='Path for encoded CSV')
    parser.add_argument('--split-dir', default='data/split', help='Directory for train/validation/test splits')
    parser.add_argument(
        '--scaled-balanced-dir',
        default='data/scaled_balanced',
        help='Directory for scaled and balanced model-ready data',
    )
    parser.add_argument('--chunk-size', default=250_000, type=int, help='Chunk size for raw cleaning')
    parser.add_argument(
        '--from-step',
        choices=STEPS,
        help='Run from this step through the end of the pipeline',
    )
    parser.add_argument(
        '--only-step',
        choices=STEPS,
        help='Run only one pipeline step',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip a selected step when all of its expected outputs already exist',
    )

    return parser.parse_args()


def main():
    run_pipeline(parse_args())
