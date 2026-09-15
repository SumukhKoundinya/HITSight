"""Prototype configuration for the integrated vHIT acquisition pipeline."""
import os
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINING_CONTRACT = {
    'channels': 2,
    'raw_sequence_length': 175,
    'cropped_sequence_length': 50,
    'window_ms': (25, 75),
    'sample_period_ms': 1.0,
    'normalization': 'mean/std per channel per feature axis, computed from training data using (X_train - mean) / std on axis=(0,2)',
    'filtering': 'no filtering is declared in the notebook; the repository crops the signal from 25:75 ms and normalizes in the training code',
    'feature_representation': 'n_impulses x 2 x n_samples, then the notebook crops X[:, :, 25:75] to the model window',
    'label_mapping': 'label_encoder.classes_ from HITClassification.ipynb',
}

# Safe defaults for the proof-of-concept CLI.
DEFAULT_SOURCE = 0
DEFAULT_DURATION = 10.0
DEFAULT_RECORDING = True


def session_output_path(prefix: str = 'session') -> str:
    stamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return os.path.join(OUTPUT_DIR, f'{prefix}_{stamp}.csv')
