"""Classifier wrapper. Safe by default: verify contract before classifying.

The repository proves the exact training signal contract in HITClassification.ipynb:
X_raw.reshape(n_impulses, 2, n_samples) and then crop X[:, :, 25:75].
The pretrained CNN is in hitsight_cnn.py and cnn4_tcn_best.pt.
"""

import os
import numpy as np


class Classifier:
    def __init__(self, checkpoint='cnn4_tcn_best.pt', device='cpu'):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.mean = None
        self.std = None
        self.classes = None

    def load(self):
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f'Classifier checkpoint not found: {self.checkpoint}')
        try:
            from hitsight_cnn import load_cnn
            self.model, self.mean, self.std, self.classes = load_cnn(self.checkpoint, self.device)
            return True
        except Exception as exc:
            raise RuntimeError(f'Could not load classifier: {exc}')

    def verify_input_shape(self, x):
        arr = np.asarray(x)
        # The notebook crops X[:, :, 25:75], so the training tensor window is 50 frames long.
        if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] == 50:
            return True
        return False

    def predict(self, x):
        if self.model is None:
            raise RuntimeError('Classifier has not been loaded. Call load() first.')
        if not self.verify_input_shape(x):
            raise ValueError('Incompatible model input. Expected n_impulses x 2 x 50 (cropped 25:75 ms window) signal tensor.')
        return {'status': 'classifier_not_integrated', 'classes': self.classes}
