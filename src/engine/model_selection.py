from typing import Tuple

import numpy as np


def kfold(dataset: Tuple[np.ndarray, np.ndarray], n_splits):
    """Split dataset to 'n_splits' folds."""
    X, y = dataset
    n_samples = X.shape[0]  # features len
    indices = np.arange(n_samples)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1

    current = 0
    for f_size in fold_sizes:
        start, stop = current, current + f_size
        fold_ind = indices[start:stop]
        other_ind = np.setdiff1d(indices, fold_ind, assume_unique=False)

        fold = np.take(X, fold_ind, axis=0), np.take(y, fold_ind, axis=0)
        other = np.take(X, other_ind, axis=0), np.take(y, other_ind, axis=0)
        yield fold, other
        current = stop
