"""
Preprocessing utilities for the rice_ml package.

This module provides:
- Input validation helpers
- Feature scaling methods
- Row normalization methods
- Train/test and train/validation/test splitting

The functions are intentionally lightweight and NumPy-only so they can be used
throughout the custom machine learning package without relying on scikit-learn.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]

__all__ = [
    "ArrayLike",
    "standardize",
    "minmax_scale",
    "maxabs_scale",
    "l1_normalize_rows",
    "l2_normalize_rows",
    "train_test_split",
    "train_val_test_split",
]


# =============================================================================
# Validation helpers
# =============================================================================

def _as_2d_numeric_array(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert input to a non-empty 2D numeric NumPy array.

    Parameters
    ----------
    X : array-like
        Input feature matrix.
    name : str
        Name used in error messages.

    Returns
    -------
    np.ndarray
        Float NumPy array with shape (n_samples, n_features).
    """
    arr = np.asarray(X)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    try:
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"All elements of {name} must be numeric.") from exc

    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values. Handle missing data first.")

    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values. Handle extreme data first.")

    return arr


def _as_1d_array(y: Optional[ArrayLike], name: str = "y") -> Optional[np.ndarray]:
    """
    Convert target labels to a 1D NumPy array.

    y can be numeric or categorical, so this function does not force float dtype.
    """
    if y is None:
        return None

    arr = np.asarray(y)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")

    return arr


def _check_matching_rows(X: np.ndarray, y: Optional[np.ndarray]) -> None:
    """Check that X and y have the same number of observations."""
    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of rows; "
            f"got X.shape[0]={X.shape[0]} and len(y)={len(y)}."
        )


def _check_fraction(value: float, name: str) -> float:
    """Validate that a split size is a float strictly between 0 and 1."""
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")

    value = float(value)

    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be in the interval (0, 1).")

    return value


def _make_rng(random_state: Optional[int]) -> np.random.Generator:
    """Create a NumPy random number generator from an optional seed."""
    if random_state is None:
        return np.random.default_rng()

    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer or None.")

    return np.random.default_rng(int(random_state))


def _safe_scale(scale: np.ndarray) -> np.ndarray:
    """
    Replace zero scale values with 1.

    This avoids division by zero for constant columns.
    """
    scale = scale.astype(float, copy=True)
    scale[scale == 0.0] = 1.0
    return scale


# =============================================================================
# Feature scaling
# =============================================================================

def standardize(
    X: ArrayLike,
    *,
    with_mean: bool = True,
    with_std: bool = True,
    ddof: int = 0,
    return_params: bool = False,
    mean: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, dict[str, np.ndarray]]]:
    """
    Standardize features column-by-column using z-score scaling.

    The usual transformation is:

        X_scaled = (X - mean) / standard_deviation

    This function supports two modes:

    1. Fit-transform mode:
       If mean and scale are not provided, they are calculated from X.

    2. Transform-only mode:
       If mean and scale are provided, they are reused to transform X.
       This is useful for applying training-set parameters to validation/test data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    with_mean : bool, default=True
        Whether to subtract the column means.
    with_std : bool, default=True
        Whether to divide by the column standard deviations.
    ddof : int, default=0
        Delta degrees of freedom used in standard deviation calculation.
    return_params : bool, default=False
        If True, return the learned mean and scale.
    mean : np.ndarray, optional
        Precomputed column means.
    scale : np.ndarray, optional
        Precomputed column standard deviations.

    Returns
    -------
    np.ndarray
        Standardized feature matrix.

    Or, if return_params=True:

    (np.ndarray, dict)
        Standardized matrix and learned parameters.
    """
    X_arr = _as_2d_numeric_array(X)

    using_saved_params = mean is not None or scale is not None

    if using_saved_params:
        if mean is None or scale is None:
            raise ValueError("Both mean and scale must be provided together.")

        mean = np.asarray(mean, dtype=float)
        scale = np.asarray(scale, dtype=float)

        if mean.shape != (X_arr.shape[1],):
            raise ValueError("mean must have shape (n_features,).")

        if scale.shape != (X_arr.shape[1],):
            raise ValueError("scale must have shape (n_features,).")

        mean_used = mean if with_mean else np.zeros(X_arr.shape[1])
        scale_used = _safe_scale(scale) if with_std else np.ones(X_arr.shape[1])

    else:
        mean_used = X_arr.mean(axis=0) if with_mean else np.zeros(X_arr.shape[1])
        centered = X_arr - mean_used

        if with_std:
            scale_used = _safe_scale(centered.std(axis=0, ddof=ddof))
        else:
            scale_used = np.ones(X_arr.shape[1])

    X_out = (X_arr - mean_used) / scale_used

    if return_params:
        return X_out, {"mean": mean_used, "scale": scale_used}

    return X_out


def minmax_scale(
    X: ArrayLike,
    *,
    feature_range: Tuple[float, float] = (0.0, 1.0),
    return_params: bool = False,
    data_min: Optional[np.ndarray] = None,
    data_range: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, dict[str, Any]]]:
    """
    Scale each feature to a chosen range.

    By default, this maps each column to the interval [0, 1].

    Formula:

        X_scaled = (X - min) / (max - min)
        X_out = X_scaled * (new_max - new_min) + new_min

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    feature_range : tuple, default=(0.0, 1.0)
        Desired output range.
    return_params : bool, default=False
        If True, return the learned min and range.
    data_min : np.ndarray, optional
        Precomputed original column minimums.
    data_range : np.ndarray, optional
        Precomputed original column ranges.

    Returns
    -------
    np.ndarray
        Min-max scaled feature matrix.
    """
    X_arr = _as_2d_numeric_array(X)

    if (
        not isinstance(feature_range, tuple)
        or len(feature_range) != 2
        or not all(isinstance(v, (int, float)) for v in feature_range)
    ):
        raise ValueError("feature_range must be a tuple of two numeric values.")

    new_min, new_max = map(float, feature_range)

    if new_min >= new_max:
        raise ValueError("feature_range must satisfy min < max.")

    using_saved_params = data_min is not None or data_range is not None

    if using_saved_params:
        if data_min is None or data_range is None:
            raise ValueError("Both data_min and data_range must be provided together.")

        data_min = np.asarray(data_min, dtype=float)
        data_range = np.asarray(data_range, dtype=float)

        if data_min.shape != (X_arr.shape[1],):
            raise ValueError("data_min must have shape (n_features,).")

        if data_range.shape != (X_arr.shape[1],):
            raise ValueError("data_range must have shape (n_features,).")

        min_used = data_min
        range_used = _safe_scale(data_range)

    else:
        min_used = X_arr.min(axis=0)
        max_used = X_arr.max(axis=0)
        range_used = _safe_scale(max_used - min_used)

    X_01 = (X_arr - min_used) / range_used
    X_out = X_01 * (new_max - new_min) + new_min

    if return_params:
        return X_out, {
            "data_min": min_used,
            "data_range": range_used,
            "feature_range": (new_min, new_max),
        }

    return X_out


def maxabs_scale(
    X: ArrayLike,
    *,
    return_params: bool = False,
    scale: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, dict[str, np.ndarray]]]:
    """
    Scale each feature by its maximum absolute value.

    Formula:

        X_scaled = X / max(abs(X))

    This keeps zero-centered data zero-centered and is often useful when data
    is sparse or already centered.
    """
    X_arr = _as_2d_numeric_array(X)

    if scale is not None:
        scale_used = np.asarray(scale, dtype=float)

        if scale_used.shape != (X_arr.shape[1],):
            raise ValueError("scale must have shape (n_features,).")

        scale_used = _safe_scale(scale_used)

    else:
        scale_used = _safe_scale(np.max(np.abs(X_arr), axis=0))

    X_out = X_arr / scale_used

    if return_params:
        return X_out, {"scale": scale_used}

    return X_out


# =============================================================================
# Row normalization
# =============================================================================

def l1_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row so its L1 norm equals 1.

    The L1 norm is the sum of absolute values in a row.

    Rows containing all zeros remain all zeros.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    X_arr = _as_2d_numeric_array(X)
    norms = np.sum(np.abs(X_arr), axis=1)
    denominator = np.maximum(norms, eps)[:, None]

    return X_arr / denominator


def l2_normalize_rows(X: ArrayLike, *, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize each row so its L2 norm equals 1.

    The L2 norm is the usual Euclidean length of a row.

    Rows containing all zeros remain all zeros.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    X_arr = _as_2d_numeric_array(X)
    norms = np.sqrt(np.sum(X_arr ** 2, axis=1))
    denominator = np.maximum(norms, eps)[:, None]

    return X_arr / denominator


# =============================================================================
# Dataset splitting
# =============================================================================

def _stratified_split_indices(
    stratify: np.ndarray,
    test_size: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test indices while preserving class proportions.

    This is used for classification tasks where each class should appear in
    approximately the same proportions in the train and test sets.
    """
    classes, encoded = np.unique(stratify, return_inverse=True)

    train_indices: list[int] = []
    test_indices: list[int] = []

    for class_id in range(len(classes)):
        class_indices = np.flatnonzero(encoded == class_id)
        rng.shuffle(class_indices)

        n_class = len(class_indices)
        n_test = int(round(test_size * n_class))

        if n_class > 1:
            n_test = min(max(n_test, 1), n_class - 1)
        else:
            n_test = 0

        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    train_indices = np.asarray(train_indices)
    test_indices = np.asarray(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return train_indices, test_indices


def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, optional
        Target vector.
    test_size : float, default=0.25
        Proportion of data placed in the test set.
    random_state : int, optional
        Seed for reproducible shuffling.
    shuffle : bool, default=True
        Whether to shuffle rows before splitting.
    stratify : array-like, optional
        Class labels used to preserve class proportions across splits.

    Returns
    -------
    If y is provided:
        X_train, X_test, y_train, y_test

    If y is not provided:
        X_train, X_test
    """
    X_arr = _as_2d_numeric_array(X)
    y_arr = _as_1d_array(y)
    _check_matching_rows(X_arr, y_arr)

    test_size = _check_fraction(test_size, "test_size")
    rng = _make_rng(random_state)

    n_samples = X_arr.shape[0]

    if n_samples < 2:
        raise ValueError("At least two samples are required to split data.")

    if stratify is not None:
        stratify_arr = _as_1d_array(stratify, name="stratify")

        if stratify_arr.shape[0] != n_samples:
            raise ValueError("stratify must have the same length as X.")

        train_idx, test_idx = _stratified_split_indices(
            stratify_arr,
            test_size=test_size,
            rng=rng,
        )

    else:
        indices = np.arange(n_samples)

        if shuffle:
            rng.shuffle(indices)

        n_test = int(round(test_size * n_samples))
        n_test = min(max(n_test, 1), n_samples - 1)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    if y_arr is None:
        return X_arr[train_idx], X_arr[test_idx]

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def train_val_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    *,
    val_size: float = 0.20,
    test_size: float = 0.20,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
):
    """
    Split data into training, validation, and test sets.

    The sizes are interpreted as proportions of the original dataset.

    Example
    -------
    If val_size=0.2 and test_size=0.2, then approximately:

    - 60% training
    - 20% validation
    - 20% test
    """
    val_size = _check_fraction(val_size, "val_size")
    test_size = _check_fraction(test_size, "test_size")

    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be less than 1.")

    if y is not None:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify,
        )

        relative_val_size = val_size / (1.0 - test_size)

        stratify_temp = y_temp if stratify is not None else None

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=relative_val_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_temp,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    X_temp, X_test = train_test_split(
        X,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    relative_val_size = val_size / (1.0 - test_size)

    X_train, X_val = train_test_split(
        X_temp,
        test_size=relative_val_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    return X_train, X_val, X_test
