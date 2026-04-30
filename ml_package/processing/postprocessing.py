"""
Postprocessing and evaluation utilities for the rice_ml package.

This module provides:
- Classification metrics
- Regression metrics
- Confusion matrix construction
- ROC AUC and log loss
- Simple model output aggregation helpers

The functions are intentionally lightweight and NumPy-only so they can be used
throughout the custom machine learning package without relying on scikit-learn.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union
import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int], Sequence[Sequence[float]]]

__all__ = [
    "ArrayLike",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "confusion_matrix",
    "roc_auc_score",
    "log_loss",
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "r2_score",
    "majority_vote",
    "weighted_average",
    "distance_weighted_average",
]

# Validation helpers
def _as_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a non-empty 1D NumPy array."""
    arr = np.asarray(x)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array; got {arr.ndim}D.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr


def _as_numeric_1d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a non-empty 1D numeric NumPy array."""
    arr = _as_1d_array(x, name)

    try:
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values.")

    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values.")

    return arr


def _as_numeric_2d_array(x: ArrayLike, name: str) -> np.ndarray:
    """Convert input to a non-empty 2D numeric NumPy array."""
    arr = np.asarray(x)

    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    try:
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values.") from exc

    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values.")

    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values.")

    return arr


def _check_same_length(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Check that y_true and y_pred have the same length."""
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length; "
            f"got {y_true.shape[0]} and {y_pred.shape[0]}."
        )


def _check_average(average: str) -> str:
    """Validate averaging method for multiclass metrics."""
    valid = {"binary", "macro", "micro", "weighted"}

    if average not in valid:
        raise ValueError(f"average must be one of {valid}; got {average!r}.")

    return average


def _safe_divide(numerator: float, denominator: float, zero_division: float = 0.0) -> float:
    """Divide safely, returning zero_division when denominator is zero."""
    if denominator == 0:
        return float(zero_division)

    return float(numerator / denominator)


def _unique_labels(y_true: np.ndarray, y_pred: Optional[np.ndarray] = None) -> np.ndarray:
    """Return sorted unique labels from y_true and optionally y_pred."""
    if y_pred is None:
        return np.unique(y_true)

    return np.unique(np.concatenate([y_true, y_pred]))


# Classification metrics
def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute classification accuracy.

    Accuracy is the proportion of predictions that are exactly correct.

    Formula
    -------
        accuracy = number of correct predictions / number of total predictions
    """
    y_true_arr = _as_1d_array(y_true, "y_true")
    y_pred_arr = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    return float(np.mean(y_true_arr == y_pred_arr))


def confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence] = None,
) -> np.ndarray:
    """
    Compute a confusion matrix for classification.

    Rows represent true classes.
    Columns represent predicted classes.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : sequence, optional
        Class label order. If None, labels are inferred and sorted.

    Returns
    -------
    np.ndarray
        Matrix where entry [i, j] counts observations with true class i
        and predicted class j.
    """
    y_true_arr = _as_1d_array(y_true, "y_true")
    y_pred_arr = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    labels_arr = np.asarray(labels) if labels is not None else _unique_labels(y_true_arr, y_pred_arr)

    label_to_index = {label: idx for idx, label in enumerate(labels_arr)}
    matrix = np.zeros((len(labels_arr), len(labels_arr)), dtype=int)

    for true_label, pred_label in zip(y_true_arr, y_pred_arr):
        if true_label not in label_to_index or pred_label not in label_to_index:
            raise ValueError("y_true and y_pred contain labels not included in labels.")

        i = label_to_index[true_label]
        j = label_to_index[pred_label]
        matrix[i, j] += 1

    return matrix


def _precision_recall_f1_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: np.ndarray,
    zero_division: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision, recall, F1, and support for each class."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    true_positive = np.diag(cm).astype(float)
    predicted_positive = cm.sum(axis=0).astype(float)
    actual_positive = cm.sum(axis=1).astype(float)

    precision = np.array([
        _safe_divide(tp, pp, zero_division)
        for tp, pp in zip(true_positive, predicted_positive)
    ])

    recall = np.array([
        _safe_divide(tp, ap, zero_division)
        for tp, ap in zip(true_positive, actual_positive)
    ])

    f1 = np.array([
        _safe_divide(2 * p * r, p + r, zero_division)
        for p, r in zip(precision, recall)
    ])

    support = actual_positive

    return precision, recall, f1, support


def _average_metric(
    values: np.ndarray,
    support: np.ndarray,
    average: str,
) -> float:
    """Average per-class metric values."""
    if average == "macro":
        return float(np.mean(values))

    if average == "weighted":
        total = support.sum()
        if total == 0:
            return 0.0
        return float(np.sum(values * support) / total)

    raise ValueError("_average_metric only supports macro and weighted.")


def precision_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: str = "binary",
    pos_label=1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute precision for classification.

    Precision answers:

        Of the observations predicted as positive, how many were correct?

    For multiclass classification, use average="macro", "micro", or "weighted".
    """
    average = _check_average(average)

    y_true_arr = _as_1d_array(y_true, "y_true")
    y_pred_arr = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    labels = _unique_labels(y_true_arr, y_pred_arr)

    if average == "binary":
        true_positive = np.sum((y_true_arr == pos_label) & (y_pred_arr == pos_label))
        false_positive = np.sum((y_true_arr != pos_label) & (y_pred_arr == pos_label))
        return _safe_divide(true_positive, true_positive + false_positive, zero_division)

    if average == "micro":
        return accuracy_score(y_true_arr, y_pred_arr)

    precision, _, _, support = _precision_recall_f1_per_class(
        y_true_arr,
        y_pred_arr,
        labels,
        zero_division,
    )

    return _average_metric(precision, support, average)


def recall_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: str = "binary",
    pos_label=1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute recall for classification.

    Recall answers:

        Of the actual positive observations, how many were correctly found?

    For multiclass classification, use average="macro", "micro", or "weighted".
    """
    average = _check_average(average)

    y_true_arr = _as_1d_array(y_true, "y_true")
    y_pred_arr = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    labels = _unique_labels(y_true_arr, y_pred_arr)

    if average == "binary":
        true_positive = np.sum((y_true_arr == pos_label) & (y_pred_arr == pos_label))
        false_negative = np.sum((y_true_arr == pos_label) & (y_pred_arr != pos_label))
        return _safe_divide(true_positive, true_positive + false_negative, zero_division)

    if average == "micro":
        return accuracy_score(y_true_arr, y_pred_arr)

    _, recall, _, support = _precision_recall_f1_per_class(
        y_true_arr,
        y_pred_arr,
        labels,
        zero_division,
    )

    return _average_metric(recall, support, average)


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    average: str = "binary",
    pos_label=1,
    zero_division: float = 0.0,
) -> float:
    """
    Compute F1 score for classification.

    F1 is the harmonic mean of precision and recall.

    For multiclass classification, use average="macro", "micro", or "weighted".
    """
    average = _check_average(average)

    y_true_arr = _as_1d_array(y_true, "y_true")
    y_pred_arr = _as_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    labels = _unique_labels(y_true_arr, y_pred_arr)

    if average == "binary":
        precision = precision_score(
            y_true_arr,
            y_pred_arr,
            average="binary",
            pos_label=pos_label,
            zero_division=zero_division,
        )
        recall = recall_score(
            y_true_arr,
            y_pred_arr,
            average="binary",
            pos_label=pos_label,
            zero_division=zero_division,
        )
        return _safe_divide(2 * precision * recall, precision + recall, zero_division)

    if average == "micro":
        return accuracy_score(y_true_arr, y_pred_arr)

    _, _, f1, support = _precision_recall_f1_per_class(
        y_true_arr,
        y_pred_arr,
        labels,
        zero_division,
    )

    return _average_metric(f1, support, average)


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike, *, pos_label=1) -> float:
    """
    Compute ROC AUC for binary classification.

    y_score should contain predicted probabilities or decision scores for the
    positive class.

    This implementation uses the rank-sum / Mann-Whitney formulation.
    """
    y_true_arr = _as_1d_array(y_true, "y_true")
    y_score_arr = _as_numeric_1d_array(y_score, "y_score")
    _check_same_length(y_true_arr, y_score_arr)

    positive = y_true_arr == pos_label
    negative = ~positive

    n_positive = int(np.sum(positive))
    n_negative = int(np.sum(negative))

    if n_positive == 0 or n_negative == 0:
        raise ValueError("ROC AUC is undefined when only one class is present.")

    order = np.argsort(y_score_arr)
    sorted_scores = y_score_arr[order]

    ranks = np.empty_like(sorted_scores, dtype=float)
    start = 0

    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        average_rank = (start + 1 + end) / 2.0
        ranks[start:end] = average_rank
        start = end

    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks

    rank_sum_positive = np.sum(original_ranks[positive])
    auc = (rank_sum_positive - n_positive * (n_positive + 1) / 2.0) / (
        n_positive * n_negative
    )

    return float(auc)


def log_loss(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    labels: Optional[Sequence] = None,
    eps: float = 1e-15,
) -> float:
    """
    Compute logistic loss / cross-entropy loss.

    Supports:
    - Binary case: y_prob is 1D with probability of the positive class.
    - Multiclass case: y_prob is 2D with one probability column per class.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_prob : array-like
        Predicted probabilities.
    labels : sequence, optional
        Class label order for multiclass probabilities.
    eps : float, default=1e-15
        Probability clipping value to avoid log(0).
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    y_true_arr = _as_1d_array(y_true, "y_true")
    prob_arr = np.asarray(y_prob, dtype=float)

    if prob_arr.ndim == 1:
        if y_true_arr.shape[0] != prob_arr.shape[0]:
            raise ValueError("y_true and y_prob must have the same number of rows.")

        classes = np.unique(y_true_arr)

        if len(classes) > 2:
            raise ValueError("1D y_prob can only be used for binary log loss.")

        if labels is None:
            if len(classes) == 1:
                raise ValueError("labels must be provided when y_true has one class.")
            negative_label, positive_label = classes[0], classes[1]
        else:
            labels_arr = np.asarray(labels)
            if len(labels_arr) != 2:
                raise ValueError("Binary log loss requires exactly two labels.")
            negative_label, positive_label = labels_arr[0], labels_arr[1]

        y_binary = (y_true_arr == positive_label).astype(float)
        p = np.clip(prob_arr, eps, 1.0 - eps)

        loss = -(y_binary * np.log(p) + (1.0 - y_binary) * np.log(1.0 - p))
        return float(np.mean(loss))

    if prob_arr.ndim == 2:
        if y_true_arr.shape[0] != prob_arr.shape[0]:
            raise ValueError("y_true and y_prob must have the same number of rows.")

        if labels is None:
            labels_arr = np.unique(y_true_arr)
        else:
            labels_arr = np.asarray(labels)

        if prob_arr.shape[1] != len(labels_arr):
            raise ValueError("Number of probability columns must match number of labels.")

        label_to_index = {label: idx for idx, label in enumerate(labels_arr)}

        true_indices = []
        for label in y_true_arr:
            if label not in label_to_index:
                raise ValueError("y_true contains labels not included in labels.")
            true_indices.append(label_to_index[label])

        true_indices = np.asarray(true_indices)

        p = np.clip(prob_arr, eps, 1.0)
        row_sums = p.sum(axis=1, keepdims=True)
        p = p / row_sums

        chosen_prob = p[np.arange(len(y_true_arr)), true_indices]
        return float(-np.mean(np.log(chosen_prob)))

    raise ValueError("y_prob must be either a 1D or 2D array.")


# Regression Metrics
def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute mean squared error.

    MSE penalizes larger errors more heavily because errors are squared.
    """
    y_true_arr = _as_numeric_1d_array(y_true, "y_true")
    y_pred_arr = _as_numeric_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def root_mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute root mean squared error.

    RMSE is in the same units as the target variable.
    """
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute mean absolute error.

    MAE is the average absolute prediction error.
    """
    y_true_arr = _as_numeric_1d_array(y_true, "y_true")
    y_pred_arr = _as_numeric_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute R-squared.

    R-squared measures the proportion of variance in y_true explained by y_pred.

    A value of 1 is perfect prediction.
    A value of 0 means the model is no better than predicting the mean.
    Negative values are possible for poor models.
    """
    y_true_arr = _as_numeric_1d_array(y_true, "y_true")
    y_pred_arr = _as_numeric_1d_array(y_pred, "y_pred")
    _check_same_length(y_true_arr, y_pred_arr)

    ss_residual = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_total = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)

    if ss_total == 0:
        raise ValueError("R-squared is undefined when y_true is constant.")

    return float(1.0 - ss_residual / ss_total)

# Decision Aggregate Helpers
def majority_vote(predictions: ArrayLike) -> np.ndarray:
    """
    Aggregate predictions using majority voting.

    Parameters
    ----------
    predictions : array-like, shape (n_models, n_samples)
        Each row contains predictions from one model.

    Returns
    -------
    np.ndarray
        Final prediction for each sample.

    Notes
    -----
    In a tie, NumPy's sorted label order determines the winning class.
    """
    pred_arr = np.asarray(predictions)

    if pred_arr.ndim != 2:
        raise ValueError("predictions must be a 2D array with shape (n_models, n_samples).")

    if pred_arr.size == 0:
        raise ValueError("predictions must be non-empty.")

    final_predictions = []

    for column in pred_arr.T:
        labels, counts = np.unique(column, return_counts=True)
        final_predictions.append(labels[np.argmax(counts)])

    return np.asarray(final_predictions)


def weighted_average(values: ArrayLike, weights: Optional[ArrayLike] = None) -> np.ndarray:
    """
    Compute a weighted average across multiple model outputs.

    Parameters
    ----------
    values : array-like, shape (n_models, n_samples)
        Predictions or scores from multiple models.
    weights : array-like, optional, shape (n_models,)
        Model weights. If None, all models receive equal weight.

    Returns
    -------
    np.ndarray
        Weighted average prediction for each sample.
    """
    values_arr = _as_numeric_2d_array(values, "values")
    n_models = values_arr.shape[0]

    if weights is None:
        weights_arr = np.ones(n_models) / n_models
    else:
        weights_arr = _as_numeric_1d_array(weights, "weights")

        if weights_arr.shape[0] != n_models:
            raise ValueError("weights must have one value per model.")

        if np.any(weights_arr < 0):
            raise ValueError("weights must be nonnegative.")

        weight_sum = np.sum(weights_arr)

        if weight_sum == 0:
            raise ValueError("At least one weight must be positive.")

        weights_arr = weights_arr / weight_sum

    return np.average(values_arr, axis=0, weights=weights_arr)


def distance_weighted_average(
    values: ArrayLike,
    distances: ArrayLike,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute a distance-weighted average.

    This is useful for algorithms like k-nearest neighbors regression.

    Closer observations receive larger weights:

        weight = 1 / max(distance, eps)

    Parameters
    ----------
    values : array-like, shape (n_neighbors, n_samples)
        Neighbor target values or model outputs.
    distances : array-like, shape (n_neighbors, n_samples)
        Corresponding distances.
    eps : float, default=1e-12
        Small value used to avoid division by zero.

    Returns
    -------
    np.ndarray
        Distance-weighted average for each sample.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")

    values_arr = _as_numeric_2d_array(values, "values")
    distances_arr = _as_numeric_2d_array(distances, "distances")

    if values_arr.shape != distances_arr.shape:
        raise ValueError("values and distances must have the same shape.")

    if np.any(distances_arr < 0):
        raise ValueError("distances must be nonnegative.")

    weights = 1.0 / np.maximum(distances_arr, eps)
    weighted_sum = np.sum(values_arr * weights, axis=0)
    weight_sum = np.sum(weights, axis=0)

    return weighted_sum / weight_sum