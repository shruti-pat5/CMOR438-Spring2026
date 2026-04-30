"""
This module contains a gradient-descent optimizer for logistic regression that
can handle:

    - binary classification using the sigmoid function
    - multiclass classification using multinomial softmax regression
    - multiclass classification using one-vs-rest logistic regression
"""

from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from ..preprocessing import ArrayLike, standardize
except ImportError:  # pragma: no cover
    from ml_package.preprocessing import ArrayLike, standardize


__all__ = ["LogisticRegression"]


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _as_2d_numeric_array(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert X to a non-empty 2D numeric NumPy array.

    This mirrors the preprocessing module's expectation that features should be
    represented as a 2D numeric matrix with shape (n_samples, n_features).
    """
    arr = np.asarray(X)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

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


def _as_1d_array(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Convert target labels to a non-empty 1D NumPy array.

    Logistic regression can handle numeric or categorical labels, so this does
    not force float dtype.
    """
    arr = np.asarray(y)

    if arr.ndim != 1:
        arr = arr.ravel()

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    return arr


def _validate_X_y(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate a feature matrix X and target vector y.
    """
    X_arr = _as_2d_numeric_array(X, name="X")
    y_arr = _as_1d_array(y, name="y")

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            "X and y must have the same number of rows; "
            f"got X.shape[0]={X_arr.shape[0]} and len(y)={len(y_arr)}."
        )

    return X_arr, y_arr


def _add_intercept_column(X: np.ndarray) -> np.ndarray:
    """
    Add a column of ones to X for the intercept term.
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def _one_hot(y_encoded: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Convert integer class labels to a one-hot encoded matrix.
    """
    Y = np.zeros((y_encoded.shape[0], n_classes))
    Y[np.arange(y_encoded.shape[0]), y_encoded] = 1.0
    return Y


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid function.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)

    positive = z >= 0
    negative = ~positive

    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[negative])
    out[negative] = exp_z / (1.0 + exp_z)

    return out


def _softmax(Z: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax function.
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)


def _binary_log_loss(y_true_01: np.ndarray, prob_pos: np.ndarray, eps: float = 1e-15) -> float:
    """
    Binary cross-entropy loss.
    """
    prob_pos = np.clip(prob_pos, eps, 1.0 - eps)
    loss = -np.mean(
        y_true_01 * np.log(prob_pos)
        + (1.0 - y_true_01) * np.log(1.0 - prob_pos)
    )
    return float(loss)


def _multiclass_log_loss(Y_one_hot: np.ndarray, probs: np.ndarray, eps: float = 1e-15) -> float:
    """
    Multiclass cross-entropy loss.
    """
    probs = np.clip(probs, eps, 1.0 - eps)
    loss = -np.mean(np.sum(Y_one_hot * np.log(probs), axis=1))
    return float(loss)


def _accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Classification accuracy.
    """
    y_true = _as_1d_array(y_true, name="y_true")
    y_pred = _as_1d_array(y_pred, name="y_pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    return float(np.mean(y_true == y_pred))


# -----------------------------------------------------------------------------
# Logistic regression
# -----------------------------------------------------------------------------

class LogisticRegression:
    """
    Logistic regression classifier trained with gradient descent.

    This class supports three classification modes:

    1. Binary classification using sigmoid regression.
    2. Multiclass classification using multinomial softmax regression.
    3. Multiclass classification using one-vs-rest sigmoid models.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Gradient descent step size.
    max_iter : int, default=1000
        Maximum number of gradient descent iterations.
    tol : float, default=1e-6
        Convergence tolerance based on change in loss.
    fit_intercept : bool, default=True
        Whether to add an intercept column.
    standardize : bool, default=True
        Whether to standardize features internally using
        preprocessing.standardize().

        If you already standardized X using preprocessing.standardize(), set
        standardize=False to avoid scaling twice.
    multi_class : {"auto", "binary", "multinomial", "ovr"}, default="auto"
        Classification strategy.

        - "auto": use binary sigmoid when there are 2 classes, otherwise
          multinomial softmax.
        - "binary": force binary sigmoid classification. Requires exactly
          2 classes.
        - "multinomial": use softmax regression. Works for 2 or more classes.
        - "ovr": train one binary classifier per class.
    l2_penalty : float, default=0.0
        Optional L2 regularization strength. The intercept is not regularized.
    random_state : int, optional
        Seed used for reproducible coefficient initialization.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
        fit_intercept: bool = True,
        standardize: bool = True,
        multi_class: str = "auto",
        l2_penalty: float = 0.0,
        random_state: Optional[int] = None,
    ):
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")
        if multi_class not in {"auto", "binary", "multinomial", "ovr"}:
            raise ValueError(
                "multi_class must be one of: 'auto', 'binary', 'multinomial', 'ovr'."
            )
        if l2_penalty < 0:
            raise ValueError("l2_penalty must be nonnegative.")
        if random_state is not None and not isinstance(random_state, (int, np.integer)):
            raise TypeError("random_state must be an integer or None.")

        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.fit_intercept = fit_intercept
        self.standardize = standardize
        self.multi_class = multi_class
        self.l2_penalty = float(l2_penalty)
        self.random_state = random_state

        self.classes_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray | float] = None
        self.is_fitted_: bool = False
        self.n_features_in_: Optional[int] = None
        self.n_iter_: int = 0
        self.loss_history_: list[float] = []
        self.mode_: Optional[str] = None

        self._standardize_params: Optional[dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_X_fit(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare feature matrix during fitting.
        """
        self.n_features_in_ = X.shape[1]

        if self.standardize:
            X, self._standardize_params = standardize(X, return_params=True)

        if self.fit_intercept:
            X = _add_intercept_column(X)

        return X

    def _prepare_X_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Prepare feature matrix during prediction.
        """
        self._check_is_fitted()
        X_arr = _as_2d_numeric_array(X, name="X")

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has the wrong number of features; "
                f"expected {self.n_features_in_}, got {X_arr.shape[1]}."
            )

        if self.standardize:
            if self._standardize_params is None:
                raise ValueError("Missing standardization parameters. Refit the model.")

            X_arr = standardize(
                X_arr,
                mean=self._standardize_params["mean"],
                scale=self._standardize_params["scale"],
            )

        if self.fit_intercept:
            X_arr = _add_intercept_column(X_arr)

        return X_arr

    def _check_is_fitted(self) -> None:
        """
        Raise an error if fit() has not been called.
        """
        if not self.is_fitted_:
            raise ValueError("This model has not been fitted yet. Call fit(X, y) first.")

    def _resolve_mode(self, n_classes: int) -> str:
        """
        Determine which classification strategy to use.
        """
        if n_classes < 2:
            raise ValueError("Logistic regression requires at least two classes.")

        if self.multi_class == "auto":
            return "binary" if n_classes == 2 else "multinomial"

        if self.multi_class == "binary" and n_classes != 2:
            raise ValueError("multi_class='binary' requires exactly two classes.")

        return self.multi_class

    def _regularization_gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute L2 regularization gradient without penalizing intercept.
        """
        grad = self.l2_penalty * weights

        if self.fit_intercept:
            if grad.ndim == 1:
                grad[0] = 0.0
            else:
                grad[0, :] = 0.0

        return grad

    def _regularization_loss(self, weights: np.ndarray) -> float:
        """
        Compute L2 regularization loss without penalizing intercept.
        """
        if self.l2_penalty == 0.0:
            return 0.0

        if self.fit_intercept:
            weights_no_intercept = weights[1:] if weights.ndim == 1 else weights[1:, :]
        else:
            weights_no_intercept = weights

        return float(0.5 * self.l2_penalty * np.sum(weights_no_intercept ** 2))

    # ------------------------------------------------------------------
    # Fitting methods
    # ------------------------------------------------------------------

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Fit logistic regression using gradient descent.
        """
        X_arr, y_arr = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X_arr)

        self.classes_, y_encoded = np.unique(y_arr, return_inverse=True)
        n_classes = len(self.classes_)
        self.mode_ = self._resolve_mode(n_classes)

        self.loss_history_ = []
        self.n_iter_ = 0

        if self.mode_ == "binary":
            self._fit_binary(X_design, y_encoded)
        elif self.mode_ == "multinomial":
            self._fit_multinomial(X_design, y_encoded, n_classes)
        elif self.mode_ == "ovr":
            self._fit_ovr(X_design, y_encoded, n_classes)
        else:  # pragma: no cover
            raise RuntimeError("Unexpected logistic regression mode.")

        self.is_fitted_ = True
        return self

    def _fit_binary(self, X: np.ndarray, y_encoded: np.ndarray) -> None:
        """
        Fit binary logistic regression using sigmoid probabilities.
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(loc=0.0, scale=0.01, size=n_features)

        y_binary = y_encoded.astype(float)
        previous_loss = np.inf

        for iteration in range(self.max_iter):
            logits = X @ weights
            prob_pos = _sigmoid(logits)

            error = prob_pos - y_binary
            gradient = (X.T @ error) / n_samples
            gradient += self._regularization_gradient(weights)

            weights -= self.learning_rate * gradient

            loss = _binary_log_loss(y_binary, prob_pos)
            loss += self._regularization_loss(weights)
            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if abs(previous_loss - loss) < self.tol:
                break

            previous_loss = loss

        if self.fit_intercept:
            self.intercept_ = float(weights[0])
            self.coef_ = weights[1:].reshape(1, -1)
        else:
            self.intercept_ = 0.0
            self.coef_ = weights.reshape(1, -1)

    def _fit_multinomial(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        n_classes: int,
    ) -> None:
        """
        Fit multinomial logistic regression using softmax probabilities.
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))

        Y = _one_hot(y_encoded, n_classes)
        previous_loss = np.inf

        for iteration in range(self.max_iter):
            logits = X @ weights
            probs = _softmax(logits)

            error = probs - Y
            gradient = (X.T @ error) / n_samples
            gradient += self._regularization_gradient(weights)

            weights -= self.learning_rate * gradient

            loss = _multiclass_log_loss(Y, probs)
            loss += self._regularization_loss(weights)
            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if abs(previous_loss - loss) < self.tol:
                break

            previous_loss = loss

        if self.fit_intercept:
            self.intercept_ = weights[0, :].astype(float, copy=False)
            self.coef_ = weights[1:, :].T.astype(float, copy=False)
        else:
            self.intercept_ = np.zeros(n_classes)
            self.coef_ = weights.T.astype(float, copy=False)

    def _fit_ovr(
        self,
        X: np.ndarray,
        y_encoded: np.ndarray,
        n_classes: int,
    ) -> None:
        """
        Fit one-vs-rest logistic regression.

        This trains one binary classifier per class. For class k, the target is:

            1 if y == k, else 0
        """
        n_samples, n_features = X.shape
        rng = np.random.default_rng(self.random_state)
        weights = rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))

        Y = _one_hot(y_encoded, n_classes)
        previous_loss = np.inf

        for iteration in range(self.max_iter):
            logits = X @ weights
            probs = _sigmoid(logits)

            error = probs - Y
            gradient = (X.T @ error) / n_samples
            gradient += self._regularization_gradient(weights)

            weights -= self.learning_rate * gradient

            # OvR loss is the average binary loss across all class classifiers.
            loss = 0.0
            for class_id in range(n_classes):
                loss += _binary_log_loss(Y[:, class_id], probs[:, class_id])
            loss /= n_classes
            loss += self._regularization_loss(weights)

            self.loss_history_.append(loss)
            self.n_iter_ = iteration + 1

            if abs(previous_loss - loss) < self.tol:
                break

            previous_loss = loss

        if self.fit_intercept:
            self.intercept_ = weights[0, :].astype(float, copy=False)
            self.coef_ = weights[1:, :].T.astype(float, copy=False)
        else:
            self.intercept_ = np.zeros(n_classes)
            self.coef_ = weights.T.astype(float, copy=False)

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    def _full_weights(self) -> np.ndarray:
        """
        Return model weights in matrix/vector form including intercept.
        """
        self._check_is_fitted()

        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model parameters are missing. Refit the model.")

        if self.mode_ == "binary":
            coef = self.coef_.ravel()

            if self.fit_intercept:
                return np.concatenate(([float(self.intercept_)], coef))

            return coef

        if self.fit_intercept:
            intercept = np.asarray(self.intercept_, dtype=float).reshape(1, -1)
            return np.vstack((intercept, self.coef_.T))

        return self.coef_.T

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class probabilities for X.

        Returns
        -------
        np.ndarray, shape (n_samples, n_classes)
            Predicted probabilities for each class in self.classes_ order.
        """
        X_design = self._prepare_X_predict(X)
        weights = self._full_weights()

        if self.mode_ == "binary":
            prob_pos = _sigmoid(X_design @ weights)
            prob_neg = 1.0 - prob_pos
            return np.column_stack((prob_neg, prob_pos))

        if self.mode_ == "multinomial":
            return _softmax(X_design @ weights)

        if self.mode_ == "ovr":
            raw_probs = _sigmoid(X_design @ weights)
            row_sums = raw_probs.sum(axis=1, keepdims=True)

            # Normalize OvR scores so each row sums to 1. If all scores are zero,
            # fall back to a uniform distribution.
            zero_rows = row_sums.ravel() == 0.0
            row_sums[zero_rows] = 1.0
            probs = raw_probs / row_sums

            if np.any(zero_rows):
                probs[zero_rows, :] = 1.0 / raw_probs.shape[1]

            return probs

        raise RuntimeError("Unexpected logistic regression mode.")  # pragma: no cover

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels for X.
        """
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes_[class_indices]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return classification accuracy on X and y.
        """
        y_arr = _as_1d_array(y, name="y")
        y_pred = self.predict(X)
        return _accuracy_score(y_arr, y_pred)

    def get_params(self) -> dict[str, object]:
        """
        Return model configuration parameters.
        """
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "fit_intercept": self.fit_intercept,
            "standardize": self.standardize,
            "multi_class": self.multi_class,
            "l2_penalty": self.l2_penalty,
            "random_state": self.random_state,
        }


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    X = np.array(
        [
            [1.0, 2.0],
            [1.5, 1.8],
            [5.0, 8.0],
            [6.0, 9.0],
            [1.0, 0.6],
            [9.0, 11.0],
        ]
    )

    y_binary = np.array([0, 0, 1, 1, 0, 1])

    binary_model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        multi_class="binary",
        random_state=42,
    )

    binary_model.fit(X, y_binary)

    print("Binary logistic regression")
    print("Classes:", binary_model.classes_)
    print("Intercept:", binary_model.intercept_)
    print("Coefficients:", binary_model.coef_)
    print("Predicted probabilities:\n", binary_model.predict_proba(X))
    print("Predictions:", binary_model.predict(X))
    print("Accuracy:", binary_model.score(X, y_binary))

    y_multiclass = np.array(["A", "A", "B", "B", "C", "C"])

    softmax_model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        multi_class="multinomial",
        random_state=42,
    )

    softmax_model.fit(X, y_multiclass)

    print("\nMultinomial softmax logistic regression")
    print("Classes:", softmax_model.classes_)
    print("Intercept:", softmax_model.intercept_)
    print("Coefficients:\n", softmax_model.coef_)
    print("Predicted probabilities:\n", softmax_model.predict_proba(X))
    print("Predictions:", softmax_model.predict(X))
    print("Accuracy:", softmax_model.score(X, y_multiclass))

    ovr_model = LogisticRegression(
        learning_rate=0.1,
        max_iter=1000,
        multi_class="ovr",
        random_state=42,
    )

    ovr_model.fit(X, y_multiclass)

    print("\nOne-vs-rest logistic regression")
    print("Classes:", ovr_model.classes_)
    print("Intercept:", ovr_model.intercept_)
    print("Coefficients:\n", ovr_model.coef_)
    print("Predicted probabilities:\n", ovr_model.predict_proba(X))
    print("Predictions:", ovr_model.predict(X))
    print("Accuracy:", ovr_model.score(X, y_multiclass))
