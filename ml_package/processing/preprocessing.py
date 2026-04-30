"""
This module implements three regression models from scratch using NumPy:

    - LinearRegression: ordinary least squares regression
    - RidgeRegression: L2-regularized linear regression
    - LassoRegression: L1-regularized linear regression
"""


from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from ..preprocessing import ArrayLike, standardize
except ImportError: 
    from ml_package.preprocessing import ArrayLike, standardize


__all__ = [
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
]


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------

def _as_2d_numeric_array(X: ArrayLike, name: str = "X") -> np.ndarray:
    """
    Convert X to a non-empty 2D numeric NumPy array.

    This mirrors the expectations of the preprocessing module: features should
    be represented as a 2D numeric matrix with shape
    (n_samples, n_features).
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


def _as_1d_numeric_array(y: ArrayLike, name: str = "y") -> np.ndarray:
    """
    Convert y to a non-empty 1D numeric NumPy array.

    Linear, ridge, and lasso regression require numeric target values.
    """
    arr = np.asarray(y)

    if arr.ndim != 1:
        arr = arr.ravel()

    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")

    try:
        arr = arr.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"All elements of {name} must be numeric for regression.") from exc

    if np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values. Handle missing data first.")

    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values. Handle extreme data first.")

    return arr


def _validate_X_y(X: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate a feature matrix X and numeric target vector y.
    """
    X_arr = _as_2d_numeric_array(X, name="X")
    y_arr = _as_1d_numeric_array(y, name="y")

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


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R^2, the coefficient of determination.

    R^2 = 1 - SSE / SST
    """
    y_true = _as_1d_numeric_array(y_true, name="y_true")
    y_pred = _as_1d_numeric_array(y_pred, name="y_pred")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1.0 - ss_res / ss_tot)


def _soft_threshold(value: float, penalty: float) -> float:
    """
    Soft-thresholding operator for lasso regression.

    This is what allows lasso to shrink some coefficients exactly to zero.
    """
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


# -----------------------------------------------------------------------------
# Base model
# -----------------------------------------------------------------------------

class BaseLinearModel:
    """
    Shared functionality for linear, ridge, and lasso regression.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    standardize : bool, default=False
        Whether to standardize features internally using the package's
        preprocessing.standardize() function.

        If you already standardized X using preprocessing.standardize(), set
        standardize=False to avoid scaling twice.
    """

    def __init__(self, fit_intercept: bool = True, standardize: bool = False):
        self.fit_intercept = fit_intercept
        self.standardize = standardize

        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None
        self.is_fitted_: bool = False

        self._standardize_params: Optional[dict[str, np.ndarray]] = None
        self.n_features_in_: Optional[int] = None

    def _prepare_X_fit(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare X during model fitting.
        """
        self.n_features_in_ = X.shape[1]

        if self.standardize:
            X, self._standardize_params = standardize(X, return_params=True)

        if self.fit_intercept:
            X = _add_intercept_column(X)

        return X

    def _prepare_X_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Prepare X during prediction.
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

    def _store_parameters(self, beta: np.ndarray) -> None:
        """
        Store learned intercept and coefficient values.
        """
        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:].astype(float, copy=False)
        else:
            self.intercept_ = 0.0
            self.coef_ = beta.astype(float, copy=False)

        self.is_fitted_ = True

    def _full_beta(self) -> np.ndarray:
        """
        Return the full coefficient vector, including intercept if needed.
        """
        self._check_is_fitted()

        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model parameters are missing. Refit the model.")

        if self.fit_intercept:
            return np.concatenate(([self.intercept_], self.coef_))

        return self.coef_

    def _check_is_fitted(self) -> None:
        """
        Raise an error if fit() has not been called.
        """
        if not self.is_fitted_:
            raise ValueError("This model has not been fitted yet. Call fit(X, y) first.")

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict target values for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        X_design = self._prepare_X_predict(X)
        beta = self._full_beta()
        return X_design @ beta

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return the R^2 score on X and y.
        """
        y_arr = _as_1d_numeric_array(y, name="y")
        y_pred = self.predict(X)
        return _r2_score(y_arr, y_pred)

    def get_params(self) -> dict[str, object]:
        """
        Return basic model configuration parameters.
        """
        return {
            "fit_intercept": self.fit_intercept,
            "standardize": self.standardize,
        }


# -----------------------------------------------------------------------------
# Linear regression
# -----------------------------------------------------------------------------

class LinearRegression(BaseLinearModel):
    """
    Ordinary least squares linear regression.

    This model minimizes the residual sum of squares:

        sum((y_i - y_hat_i)^2)

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    standardize : bool, default=False
        Whether to standardize features internally.

        Linear regression does not require standardization, but it can still be
        useful for comparing coefficient sizes.
    """

    def __init__(self, fit_intercept: bool = True, standardize: bool = False):
        super().__init__(fit_intercept=fit_intercept, standardize=standardize)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit the ordinary least squares model.
        """
        X_arr, y_arr = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X_arr)

        # The pseudoinverse is more stable than manually computing
        # inverse(X.T @ X) @ X.T @ y, especially if columns are correlated.
        beta = np.linalg.pinv(X_design) @ y_arr

        self._store_parameters(beta)
        return self


# -----------------------------------------------------------------------------
# Ridge regression
# -----------------------------------------------------------------------------

class RidgeRegression(BaseLinearModel):
    """
    Ridge regression with L2 regularization.

    This model minimizes:

        sum((y_i - y_hat_i)^2) + alpha * sum(beta_j^2)

    The intercept is not regularized.

    Parameters
    ----------
    alpha : float, default=1.0
        Strength of L2 regularization. Must be nonnegative.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    standardize : bool, default=True
        Whether to standardize features internally.

        Ridge regression is sensitive to feature scale, so this defaults to
        True. If you already used preprocessing.standardize(), set this to
        False.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        standardize: bool = True,
    ):
        super().__init__(fit_intercept=fit_intercept, standardize=standardize)

        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        self.alpha = float(alpha)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RidgeRegression":
        """
        Fit the ridge regression model.
        """
        X_arr, y_arr = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X_arr)

        n_features = X_design.shape[1]
        penalty_matrix = np.eye(n_features)

        # Do not penalize the intercept.
        if self.fit_intercept:
            penalty_matrix[0, 0] = 0.0

        beta = (
            np.linalg.pinv(X_design.T @ X_design + self.alpha * penalty_matrix)
            @ X_design.T
            @ y_arr
        )

        self._store_parameters(beta)
        return self

    def get_params(self) -> dict[str, object]:
        """
        Return model configuration parameters.
        """
        params = super().get_params()
        params["alpha"] = self.alpha
        return params


# -----------------------------------------------------------------------------
# Lasso regression
# -----------------------------------------------------------------------------

class LassoRegression(BaseLinearModel):
    """
    Lasso regression with L1 regularization using coordinate descent.

    This model minimizes approximately:

        (1 / (2n)) * sum((y_i - y_hat_i)^2) + alpha * sum(abs(beta_j))

    The intercept is not regularized.

    Parameters
    ----------
    alpha : float, default=1.0
        Strength of L1 regularization. Must be nonnegative.
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    standardize : bool, default=True
        Whether to standardize features internally.

        Lasso regression is sensitive to feature scale, so this defaults to
        True. If you already used preprocessing.standardize(), set this to
        False.
    max_iter : int, default=1000
        Maximum number of coordinate descent iterations.
    tol : float, default=1e-6
        Convergence tolerance based on the largest coefficient change.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        standardize: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ):
        super().__init__(fit_intercept=fit_intercept, standardize=standardize)

        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_iter_: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LassoRegression":
        """
        Fit the lasso regression model using coordinate descent.
        """
        X_arr, y_arr = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X_arr)

        n_samples, n_features = X_design.shape
        beta = np.zeros(n_features)

        # Average squared column norm. Used in the coordinate update formula.
        column_norms = np.sum(X_design ** 2, axis=0) / n_samples

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(n_features):
                if column_norms[j] == 0:
                    continue

                # Partial residual for coordinate j:
                # remove current contribution of feature j, then update beta[j].
                y_pred = X_design @ beta
                residual = y_arr - y_pred + X_design[:, j] * beta[j]

                rho = np.sum(X_design[:, j] * residual) / n_samples

                # Do not regularize the intercept column.
                if self.fit_intercept and j == 0:
                    beta[j] = rho / column_norms[j]
                else:
                    beta[j] = _soft_threshold(rho, self.alpha) / column_norms[j]

            max_change = np.max(np.abs(beta - beta_old))
            self.n_iter_ = iteration + 1

            if max_change < self.tol:
                break

        self._store_parameters(beta)
        return self

    def get_params(self) -> dict[str, object]:
        """
        Return model configuration parameters.
        """
        params = super().get_params()
        params.update(
            {
                "alpha": self.alpha,
                "max_iter": self.max_iter,
                "tol": self.tol,
            }
        )
        return params


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [5.0, 5.0],
        ]
    )

    y = np.array([3.0, 3.5, 6.0, 6.5, 8.0])

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": RidgeRegression(alpha=1.0),
        "Lasso Regression": LassoRegression(alpha=0.1),
    }

    for name, model in models.items():
        model.fit(X, y)
        predictions = model.predict(X)

        print(f"\n{name}")
        print("Intercept:", model.intercept_)
        print("Coefficients:", model.coef_)
        print("Predictions:", predictions)
        print("R^2:", model.score(X, y))
