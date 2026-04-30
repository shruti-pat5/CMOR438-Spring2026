"""
This module implements Linear Regression, Ridge Regression, and Lasso Regression
from scratch using NumPy. The classes follow a simple scikit-learn-like API:

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = model.score(X_test, y_test)

Models included:
    - LinearRegression: ordinary least squares using the closed-form solution
    - RidgeRegression: L2-regularized regression using the closed-form solution
    - LassoRegression: L1-regularized regression using coordinate descent
"""

from __future__ import annotations

import numpy as np


# Validation
def _validate_X_y(X, y):
    """
    Validate feature matrix X and target vector y.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    X : np.ndarray
        Validated 2D feature matrix.
    y : np.ndarray
        Validated 1D target vector.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array or convertible to a 2D array.")

    if y.ndim != 1:
        y = y.ravel()

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}."
        )

    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("X and y cannot contain NaN values.")

    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("X and y cannot contain infinite values.")

    return X, y


def _validate_X(X):
    """
    Validate a feature matrix used for prediction.
    """
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array or convertible to a 2D array.")

    if np.isnan(X).any():
        raise ValueError("X cannot contain NaN values.")

    if np.isinf(X).any():
        raise ValueError("X cannot contain infinite values.")

    return X


def _add_intercept_column(X):
    """
    Add a column of ones to X for the intercept term.
    """
    ones = np.ones((X.shape[0], 1))
    return np.hstack((ones, X))


def _standardize_fit(X):
    """
    Compute column-wise mean and standard deviation, then standardize X.

    Standardization formula:
        z = (x - mean) / std

    Columns with zero standard deviation are left unchanged by using std = 1.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std_safe = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std_safe
    return X_scaled, mean, std_safe


def _standardize_transform(X, mean, std):
    """
    Standardize X using previously computed mean and standard deviation.
    """
    return (X - mean) / std


def _r2_score(y_true, y_pred):
    """
    Compute the coefficient of determination R^2.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def _soft_threshold(value, penalty):
    """
    Soft-thresholding operator used in lasso coordinate descent.

    This shrinks coefficients toward zero and can set them exactly equal to zero.
    """
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


# -----------------------------------------------------------------------------
# Base class
# -----------------------------------------------------------------------------

class BaseLinearModel:
    """
    Shared functionality for linear regression models.
    """

    def __init__(self, fit_intercept=True, standardize=False):
        self.fit_intercept = fit_intercept
        self.standardize = standardize

        self.coef_ = None
        self.intercept_ = None
        self.is_fitted_ = False

        self._X_mean = None
        self._X_std = None

    def _prepare_X_fit(self, X):
        """
        Prepare X during fitting.
        """
        if self.standardize:
            X, self._X_mean, self._X_std = _standardize_fit(X)

        if self.fit_intercept:
            X = _add_intercept_column(X)

        return X

    def _prepare_X_predict(self, X):
        """
        Prepare X during prediction.
        """
        X = _validate_X(X)

        if self.standardize:
            X = _standardize_transform(X, self._X_mean, self._X_std)

        if self.fit_intercept:
            X = _add_intercept_column(X)

        return X

    def _store_parameters(self, beta):
        """
        Store intercept and coefficients after fitting.
        """
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        self.is_fitted_ = True

    def _check_is_fitted(self):
        """
        Make sure model has been fitted before prediction or scoring.
        """
        if not self.is_fitted_:
            raise ValueError("This model has not been fitted yet. Call fit(X, y) first.")

    def predict(self, X):
        """
        Predict target values for X.
        """
        self._check_is_fitted()
        X = self._prepare_X_predict(X)

        if self.fit_intercept:
            beta = np.concatenate(([self.intercept_], self.coef_))
        else:
            beta = self.coef_

        return X @ beta

    def score(self, X, y):
        """
        Return the R^2 score of the model on X and y.
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        return _r2_score(y, y_pred)


# Linear Regression
class LinearRegression(BaseLinearModel):
    """
    Ordinary least squares linear regression.

    This model minimizes:
        sum((y_i - y_hat_i)^2)

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term.
    standardize : bool, default=False
        Whether to standardize features before fitting.
    """

    def __init__(self, fit_intercept=True, standardize=False):
        super().__init__(fit_intercept=fit_intercept, standardize=standardize)

    def fit(self, X, y):
        """
        Fit the linear regression model.
        """
        X, y = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X)

        # Pseudoinverse is more numerically stable than directly inverting X^T X.
        beta = np.linalg.pinv(X_design) @ y

        self._store_parameters(beta)
        return self


# Ridge Regression
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
        Whether to standardize features before fitting.
    """

    def __init__(self, alpha=1.0, fit_intercept=True, standardize=True):
        super().__init__(fit_intercept=fit_intercept, standardize=standardize)

        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")

        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the ridge regression model.
        """
        X, y = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X)

        n_features = X_design.shape[1]
        penalty_matrix = np.eye(n_features)

        # Do not penalize the intercept term.
        if self.fit_intercept:
            penalty_matrix[0, 0] = 0.0

        beta = np.linalg.pinv(
            X_design.T @ X_design + self.alpha * penalty_matrix
        ) @ X_design.T @ y

        self._store_parameters(beta)
        return self


# Lasso Regression
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
        Whether to standardize features before fitting.
    max_iter : int, default=1000
        Maximum number of coordinate descent passes.
    tol : float, default=1e-6
        Convergence tolerance based on coefficient changes.
    """

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,
        standardize=True,
        max_iter=1000,
        tol=1e-6,
    ):
        super().__init__(fit_intercept = fit_intercept, standardize = standardize)

        if alpha < 0:
            raise ValueError("alpha must be nonnegative.")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if tol <= 0:
            raise ValueError("tol must be positive.")

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.n_iter_ = 0

    def fit(self, X, y):
        """
        Fit the lasso regression model using coordinate descent.
        """
        X, y = _validate_X_y(X, y)
        X_design = self._prepare_X_fit(X)

        n_samples, n_features = X_design.shape
        beta = np.zeros(n_features)

        # Precompute squared column norms divided by n for efficiency.
        column_norms = np.sum(X_design ** 2, axis=0) / n_samples

        for iteration in range(self.max_iter):
            beta_old = beta.copy()

            for j in range(n_features):
                if column_norms[j] == 0:
                    continue

                # Compute partial residual excluding feature j.
                y_pred = X_design @ beta
                residual = y - y_pred + X_design[:, j] * beta[j]

                rho = np.sum(X_design[:, j] * residual) / n_samples

                # Do not regularize intercept.
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


# Example dataset
if __name__ == "__main__":
    # Small example dataset
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 5.0],
    ])

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
