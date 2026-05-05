"""
Unit tests for linear_regression.py

Covers:
    - Helper functions: _validate_X_y, _validate_X, _add_intercept_column,
      _standardize_fit, _standardize_transform, _r2_score, _soft_threshold
    - LinearRegression
    - RidgeRegression
    - LassoRegression
    - BaseLinearModel shared behaviour (predict before fit, score)

Run with:
    pytest linreg_test.py -v
"""

import numpy as np
import pytest

from supervised.linear_regression import (
    _validate_X_y,
    _validate_X,
    _add_intercept_column,
    _standardize_fit,
    _standardize_transform,
    _r2_score,
    _soft_threshold,
    LinearRegression,
    RidgeRegression,
    LassoRegression,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_X():
    """Small feature matrix used across many tests."""
    return np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0],
        [5.0, 5.0],
    ])


@pytest.fixture
def simple_y():
    """Target vector compatible with simple_X."""
    return np.array([3.0, 3.5, 6.0, 6.5, 8.0])


@pytest.fixture
def perfect_X():
    """X where y = 2*x + 1 exactly (perfect linear relationship)."""
    return np.arange(1, 11, dtype=float).reshape(-1, 1)


@pytest.fixture
def perfect_y(perfect_X):
    return 2.0 * perfect_X.ravel() + 1.0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestValidateXy:

    def test_valid_2d_X_1d_y(self, simple_X, simple_y):
        X_out, y_out = _validate_X_y(simple_X, simple_y)
        assert X_out.shape == (5, 2)
        assert y_out.shape == (5,)

    def test_1d_X_is_reshaped(self, simple_y):
        X_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        X_out, _ = _validate_X_y(X_1d, simple_y)
        assert X_out.ndim == 2
        assert X_out.shape == (5, 1)

    def test_2d_y_is_ravelled(self, simple_X):
        y_2d = np.array([[3.0], [3.5], [6.0], [6.5], [8.0]])
        _, y_out = _validate_X_y(simple_X, y_2d)
        assert y_out.ndim == 1

    def test_mismatched_samples_raises(self, simple_X):
        y_bad = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="same number of samples"):
            _validate_X_y(simple_X, y_bad)

    def test_nan_in_X_raises(self, simple_y):
        X_nan = np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0],
                           [4.0, 3.0], [5.0, 5.0]])
        with pytest.raises(ValueError, match="NaN"):
            _validate_X_y(X_nan, simple_y)

    def test_nan_in_y_raises(self, simple_X):
        y_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="NaN"):
            _validate_X_y(simple_X, y_nan)

    def test_inf_in_X_raises(self, simple_y):
        X_inf = np.array([[np.inf, 1.0], [2.0, 3.0], [3.0, 4.0],
                           [4.0, 3.0], [5.0, 5.0]])
        with pytest.raises(ValueError, match="infinite"):
            _validate_X_y(X_inf, simple_y)

    def test_inf_in_y_raises(self, simple_X):
        y_inf = np.array([1.0, np.inf, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="infinite"):
            _validate_X_y(simple_X, y_inf)

    def test_list_inputs_accepted(self, simple_y):
        X_list = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
        X_out, y_out = _validate_X_y(X_list, simple_y)
        assert isinstance(X_out, np.ndarray)
        assert X_out.dtype == float

    def test_output_dtype_is_float(self, simple_X, simple_y):
        X_int = simple_X.astype(int)
        y_int = simple_y.astype(int)
        X_out, y_out = _validate_X_y(X_int, y_int)
        assert X_out.dtype == float
        assert y_out.dtype == float


class TestValidateX:

    def test_valid_2d(self, simple_X):
        X_out = _validate_X(simple_X)
        assert X_out.shape == (5, 2)

    def test_1d_reshaped(self):
        X_1d = np.array([1.0, 2.0, 3.0])
        X_out = _validate_X(X_1d)
        assert X_out.shape == (3, 1)

    def test_nan_raises(self):
        X_nan = np.array([[1.0, np.nan]])
        with pytest.raises(ValueError, match="NaN"):
            _validate_X(X_nan)

    def test_inf_raises(self):
        X_inf = np.array([[np.inf, 1.0]])
        with pytest.raises(ValueError, match="infinite"):
            _validate_X(X_inf)


class TestAddInterceptColumn:

    def test_column_of_ones_prepended(self, simple_X):
        X_aug = _add_intercept_column(simple_X)
        assert X_aug.shape == (5, 3)
        np.testing.assert_array_equal(X_aug[:, 0], np.ones(5))

    def test_original_features_preserved(self, simple_X):
        X_aug = _add_intercept_column(simple_X)
        np.testing.assert_array_equal(X_aug[:, 1:], simple_X)


class TestStandardize:

    def test_standardize_fit_zero_mean(self, simple_X):
        X_scaled, mean, std = _standardize_fit(simple_X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-10)

    def test_standardize_fit_unit_std(self, simple_X):
        X_scaled, mean, std = _standardize_fit(simple_X)
        np.testing.assert_allclose(X_scaled.std(axis=0), 1.0, atol=1e-10)

    def test_standardize_fit_returns_correct_mean_std(self, simple_X):
        _, mean, std = _standardize_fit(simple_X)
        np.testing.assert_allclose(mean, simple_X.mean(axis=0))
        np.testing.assert_allclose(std, simple_X.std(axis=0))

    def test_constant_column_std_is_1(self):
        """Columns with zero std should not cause division by zero."""
        X_const = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
        X_scaled, mean, std = _standardize_fit(X_const)
        assert std[0] == 1.0          # safe std for constant column
        assert X_scaled[:, 0].std() == 0.0  # constant column stays constant

    def test_standardize_transform_matches_fit(self, simple_X):
        X_scaled_fit, mean, std = _standardize_fit(simple_X)
        X_scaled_transform = _standardize_transform(simple_X, mean, std)
        np.testing.assert_allclose(X_scaled_fit, X_scaled_transform)


class TestR2Score:

    def test_perfect_predictions_return_1(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert _r2_score(y, y) == pytest.approx(1.0)

    def test_constant_prediction_returns_0(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert _r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_constant_target_returns_0(self):
        """When ss_tot == 0 the function should return 0.0, not raise."""
        y_const = np.array([5.0, 5.0, 5.0])
        assert _r2_score(y_const, y_const) == 0.0

    def test_negative_r2_possible(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 2.0, 1.0])   # reversed — worse than mean
        assert _r2_score(y_true, y_pred) < 0.0

    def test_known_value(self):
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5,  0.0, 2.0, 8.0])
        # ss_res = 0.25 + 0.25 + 0 + 1 = 1.5
        # ss_tot = (3-3)^2 + (-0.5-3)^2 + (2-3)^2 + (7-3)^2
        #        = 0 + 12.25 + 1 + 16 = 29.25   (mean=3)
        expected = 1 - 1.5 / 29.25
        assert _r2_score(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


class TestSoftThreshold:

    def test_positive_above_penalty(self):
        assert _soft_threshold(5.0, 2.0) == pytest.approx(3.0)

    def test_negative_below_negative_penalty(self):
        assert _soft_threshold(-5.0, 2.0) == pytest.approx(-3.0)

    def test_within_penalty_returns_zero(self):
        assert _soft_threshold(1.0, 2.0) == 0.0
        assert _soft_threshold(-1.0, 2.0) == 0.0
        assert _soft_threshold(0.0, 2.0) == 0.0

    def test_exact_boundary_positive(self):
        assert _soft_threshold(2.0, 2.0) == 0.0

    def test_exact_boundary_negative(self):
        assert _soft_threshold(-2.0, 2.0) == 0.0

    def test_zero_penalty(self):
        assert _soft_threshold(3.0, 0.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# LinearRegression tests
# ---------------------------------------------------------------------------

class TestLinearRegression:

    def test_fit_returns_self(self, simple_X, simple_y):
        model = LinearRegression()
        result = model.fit(simple_X, simple_y)
        assert result is model

    def test_is_fitted_after_fit(self, simple_X, simple_y):
        model = LinearRegression()
        model.fit(simple_X, simple_y)
        assert model.is_fitted_

    def test_coef_shape(self, simple_X, simple_y):
        model = LinearRegression()
        model.fit(simple_X, simple_y)
        assert model.coef_.shape == (2,)

    def test_intercept_is_scalar(self, simple_X, simple_y):
        model = LinearRegression()
        model.fit(simple_X, simple_y)
        assert np.isscalar(model.intercept_) or model.intercept_.ndim == 0

    def test_perfect_fit_recovers_coefficients(self, perfect_X, perfect_y):
        """y = 2x + 1 should be recovered exactly."""
        model = LinearRegression()
        model.fit(perfect_X, perfect_y)
        assert model.intercept_ == pytest.approx(1.0, abs=1e-6)
        assert model.coef_[0] == pytest.approx(2.0, abs=1e-6)

    def test_perfect_fit_r2_is_1(self, perfect_X, perfect_y):
        model = LinearRegression()
        model.fit(perfect_X, perfect_y)
        assert model.score(perfect_X, perfect_y) == pytest.approx(1.0, abs=1e-6)

    def test_predict_shape(self, simple_X, simple_y):
        model = LinearRegression()
        model.fit(simple_X, simple_y)
        preds = model.predict(simple_X)
        assert preds.shape == (5,)

    def test_predict_before_fit_raises(self, simple_X):
        model = LinearRegression()
        with pytest.raises(ValueError, match="not been fitted"):
            model.predict(simple_X)

    def test_score_before_fit_raises(self, simple_X, simple_y):
        model = LinearRegression()
        with pytest.raises(ValueError, match="not been fitted"):
            model.score(simple_X, simple_y)

    def test_no_intercept(self, perfect_X, perfect_y):
        model = LinearRegression(fit_intercept=False)
        model.fit(perfect_X, perfect_y)
        assert model.intercept_ == 0.0

    def test_with_standardize(self, simple_X, simple_y):
        model = LinearRegression(standardize=True)
        model.fit(simple_X, simple_y)
        preds = model.predict(simple_X)
        assert preds.shape == (5,)

    def test_r2_between_0_and_1_for_reasonable_data(self, simple_X, simple_y):
        model = LinearRegression()
        model.fit(simple_X, simple_y)
        r2 = model.score(simple_X, simple_y)
        assert 0.0 <= r2 <= 1.0

    def test_single_feature_1d_X(self, simple_y):
        X_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        model = LinearRegression()
        model.fit(X_1d, simple_y)
        preds = model.predict(X_1d)
        assert preds.shape == (5,)

    def test_list_input_accepted(self, simple_y):
        X_list = [[1, 2], [2, 1], [3, 4], [4, 3], [5, 5]]
        model = LinearRegression()
        model.fit(X_list, simple_y)
        assert model.is_fitted_

    def test_predictions_close_to_sklearn(self, simple_X, simple_y):
        """Compare with sklearn's LinearRegression as a reference."""
        from sklearn.linear_model import LinearRegression as SKLearnLR
        sk_model = SKLearnLR()
        sk_model.fit(simple_X, simple_y)

        our_model = LinearRegression()
        our_model.fit(simple_X, simple_y)

        np.testing.assert_allclose(
            our_model.predict(simple_X),
            sk_model.predict(simple_X),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# RidgeRegression tests
# ---------------------------------------------------------------------------

class TestRidgeRegression:

    def test_fit_returns_self(self, simple_X, simple_y):
        model = RidgeRegression()
        assert model.fit(simple_X, simple_y) is model

    def test_is_fitted_after_fit(self, simple_X, simple_y):
        model = RidgeRegression()
        model.fit(simple_X, simple_y)
        assert model.is_fitted_

    def test_coef_shape(self, simple_X, simple_y):
        model = RidgeRegression()
        model.fit(simple_X, simple_y)
        assert model.coef_.shape == (2,)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="nonnegative"):
            RidgeRegression(alpha=-1.0)

    def test_zero_alpha_approaches_ols(self, simple_X, simple_y):
        """With alpha=0, Ridge should match OLS closely."""
        ridge = RidgeRegression(alpha=0.0, standardize=False)
        ols   = LinearRegression(standardize=False)
        ridge.fit(simple_X, simple_y)
        ols.fit(simple_X, simple_y)
        np.testing.assert_allclose(
            ridge.predict(simple_X), ols.predict(simple_X), atol=1e-4
        )

    def test_large_alpha_shrinks_coefficients(self, simple_X, simple_y):
        """Higher alpha should produce smaller coefficient magnitudes."""
        ridge_small = RidgeRegression(alpha=0.01)
        ridge_large = RidgeRegression(alpha=1e6)
        ridge_small.fit(simple_X, simple_y)
        ridge_large.fit(simple_X, simple_y)
        assert np.linalg.norm(ridge_large.coef_) < np.linalg.norm(ridge_small.coef_)

    def test_predict_shape(self, simple_X, simple_y):
        model = RidgeRegression()
        model.fit(simple_X, simple_y)
        assert model.predict(simple_X).shape == (5,)

    def test_predict_before_fit_raises(self, simple_X):
        with pytest.raises(ValueError, match="not been fitted"):
            RidgeRegression().predict(simple_X)

    def test_r2_nonnegative_for_reasonable_data(self, simple_X, simple_y):
        model = RidgeRegression(alpha=0.1)
        model.fit(simple_X, simple_y)
        assert model.score(simple_X, simple_y) >= 0.0

    def test_no_intercept(self, simple_X, simple_y):
        model = RidgeRegression(fit_intercept=False)
        model.fit(simple_X, simple_y)
        assert model.intercept_ == 0.0

    def test_predictions_close_to_sklearn(self, simple_X, simple_y):
        from sklearn.linear_model import Ridge as SKLearnRidge
        alpha = 1.0
        sk_model = SKLearnRidge(alpha=alpha)
        sk_model.fit(simple_X, simple_y)

        our_model = RidgeRegression(alpha=alpha, standardize=False)
        our_model.fit(simple_X, simple_y)

        np.testing.assert_allclose(
            our_model.predict(simple_X),
            sk_model.predict(simple_X),
            atol=1e-5,
        )

    def test_various_alpha_values(self, simple_X, simple_y):
        for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
            model = RidgeRegression(alpha=alpha)
            model.fit(simple_X, simple_y)
            preds = model.predict(simple_X)
            assert preds.shape == (5,)
            assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# LassoRegression tests
# ---------------------------------------------------------------------------

class TestLassoRegression:

    def test_fit_returns_self(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01)
        assert model.fit(simple_X, simple_y) is model

    def test_is_fitted_after_fit(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01)
        model.fit(simple_X, simple_y)
        assert model.is_fitted_

    def test_coef_shape(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01)
        model.fit(simple_X, simple_y)
        assert model.coef_.shape == (2,)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError, match="nonnegative"):
            LassoRegression(alpha=-0.5)

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            LassoRegression(max_iter=0)

    def test_invalid_tol_raises(self):
        with pytest.raises(ValueError, match="tol"):
            LassoRegression(tol=-1e-6)

    def test_large_alpha_produces_sparse_coefficients(self, simple_X, simple_y):
        """High alpha should drive many coefficients to exactly zero."""
        model = LassoRegression(alpha=10.0)
        model.fit(simple_X, simple_y)
        assert np.sum(model.coef_ == 0.0) >= 1

    def test_small_alpha_nonzero_coefficients(self, simple_X, simple_y):
        """Very small alpha should keep coefficients non-zero."""
        model = LassoRegression(alpha=1e-6)
        model.fit(simple_X, simple_y)
        assert np.any(model.coef_ != 0.0)

    def test_predict_shape(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01)
        model.fit(simple_X, simple_y)
        assert model.predict(simple_X).shape == (5,)

    def test_predict_before_fit_raises(self, simple_X):
        with pytest.raises(ValueError, match="not been fitted"):
            LassoRegression().predict(simple_X)

    def test_n_iter_recorded(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01, max_iter=500)
        model.fit(simple_X, simple_y)
        assert 1 <= model.n_iter_ <= 500

    def test_no_intercept(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01, fit_intercept=False)
        model.fit(simple_X, simple_y)
        assert model.intercept_ == 0.0

    def test_r2_nonnegative_for_reasonable_data(self, simple_X, simple_y):
        model = LassoRegression(alpha=0.01)
        model.fit(simple_X, simple_y)
        assert model.score(simple_X, simple_y) >= 0.0

    def test_predictions_close_to_sklearn(self, simple_X, simple_y):
        from sklearn.linear_model import Lasso as SKLearnLasso
        alpha = 0.01
        sk_model = SKLearnLasso(alpha=alpha, max_iter=10000, tol=1e-8)
        sk_model.fit(simple_X, simple_y)

        our_model = LassoRegression(alpha=alpha, max_iter=10000, tol=1e-8)
        our_model.fit(simple_X, simple_y)

        np.testing.assert_allclose(
            our_model.predict(simple_X),
            sk_model.predict(simple_X),
            atol=1e-3,
        )

    def test_various_alpha_values(self, simple_X, simple_y):
        for alpha in [0.001, 0.01, 0.1, 1.0]:
            model = LassoRegression(alpha=alpha)
            model.fit(simple_X, simple_y)
            preds = model.predict(simple_X)
            assert preds.shape == (5,)
            assert not np.any(np.isnan(preds))


# ---------------------------------------------------------------------------
# Cross-model comparison tests
# ---------------------------------------------------------------------------

class TestCrossModelComparisons:

    def test_all_models_fit_and_predict(self, simple_X, simple_y):
        for Model, kwargs in [
            (LinearRegression, {}),
            (RidgeRegression, {"alpha": 1.0}),
            (LassoRegression, {"alpha": 0.01}),
        ]:
            model = Model(**kwargs)
            model.fit(simple_X, simple_y)
            preds = model.predict(simple_X)
            assert preds.shape == (5,)

    def test_regularization_reduces_r2_vs_ols(self, perfect_X, perfect_y):
        """Ridge and Lasso should not outperform OLS on training data."""
        ols   = LinearRegression()
        ridge = RidgeRegression(alpha=5.0, standardize=False)
        lasso = LassoRegression(alpha=0.1)

        ols.fit(perfect_X, perfect_y)
        ridge.fit(perfect_X, perfect_y)
        lasso.fit(perfect_X, perfect_y)

        r2_ols   = ols.score(perfect_X, perfect_y)
        r2_ridge = ridge.score(perfect_X, perfect_y)
        r2_lasso = lasso.score(perfect_X, perfect_y)

        assert r2_ols >= r2_ridge - 1e-6
        assert r2_ols >= r2_lasso - 1e-6

    def test_models_agree_on_trivial_constant_target(self, simple_X):
        """When y is constant all models should predict a constant."""
        y_const = np.full(5, 7.0)
        for Model, kwargs in [
            (LinearRegression, {}),
            (RidgeRegression, {"alpha": 1.0}),
            (LassoRegression, {"alpha": 0.01}),
        ]:
            model = Model(**kwargs)
            model.fit(simple_X, y_const)
            preds = model.predict(simple_X)
            np.testing.assert_allclose(preds, 7.0, atol=1e-4)

    def test_standardize_does_not_change_predictions_materially(
        self, simple_X, simple_y
    ):
        """Standardising should not drastically change predictions."""
        m1 = LinearRegression(standardize=False)
        m2 = LinearRegression(standardize=True)
        m1.fit(simple_X, simple_y)
        m2.fit(simple_X, simple_y)
        np.testing.assert_allclose(
            m1.predict(simple_X), m2.predict(simple_X), atol=1e-6
        )