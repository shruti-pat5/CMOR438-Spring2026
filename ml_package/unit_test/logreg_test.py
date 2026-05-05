"""
Unit tests for logistic_regression.py

Covers:
    - Helper functions: _as_2d_numeric_array, _as_1d_array, _validate_X_y,
      _add_intercept_column, _one_hot, _sigmoid, _softmax,
      _binary_log_loss, _multiclass_log_loss, _accuracy_score
    - LogisticRegression constructor validation
    - Binary classification (_fit_binary)
    - Multinomial softmax classification (_fit_multinomial)
    - One-vs-rest classification (_fit_ovr)
    - auto mode resolution
    - predict_proba, predict, score
    - L2 regularization
    - Standardization
    - fit_intercept=False
    - Edge cases and error paths

Run with:
    pytest logreg_test.py -v
"""

import numpy as np
import pytest

from supervised.logistic_regression import (
    _as_2d_numeric_array,
    _as_1d_array,
    _validate_X_y,
    _add_intercept_column,
    _one_hot,
    _sigmoid,
    _softmax,
    _binary_log_loss,
    _multiclass_log_loss,
    _accuracy_score,
    LogisticRegression,
)

# Shared fixtures
@pytest.fixture
def binary_X():
    return np.array([
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [6.0, 9.0],
        [1.0, 0.6],
        [9.0, 11.0],
    ])

@pytest.fixture
def binary_y():
    return np.array([0, 0, 1, 1, 0, 1])

@pytest.fixture
def multi_X():
    """Three clearly separable clusters."""
    rng = np.random.default_rng(0)
    X0 = rng.normal([0, 0], 0.3, size=(20, 2))
    X1 = rng.normal([5, 0], 0.3, size=(20, 2))
    X2 = rng.normal([0, 5], 0.3, size=(20, 2))
    return np.vstack([X0, X1, X2])

@pytest.fixture
def multi_y():
    return np.array([0]*20 + [1]*20 + [2]*20)

@pytest.fixture
def multi_y_str():
    return np.array(["A"]*20 + ["B"]*20 + ["C"]*20)


# Helper function tasks
class TestAs2dNumericArray:

    def test_valid_2d(self, binary_X):
        out = _as_2d_numeric_array(binary_X)
        assert out.shape == (6, 2)
        assert out.dtype == float

    def test_1d_reshaped_to_column(self):
        out = _as_2d_numeric_array(np.array([1.0, 2.0, 3.0]))
        assert out.shape == (3, 1)

    def test_list_accepted(self):
        out = _as_2d_numeric_array([[1, 2], [3, 4]])
        assert isinstance(out, np.ndarray)
        assert out.dtype == float

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _as_2d_numeric_array(np.empty((0, 2)))

    def test_nan_raises(self, binary_X):
        X_bad = binary_X.copy()
        X_bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            _as_2d_numeric_array(X_bad)

    def test_inf_raises(self, binary_X):
        X_bad = binary_X.copy()
        X_bad[0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite"):
            _as_2d_numeric_array(X_bad)

    def test_non_numeric_raises(self):
        with pytest.raises(TypeError, match="numeric"):
            _as_2d_numeric_array([["a", "b"], ["c", "d"]])

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2D"):
            _as_2d_numeric_array(np.ones((2, 2, 2)))


class TestAs1dArray:

    def test_valid_1d(self, binary_y):
        out = _as_1d_array(binary_y)
        assert out.ndim == 1

    def test_2d_ravelled(self):
        y_2d = np.array([[0], [1], [0]])
        out = _as_1d_array(y_2d)
        assert out.ndim == 1
        assert out.shape == (3,)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _as_1d_array(np.array([]))

    def test_string_labels_preserved(self):
        y_str = np.array(["A", "B", "A"])
        out = _as_1d_array(y_str)
        assert out.tolist() == ["A", "B", "A"]


class TestValidateXy:

    def test_valid_inputs(self, binary_X, binary_y):
        X_out, y_out = _validate_X_y(binary_X, binary_y)
        assert X_out.shape == (6, 2)
        assert y_out.shape == (6,)

    def test_mismatched_samples_raises(self, binary_X):
        y_bad = np.array([0, 1])
        with pytest.raises(ValueError, match="same number of rows"):
            _validate_X_y(binary_X, y_bad)

    def test_nan_in_X_raises(self, binary_y):
        X_nan = np.array([[np.nan, 1.], [2., 3.], [3., 4.],
                           [4., 3.], [5., 5.], [6., 6.]])
        with pytest.raises(ValueError, match="NaN"):
            _validate_X_y(X_nan, binary_y)


class TestAddInterceptColumn:

    def test_ones_prepended(self, binary_X):
        X_aug = _add_intercept_column(binary_X)
        assert X_aug.shape == (6, 3)
        np.testing.assert_array_equal(X_aug[:, 0], np.ones(6))

    def test_original_features_unchanged(self, binary_X):
        X_aug = _add_intercept_column(binary_X)
        np.testing.assert_array_equal(X_aug[:, 1:], binary_X)


class TestOneHot:

    def test_shape(self):
        y = np.array([0, 1, 2, 0, 2])
        Y = _one_hot(y, n_classes=3)
        assert Y.shape == (5, 3)

    def test_correct_encoding(self):
        y = np.array([0, 1, 2])
        Y = _one_hot(y, n_classes=3)
        expected = np.eye(3)
        np.testing.assert_array_equal(Y, expected)

    def test_row_sums_are_one(self):
        y = np.array([0, 1, 2, 1, 0])
        Y = _one_hot(y, n_classes=3)
        np.testing.assert_array_equal(Y.sum(axis=1), np.ones(5))


class TestSigmoid:

    def test_zero_maps_to_half(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive_approaches_one(self):
        assert _sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)

    def test_large_negative_approaches_zero(self):
        assert _sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)

    def test_output_range(self):
        z = np.linspace(-10, 10, 100)
        out = _sigmoid(z)
        assert np.all(out > 0) and np.all(out < 1)

    def test_symmetry(self):
        z = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_sigmoid(z) + _sigmoid(-z), 1.0, atol=1e-12)

    def test_no_overflow_large_negative(self):
        """Should not produce inf or nan for very negative inputs."""
        out = _sigmoid(np.array([-1000.0]))
        assert np.isfinite(out[0])


class TestSoftmax:

    def test_rows_sum_to_one(self):
        Z = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
        probs = _softmax(Z)
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-12)

    def test_output_positive(self):
        Z = np.array([[1.0, -1.0, 0.0]])
        assert np.all(_softmax(Z) > 0)

    def test_uniform_input(self):
        Z = np.zeros((3, 4))
        probs = _softmax(Z)
        np.testing.assert_allclose(probs, np.full((3, 4), 0.25), atol=1e-12)

    def test_numerically_stable_large_values(self):
        Z = np.array([[1000.0, 1001.0, 1002.0]])
        probs = _softmax(Z)
        assert np.all(np.isfinite(probs))
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-12)

    def test_argmax_correct(self):
        Z = np.array([[1.0, 10.0, 2.0]])
        assert np.argmax(_softmax(Z)) == 1


class TestBinaryLogLoss:

    def test_perfect_predictions_near_zero(self):
        y = np.array([1.0, 0.0, 1.0])
        p = np.array([1.0 - 1e-10, 1e-10, 1.0 - 1e-10])
        assert _binary_log_loss(y, p) < 1e-6

    def test_wrong_predictions_high_loss(self):
        y = np.array([1.0, 1.0, 1.0])
        p = np.array([0.01, 0.01, 0.01])
        assert _binary_log_loss(y, p) > 3.0

    def test_returns_float(self, binary_y):
        p = np.full(6, 0.5)
        result = _binary_log_loss(binary_y.astype(float), p)
        assert isinstance(result, float)

    def test_clipping_prevents_log_zero(self):
        y = np.array([1.0])
        p = np.array([0.0])   # would be -inf without clipping
        loss = _binary_log_loss(y, p)
        assert np.isfinite(loss)


class TestMulticlassLogLoss:

    def test_perfect_predictions_near_zero(self):
        Y = np.eye(3)
        p = np.eye(3) * (1 - 1e-10) + 1e-10 / 3
        assert _multiclass_log_loss(Y, p) < 1e-4

    def test_uniform_predictions_higher_loss(self):
        Y = np.eye(3)
        p = np.full((3, 3), 1/3)
        loss = _multiclass_log_loss(Y, p)
        assert loss > 0.9   # log(3) ≈ 1.099

    def test_returns_float(self):
        Y = np.array([[1, 0], [0, 1]])
        p = np.array([[0.8, 0.2], [0.3, 0.7]])
        assert isinstance(_multiclass_log_loss(Y, p), float)


class TestAccuracyScore:

    def test_all_correct(self):
        y = np.array([0, 1, 2])
        assert _accuracy_score(y, y) == pytest.approx(1.0)

    def test_none_correct(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([1, 2, 0])
        assert _accuracy_score(y_true, y_pred) == pytest.approx(0.0)

    def test_partial(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        assert _accuracy_score(y_true, y_pred) == pytest.approx(0.5)

    def test_string_labels(self):
        y_true = np.array(["A", "B", "A"])
        y_pred = np.array(["A", "B", "B"])
        assert _accuracy_score(y_true, y_pred) == pytest.approx(2/3)

    def test_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="same length"):
            _accuracy_score(np.array([0, 1]), np.array([0]))


# Constructor validation
class TestConstructorValidation:

    def test_invalid_learning_rate_raises(self):
        with pytest.raises(ValueError, match="learning_rate"):
            LogisticRegression(learning_rate=0.0)

    def test_invalid_max_iter_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            LogisticRegression(max_iter=0)

    def test_invalid_tol_raises(self):
        with pytest.raises(ValueError, match="tol"):
            LogisticRegression(tol=-1e-6)

    def test_invalid_multi_class_raises(self):
        with pytest.raises(ValueError, match="multi_class"):
            LogisticRegression(multi_class="bad_value")

    def test_negative_l2_penalty_raises(self):
        with pytest.raises(ValueError, match="l2_penalty"):
            LogisticRegression(l2_penalty=-0.1)

    def test_non_integer_random_state_raises(self):
        with pytest.raises(TypeError, match="random_state"):
            LogisticRegression(random_state="abc")

    def test_valid_random_state_none(self):
        m = LogisticRegression(random_state=None)
        assert m.random_state is None

    def test_valid_constructor(self):
        m = LogisticRegression(
            learning_rate=0.01, max_iter=500, tol=1e-5,
            multi_class="multinomial", l2_penalty=0.1, random_state=42
        )
        assert m.learning_rate == pytest.approx(0.01)
        assert m.max_iter == 500

    def test_get_params_returns_all_keys(self):
        m = LogisticRegression()
        params = m.get_params()
        for key in ["learning_rate", "max_iter", "tol", "fit_intercept",
                    "standardize", "multi_class", "l2_penalty", "random_state"]:
            assert key in params


# Binary classification
class TestBinaryClassification:

    def test_fit_returns_self(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        assert m.fit(binary_X, binary_y) is m

    def test_is_fitted_after_fit(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert m.is_fitted_

    def test_mode_is_binary(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert m.mode_ == "binary"

    def test_classes_detected(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        np.testing.assert_array_equal(m.classes_, [0, 1])

    def test_coef_shape(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert m.coef_.shape == (1, 2)

    def test_intercept_is_scalar(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert np.isscalar(m.intercept_) or np.asarray(m.intercept_).ndim == 0

    def test_predict_proba_shape(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        proba = m.predict_proba(binary_X)
        assert proba.shape == (6, 2)

    def test_predict_proba_rows_sum_to_one(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        proba = m.predict_proba(binary_X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(6), atol=1e-10)

    def test_predict_returns_valid_classes(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        preds = m.predict(binary_X)
        assert set(preds).issubset({0, 1})

    def test_predict_shape(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert m.predict(binary_X).shape == (6,)

    def test_score_between_0_and_1(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert 0.0 <= m.score(binary_X, binary_y) <= 1.0

    def test_loss_history_populated(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert len(m.loss_history_) > 0

    def test_loss_decreases_overall(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", max_iter=500, random_state=42)
        m.fit(binary_X, binary_y)
        assert m.loss_history_[-1] < m.loss_history_[0]

    def test_n_iter_recorded(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", max_iter=500, random_state=42)
        m.fit(binary_X, binary_y)
        assert 1 <= m.n_iter_ <= 500

    def test_predict_before_fit_raises(self, binary_X):
        with pytest.raises(ValueError, match="not been fitted"):
            LogisticRegression().predict(binary_X)

    def test_score_before_fit_raises(self, binary_X, binary_y):
        with pytest.raises(ValueError, match="not been fitted"):
            LogisticRegression().score(binary_X, binary_y)

    def test_binary_requires_two_classes(self, binary_X):
        y_single_class = np.zeros(6, dtype=int)
        with pytest.raises(ValueError):
            LogisticRegression(multi_class="binary", random_state=42).fit(
                binary_X, y_single_class
            )

    def test_binary_with_three_classes_raises(self, multi_X, multi_y):
        with pytest.raises(ValueError, match="exactly two classes"):
            LogisticRegression(multi_class="binary", random_state=42).fit(
                multi_X, multi_y
            )

    def test_separable_data_high_accuracy(self):
        """Clearly linearly separable data should achieve perfect accuracy."""
        X = np.array([[1.], [2.], [8.], [9.], [10.]])
        y = np.array([0, 0, 1, 1, 1])
        m = LogisticRegression(multi_class="binary", max_iter=2000, random_state=0)
        m.fit(X, y)
        assert m.score(X, y) == pytest.approx(1.0)

    def test_wrong_feature_count_at_predict_raises(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        X_bad = np.ones((3, 5))
        with pytest.raises(ValueError, match="wrong number of features"):
            m.predict(X_bad)


# Multimodal classification
class TestMultinomialClassification:

    def test_mode_is_multinomial(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.mode_ == "multinomial"

    def test_classes_detected(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        np.testing.assert_array_equal(m.classes_, [0, 1, 2])

    def test_coef_shape(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.coef_.shape == (3, 2)   # (n_classes, n_features)

    def test_intercept_shape(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        assert np.asarray(m.intercept_).shape == (3,)

    def test_predict_proba_shape(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.predict_proba(multi_X).shape == (60, 3)

    def test_predict_proba_rows_sum_to_one(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        proba = m.predict_proba(multi_X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(60), atol=1e-10)

    def test_predict_shape(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.predict(multi_X).shape == (60,)

    def test_separable_clusters_high_accuracy(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", max_iter=2000, random_state=42)
        m.fit(multi_X, multi_y)
        assert m.score(multi_X, multi_y) >= 0.95

    def test_string_labels_handled(self, multi_X, multi_y_str):
        m = LogisticRegression(multi_class="multinomial", random_state=42)
        m.fit(multi_X, multi_y_str)
        np.testing.assert_array_equal(m.classes_, ["A", "B", "C"])
        preds = m.predict(multi_X)
        assert set(preds).issubset({"A", "B", "C"})

    def test_loss_decreases_overall(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", max_iter=500, random_state=42)
        m.fit(multi_X, multi_y)
        assert m.loss_history_[-1] < m.loss_history_[0]


# One vs. rest classification
class TestOvrClassification:

    def test_mode_is_ovr(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="ovr", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.mode_ == "ovr"

    def test_coef_shape(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="ovr", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.coef_.shape == (3, 2)

    def test_predict_proba_rows_sum_to_one(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="ovr", random_state=42)
        m.fit(multi_X, multi_y)
        proba = m.predict_proba(multi_X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(60), atol=1e-6)

    def test_predict_proba_shape(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="ovr", random_state=42)
        m.fit(multi_X, multi_y)
        assert m.predict_proba(multi_X).shape == (60, 3)

    def test_separable_clusters_high_accuracy(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="ovr", max_iter=2000, random_state=42)
        m.fit(multi_X, multi_y)
        assert m.score(multi_X, multi_y) >= 0.95

    def test_loss_decreases_overall(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="ovr", max_iter=500, random_state=42)
        m.fit(multi_X, multi_y)
        assert m.loss_history_[-1] < m.loss_history_[0]

    def test_string_labels_handled(self, multi_X, multi_y_str):
        m = LogisticRegression(multi_class="ovr", random_state=42)
        m.fit(multi_X, multi_y_str)
        assert set(m.predict(multi_X)).issubset({"A", "B", "C"})



# L2 Regularization
class TestL2Regularization:

    def test_large_l2_shrinks_coef(self, binary_X, binary_y):
        # standardize = True (the documented default for regularised models)
        # prevents gradient overflow when l2_penalty is large.
        m_small = LogisticRegression(multi_class="binary", l2_penalty=0.001,
                                     standardize=True, random_state=42)
        m_large = LogisticRegression(multi_class="binary", l2_penalty=10.0,
                                     standardize=True, random_state=42)
        m_small.fit(binary_X, binary_y)
        m_large.fit(binary_X, binary_y)
        assert np.linalg.norm(m_large.coef_) < np.linalg.norm(m_small.coef_)

    def test_zero_penalty_is_unregularized(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", l2_penalty=0.0, random_state=42)
        m.fit(binary_X, binary_y)
        assert m.is_fitted_

    def test_regularization_loss_zero_for_zero_penalty(self):
        m = LogisticRegression(l2_penalty=0.0)
        weights = np.array([1.0, 2.0, 3.0])
        assert m._regularization_loss(weights) == 0.0

    def test_regularization_loss_positive_for_nonzero_penalty(self):
        m = LogisticRegression(l2_penalty=1.0, fit_intercept=True)
        weights = np.array([5.0, 2.0, 3.0])   # first element is intercept
        loss = m._regularization_loss(weights)
        assert loss > 0.0

    def test_intercept_not_penalized(self):
        """Gradient regularization should zero out the intercept position."""
        m = LogisticRegression(l2_penalty=1.0, fit_intercept=True)
        weights = np.array([10.0, 2.0, 3.0])
        grad = m._regularization_gradient(weights.copy())
        assert grad[0] == 0.0    # intercept slot must be zero

    def test_multinomial_large_l2_shrinks_coef(self, multi_X, multi_y):
        m_small = LogisticRegression(multi_class="multinomial", l2_penalty=0.001,
                                     standardize=True, random_state=42)
        m_large = LogisticRegression(multi_class="multinomial", l2_penalty=5.0,
                                     standardize=True, random_state=42)
        m_small.fit(multi_X, multi_y)
        m_large.fit(multi_X, multi_y)
        assert np.linalg.norm(m_large.coef_) < np.linalg.norm(m_small.coef_)


# ---------------------------------------------------------------------------
# fit_intercept=False
# ---------------------------------------------------------------------------

class TestFitInterceptFalse:

    def test_binary_no_intercept(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", fit_intercept=False, random_state=42)
        m.fit(binary_X, binary_y)
        assert m.intercept_ == 0.0

    def test_multinomial_no_intercept(self, multi_X, multi_y):
        m = LogisticRegression(multi_class="multinomial", fit_intercept=False, random_state=42)
        m.fit(multi_X, multi_y)
        np.testing.assert_array_equal(m.intercept_, np.zeros(3))

    def test_predictions_still_valid(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", fit_intercept=False, random_state=42)
        m.fit(binary_X, binary_y)
        preds = m.predict(binary_X)
        assert preds.shape == (6,)
        assert set(preds).issubset({0, 1})


# Standardization
class TestStandardization:

    def test_standardize_true_stores_params(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", standardize=True, random_state=42)
        m.fit(binary_X, binary_y)
        assert m._standardize_params is not None
        assert "mean" in m._standardize_params
        assert "scale" in m._standardize_params

    def test_standardize_false_no_params(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", standardize=False, random_state=42)
        m.fit(binary_X, binary_y)
        assert m._standardize_params is None

    def test_predictions_finite_with_standardize(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", standardize=True, random_state=42)
        m.fit(binary_X, binary_y)
        proba = m.predict_proba(binary_X)
        assert np.all(np.isfinite(proba))

    def test_n_features_in_recorded(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        assert m.n_features_in_ == 2


# Cross-Mode comparisons
class TestCrossModeComparisons:

    def test_all_modes_fit_separable_data(self, multi_X, multi_y):
        for mode in ["multinomial", "ovr"]:
            m = LogisticRegression(multi_class=mode, max_iter=2000, random_state=42)
            m.fit(multi_X, multi_y)
            assert m.score(multi_X, multi_y) >= 0.90

    def test_all_modes_produce_valid_probabilities(self, multi_X, multi_y):
        for mode in ["multinomial", "ovr"]:
            m = LogisticRegression(multi_class=mode, random_state=42)
            m.fit(multi_X, multi_y)
            proba = m.predict_proba(multi_X)
            assert np.all(proba >= 0)
            assert np.all(proba <= 1)
            np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_reproducibility_with_random_state(self, binary_X, binary_y):
        m1 = LogisticRegression(multi_class="binary", random_state=7)
        m2 = LogisticRegression(multi_class="binary", random_state=7)
        m1.fit(binary_X, binary_y)
        m2.fit(binary_X, binary_y)
        np.testing.assert_array_equal(m1.predict(binary_X), m2.predict(binary_X))

    def test_different_random_states_may_differ(self, binary_X, binary_y):
        """Two different seeds should (very likely) produce different weight inits."""
        m1 = LogisticRegression(multi_class="binary", max_iter=1, random_state=1)
        m2 = LogisticRegression(multi_class="binary", max_iter=1, random_state=999)
        m1.fit(binary_X, binary_y)
        m2.fit(binary_X, binary_y)
        # After just 1 iteration weights will reflect the different initializations
        assert not np.allclose(m1.coef_, m2.coef_)

    def test_fit_then_refit_resets_history(self, binary_X, binary_y):
        m = LogisticRegression(multi_class="binary", random_state=42)
        m.fit(binary_X, binary_y)
        first_len = len(m.loss_history_)
        m.fit(binary_X, binary_y)
        # loss_history_ should be reset on refit, not accumulated
        assert len(m.loss_history_) == first_len