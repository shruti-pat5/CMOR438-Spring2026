# ml_package

A machine learning package built from scratch. `rice_ml` implements preprocessing utilities, evaluation metrics, and supervised learning models with a scikit-learn-style API — `fit`, `predict`, `score` — without any scikit-learn dependency at runtime.

---

## Package Structure

```
ml_package/
│
├── __init__.py                          # Re-exports preprocessing and postprocessing
│
├── preprocessing.py                     # Feature scaling, row normalisation, dataset splitting
├── postprocessing.py                    # Classification and regression metrics, aggregation helpers
│
└── supervised/
    ├── linear_regression.py             # LinearRegression, RidgeRegression, LassoRegression
    └── logistic_regression.py           # LogisticRegression (binary, multinomial, OvR)
│
tests/
    ├── linreg_test.py                   # pytest suite for linear regression models
    └── logreg_test.py                   # pytest suite for logistic regression model

linear_regression.py                    # Standalone version of the linear models (no package dependency)
```

---

## Installation

No installation is required. 

**Runtime dependencies:** `numpy` only.

**Test dependencies:** `numpy`, `pytest`, `scikit-learn` (used as a reference implementation in cross-model comparison tests only).

---

## Modules

### `preprocessing.py`

Feature scaling and dataset splitting utilities. All functions are NumPy-only and follow a fit-transform pattern compatible with the supervised models.

#### Feature Scaling

| Function | Description | Key parameters |
|:---------|:------------|:---------------|
| `standardize(X)` | Z-score scaling: $(x - \mu) / \sigma$ | `with_mean`, `with_std`, `ddof`, `return_params`, `mean`, `scale` |
| `minmax_scale(X)` | Scales each feature to a target range (default `[0, 1]`) | `feature_range`, `return_params`, `data_min`, `data_range` |
| `maxabs_scale(X)` | Divides each feature by its maximum absolute value | `return_params`, `scale` |

All three scaling functions support a **fit-transform / transform-only** pattern via the `return_params` and saved-params arguments:

```python
# Fit on training data and save parameters
X_train_scaled, params = standardize(X_train, return_params=True)

# Apply saved parameters to test data (no data leakage)
X_test_scaled = standardize(X_test, mean=params["mean"], scale=params["scale"])
```

Constant columns (zero standard deviation / zero range) are handled safely — their scale is set to 1 to avoid division by zero.

#### Row Normalisation

| Function | Description |
|:---------|:------------|
| `l1_normalize_rows(X)` | Scales each row so its L1 norm (sum of absolute values) equals 1 |
| `l2_normalize_rows(X)` | Scales each row so its L2 norm (Euclidean length) equals 1 |

All-zero rows are left unchanged. An `eps` parameter prevents division by zero.

#### Dataset Splitting

| Function | Description | Key parameters |
|:---------|:------------|:---------------|
| `train_test_split(X, y)` | Split into training and test sets | `test_size`, `random_state`, `shuffle`, `stratify` |
| `train_val_test_split(X, y)` | Split into training, validation, and test sets | `val_size`, `test_size`, `random_state`, `shuffle`, `stratify` |

Both functions support **stratified splitting** — passing `stratify=y` preserves class proportions across splits, which is essential for small or imbalanced datasets.

```python
from rice_ml.preprocessing import train_test_split, standardize

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

X_train_scaled, params = standardize(X_train, return_params=True)
X_test_scaled = standardize(X_test, mean=params["mean"], scale=params["scale"])
```

---

### `postprocessing.py`

Evaluation metrics and prediction aggregation helpers. All functions are NumPy-only.

#### Classification Metrics

| Function | Description | Key parameters |
|:---------|:------------|:---------------|
| `accuracy_score(y_true, y_pred)` | Fraction of correct predictions | — |
| `precision_score(y_true, y_pred)` | TP / (TP + FP) | `average` (`binary`, `macro`, `micro`, `weighted`), `pos_label`, `zero_division` |
| `recall_score(y_true, y_pred)` | TP / (TP + FN) | `average`, `pos_label`, `zero_division` |
| `f1_score(y_true, y_pred)` | Harmonic mean of precision and recall | `average`, `pos_label`, `zero_division` |
| `confusion_matrix(y_true, y_pred)` | Count matrix of (true class, predicted class) pairs | `labels` |
| `roc_auc_score(y_true, y_score)` | Area under the ROC curve (binary, rank-sum method) | `pos_label` |
| `log_loss(y_true, y_prob)` | Cross-entropy loss; supports binary (1D) and multiclass (2D) | `labels`, `eps` |

All multiclass metrics support `average="macro"` (equal class weight), `average="weighted"` (weighted by support), and `average="micro"` (equivalent to accuracy).

#### Regression Metrics

| Function | Description |
|:---------|:------------|
| `mean_squared_error(y_true, y_pred)` | Average squared residual |
| `root_mean_squared_error(y_true, y_pred)` | Square root of MSE; same units as target |
| `mean_absolute_error(y_true, y_pred)` | Average absolute residual |
| `r2_score(y_true, y_pred)` | Proportion of variance explained; 1 = perfect, 0 = mean predictor |

#### Aggregation Helpers

| Function | Description |
|:---------|:------------|
| `majority_vote(predictions)` | Takes `(n_models, n_samples)` array; returns the plurality label per sample |
| `weighted_average(values, weights)` | Weighted mean across models; uniform weights if `None` |
| `distance_weighted_average(values, distances)` | Weights by $1 / \text{distance}$; useful for KNN regression |

---

### `supervised/linear_regression.py`

Three linear regression models sharing a common base class. All use a scikit-learn-style API.

#### `LinearRegression`

Ordinary least squares via the pseudoinverse (numerically more stable than directly inverting $X^\top X$):

$$\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top y$$

```python
from rice_ml.supervised.linear_regression import LinearRegression

model = LinearRegression(fit_intercept=True, standardize=False)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
```

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `fit_intercept` | `True` | Prepend a column of ones for the bias term |
| `standardize` | `False` | Z-score features internally before fitting |

---

#### `RidgeRegression`

L2-regularised regression. Closed-form solution; the intercept is **not** penalised:

$$\hat{\boldsymbol{\beta}} = (X^\top X + \alpha I')^{-1} X^\top y$$

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `alpha` | `1.0` | L2 penalty strength; must be ≥ 0 |
| `fit_intercept` | `True` | Include bias term |
| `standardize` | `True` | Recommended when using regularisation |

---

#### `LassoRegression`

L1-regularised regression via **coordinate descent**. Produces sparse solutions — coefficients can be driven to exactly zero:

$$\min_{\boldsymbol{\beta}} \frac{1}{2n}\sum(y_i - \hat{y}_i)^2 + \alpha \sum|\beta_j|$$

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `alpha` | `1.0` | L1 penalty strength; must be ≥ 0 |
| `fit_intercept` | `True` | Include bias term (not penalised) |
| `standardize` | `True` | Recommended when using regularisation |
| `max_iter` | `1000` | Maximum coordinate descent passes |
| `tol` | `1e-6` | Convergence tolerance on max coefficient change |

After fitting, `model.n_iter_` records how many iterations were needed.

---

#### Common API (all models)

```python
model.fit(X, y)          # Returns self
model.predict(X)         # Returns np.ndarray of shape (n_samples,)
model.score(X, y)        # Returns R²
model.coef_              # np.ndarray of shape (n_features,)
model.intercept_         # float (0.0 if fit_intercept=False)
model.is_fitted_         # bool
```

---

### `supervised/logistic_regression.py`

Gradient-descent logistic regression supporting three classification strategies.

#### `LogisticRegression`

```python
from rice_ml.supervised.logistic_regression import LogisticRegression

model = LogisticRegression(
    learning_rate=0.1,
    max_iter=1000,
    multi_class="auto",     # or "binary", "multinomial", "ovr"
    l2_penalty=0.0,
    standardize=True,
    random_state=42,
)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)   # (n_samples, n_classes)
preds = model.predict(X_test)         # (n_samples,)
acc   = model.score(X_test, y_test)
```

#### Classification Strategies

| `multi_class` | When used | How it works |
|:--------------|:----------|:-------------|
| `"auto"` | Default | Selects `"binary"` for 2 classes, `"multinomial"` otherwise |
| `"binary"` | Exactly 2 classes | Sigmoid activation; standard binary cross-entropy loss |
| `"multinomial"` | 2+ classes | Single softmax model; multiclass cross-entropy loss |
| `"ovr"` | 2+ classes | One binary sigmoid classifier per class; normalised score determines prediction |

#### Parameters

| Parameter | Default | Description |
|:----------|:-------:|:------------|
| `learning_rate` | `0.1` | Gradient descent step size; must be > 0 |
| `max_iter` | `1000` | Maximum gradient descent iterations |
| `tol` | `1e-6` | Convergence tolerance on loss change |
| `fit_intercept` | `True` | Include bias term (not penalised by L2) |
| `standardize` | `True` | Z-score features using training set statistics |
| `multi_class` | `"auto"` | Classification strategy (see above) |
| `l2_penalty` | `0.0` | L2 regularisation strength; must be ≥ 0 |
| `random_state` | `None` | Seed for reproducible weight initialisation |

#### Post-fit attributes

```python
model.classes_          # np.ndarray of unique class labels
model.coef_             # (1, n_features) for binary; (n_classes, n_features) for multi
model.intercept_        # float for binary; np.ndarray of shape (n_classes,) for multi
model.mode_             # resolved strategy: "binary", "multinomial", or "ovr"
model.n_iter_           # number of gradient descent steps taken
model.loss_history_     # list of loss values per iteration
model.n_features_in_    # number of input features seen during fit
model.is_fitted_        # bool
```

---

## Running the Tests

```bash
# From the package root
pytest tests/linreg_test.py -v
pytest tests/logreg_test.py -v

# Or run both together
pytest tests/ -v
```

The test suites cover:

| File | Tests | What is covered |
|:-----|------:|:----------------|
| `linreg_test.py` | ~49 | All three linear models, all helper functions, sklearn cross-checks |
| `logreg_test.py` | ~85 | All three classification strategies, L2 regularisation, standardisation, helper functions, sklearn reference comparisons |



