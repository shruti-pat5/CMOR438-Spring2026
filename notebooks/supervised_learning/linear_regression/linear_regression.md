# Linear Regression

> **Topic:** Linear Regression, Ridge Regression, and Lasso Regression
> **Dataset:** Wine Dataset

---

## Overview

This notebook applies **supervised regression** to the Wine dataset, predicting a wine's **alcohol content** from its 12 remaining chemical features. It serves as a practical demonstration of the `ml_package` linear models API, which implements all three models from scratch using NumPy.

This notebook compares three regression approaches of **Ordinary Least Squares (OLS)**, **Ridge** (L2 regularization), and **Lasso** (L1 regularization) across multiple regularization strengths, and evaluates each using MSE, RMSE, MAE, and R². The notebook concludes with Lasso feature selection, showing which chemical properties are most predictive of alcohol content.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | Numerical computation and metric calculation |
| `pandas` | Data manipulation and results tables |
| `matplotlib` | Visualisation (histograms, scatter plots, bar charts) |
| `sklearn` | Loading the Wine dataset (`load_wine`) |
| `ml_package` | From-scratch `LinearRegression`, `RidgeRegression`, `LassoRegression`, `train_test_split`, `standardize` |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine(as_frame=True)` |
| **Samples** | 178 |
| **Features** | 12 numeric chemical features (all columns except `alcohol`) |
| **Target** | `alcohol`, a continuous numeric value (range ≈ 11.0 – 14.8) |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Load Dataset | Load the Wine dataset, separate `alcohol` as the target, inspect distributions |
| 2. Exploratory Data Analysis | Feature correlations with `alcohol`; identify strongest predictors |
| 3. Train / Test Split | 80 / 20 stratified split using `ml_package.train_test_split` |
| 4. Standardize Features | Fit scaler on training set only; apply to test set to prevent data leakage |
| 5. Metric Helpers | Define `mse`, `rmse`, `mae`, `r2`, and a unified `evaluate` function |
| 6. Fit OLS Linear Regression | Fit and evaluate the baseline closed-form OLS model; inspect coefficients |
| 7. Actual vs. Predicted | Scatter plot of true vs. predicted alcohol for the OLS model |
| 8. Residual Analysis | Residuals vs. predicted plot and residual distribution histogram |
| 9. Compare OLS, Ridge, and Lasso | Fit 7 models across multiple `alpha` values; compare by Test RMSE |
| 10. Lasso Feature Selection | Inspect which features Lasso zeroes out; visualise the best model's predictions |

---

## Key Ideas

### Ordinary Least Squares (OLS)

OLS minimizes the sum of squared residuals and has a closed-form solution via the pseudoinverse:

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}$$

No regularization is applied, so all 12 features receive non-zero coefficients.

---

### Ridge Regression (L2)

Ridge adds a penalty on the squared magnitude of the coefficients, shrinking them toward zero without eliminating any:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} \beta_j^2$$

The intercept is not penalized. Larger `alpha` → stronger shrinkage.

---

### Lasso Regression (L1)

Lasso adds a penalty on the absolute value of the coefficients. Because of the geometry of the L1 ball, it can drive coefficients to **exactly zero**, performing automatic feature selection:

$$\min_{\boldsymbol{\beta}} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \alpha \sum_{j=1}^{p} |\beta_j|$$

Solved via **coordinate descent** rather than a closed-form expression.

---

### Evaluation Metrics

| Metric | Formula | Interpretation |
|:-------|:--------|:---------------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Mean squared error; penalises large errors heavily |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as target; easier to interpret |
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Robust to outliers |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained; 1.0 = perfect |

---

## Results & Key Findings

- Standardizing features is essential before applying Ridge or Lasso — without it, features on different scales receive unfairly large or small penalties.
- Lasso with `alpha=0.1` zeroes out several features, revealing the most chemically informative predictors of alcohol content.
- Ridge shrinks coefficients smoothly but never eliminates them entirely, unlike Lasso.
- OLS provides a useful baseline but may overfit on noisier features without any regularization.

---

