# Regression Trees

> **Topic:** Decision Tree Classification and Regression Tree (from sklearn), Bias–Variance Trade-off, Feature Importance
> **Dataset:** Wine 


---


## Overview

This notebook introduces **Decision Trees** as both classifiers and regressors using the Wine dataset.

The first half fits a `DecisionTreeClassifier` to predict wine cultivar class, visualises the learned tree structure and decision rules in text form, and plots 2D decision regions projected onto the two most discriminating features (Flavanoids vs. Proline).

The second half reframes the task as a **regression problem**: predicting the continuous `alcohol` content of a wine from its other 12 chemical features. A `DecisionTreeRegressor` is trained, and the `max_depth` hyperparameter is swept from 1 to 15 to demonstrate the bias–variance trade-off. The best depth is selected by test MSE and a final model is evaluated with predicted-vs-actual and residual plots.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | Array operations |
| `pandas` | Data loading and display |
| `matplotlib` / `seaborn` | Visualisations (scatter plots, heatmap, tree diagrams, MSE curves) |
| `sklearn.datasets` | `load_wine` |
| `sklearn.model_selection` | `train_test_split` |
| `sklearn.tree` | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `plot_tree`, `export_text` |
| `sklearn.metrics` | `confusion_matrix`, `classification_report`, `mean_squared_error`, `r2_score` |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine()` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features |
| **Classification target** | Cultivar class — 3 classes (class_0, class_1, class_2) |
| **Regression target** | `alcohol` — continuous (range ≈ 11.0 – 14.8) |
| **Train / Test split** | 60% / 40% (stratified for classifier; random for regressor) |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Setup | Imports and plot configuration |
| 2. Load the Wine Dataset | Load data, inspect shapes, class distribution, and descriptive statistics |
| 3. Exploratory Data Analysis | Class distribution bar chart, correlation heatmap, Flavanoids vs. Proline scatter plot |
| 4A. Train / Test Split | 60/40 stratified split for classification |
| 4B. Fit Decision Tree Classifier | `DecisionTreeClassifier(max_depth=3)` — depth, leaf count |
| 4C. Visualise Tree Structure | Text rules via `export_text`; graphical tree via `plot_tree` |
| 4D. 2D Decision Regions | Mesh grid decision boundary projected onto Flavanoids vs. Proline |
| 4E. Evaluate the Classifier | Confusion matrix heatmap and `classification_report` |
| 5A. Prepare Regression Data | Separate `alcohol` as target; split remaining 12 features |
| 5B. Shallow vs. Deep Trees | Compare `max_depth=1` vs. `max_depth=20`; MSE and R² for both |
| 5C. Visualise Shallow Tree | `plot_tree` on the depth-1 regressor |
| 5D. Predicted vs. Actual | Side-by-side scatter plots for shallow and deep trees |
| 6. Depth Sweep | MSE vs. `max_depth` table and plot; identify best depth by test MSE |
| 7. Best Regression Tree | Retrain at optimal depth; predicted-vs-actual and residual plots |
| 8. Feature Importance | Ranked bar chart of MSE-reduction importance for the regression tree |
| 9. Summary | Key takeaways for both the classifier and the regressor |

---

## Key Ideas

### Decision Tree Classifier

At each internal node the tree selects the feature and threshold that most reduces **Gini impurity** across the two child nodes:

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

The predicted class at each leaf is the **majority class** of all training samples that reach it.

---

### Decision Tree Regressor

For regression, the same recursive splitting procedure is used but the impurity measure is **MSE** rather than Gini:

$$J_{\text{split}} = \frac{n_L}{n} \cdot \text{MSE}_L + \frac{n_R}{n} \cdot \text{MSE}_R$$

The predicted value at each leaf is the **mean** of all training targets that reach it.

---

### Regression Metrics

| Metric | Formula | Interpretation |
|:-------|:--------|:---------------|
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error; penalises large errors heavily |
| RMSE | $\sqrt{\text{MSE}}$ | Same units as the target (`alcohol` % vol.) |
| R² | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | Proportion of alcohol variance explained; 1.0 = perfect |

---

### Bias–Variance Trade-off

The `max_depth` sweep (Section 6) makes this concrete for regression trees:

- **`max_depth=1`** — a single split (stump). High bias, cannot capture the structure of the data.
- **`max_depth=20`** — essentially unconstrained. Memorises training data; test MSE is worse than an intermediate depth.
- **Optimal depth** — identified as the `max_depth` that minimises test MSE.

---

### 2D Decision Regions

Because the classifier operates in 13-dimensional space, we re-train a separate 2D model using only Flavanoids and Proline. A dense mesh grid is scored and colour-filled to reveal the axis-aligned rectangular boundaries that decision trees produce.



