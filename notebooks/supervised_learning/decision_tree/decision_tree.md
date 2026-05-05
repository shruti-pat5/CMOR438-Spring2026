# Decision Tree Classifier

> **Topic:** Decision Tree Classification, Hyperparameter Tuning, Feature Importance, and Pruning
> **Dataset:** Wine Dataset

---

## Overview

This notebook applies a **Decision Tree Classifier** to the Wine dataset, predicting cultivar class (0, 1, or 2) from 13 chemical features. The focus is less on the algorithm itself and more on the full **model development workflow**: establishing a baseline, defining a validated hyperparameter search space, running an exhaustive grid search, and rigorously evaluating the best configuration on a held-out test set.

Two complementary approaches to feature importance are compared — **Gini (mean decrease in impurity)** and **permutation importance** — and the bias–variance trade-off is made concrete through both a `max_depth` sweep and a **cost-complexity pruning** path.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | Numerical operations |
| `pandas` | Data loading, results tables, long-format reshaping for plots |
| `matplotlib` / `seaborn` | All visualisations (pairplot, heatmap, boxplots, bar charts, tree diagram) |
| `sklearn.datasets` | `load_wine` — dataset loading |
| `sklearn.model_selection` | `train_test_split`, `StratifiedKFold`, `GridSearchCV`, `cross_validate` |
| `sklearn.tree` | `DecisionTreeClassifier`, `plot_tree` |
| `sklearn.metrics` | `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`, `accuracy_score`, `f1_score` |
| `sklearn.inspection` | `permutation_importance` — model-agnostic feature importance |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine()` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features (e.g. flavanoids, proline, alcohol, color intensity) |
| **Target** | Cultivar class — 3 classes (class_0, class_1, class_2) |
| **Task type** | Multiclass Classification |
| **Train / Test split** | 80% / 20% (stratified) |

---

## Notebook Structure

| Cell | Description |
|:-----|:------------|
| 0 | Setup — imports and global plot configuration |
| 1 | Load dataset — build labelled DataFrame, inspect class distribution and descriptive stats |
| 2 | EDA — class distribution bar chart and feature correlation heatmap |
| 3 | EDA — pairplot of the five most discriminating features, coloured by cultivar |
| 4 | Train / test split — 80/20 stratified split |
| 5 | `validate_decision_tree_param_grid` — custom validation function for the search space |
| 6 | Define hyperparameter grid — 8 parameters, ~9,200 total combinations |
| 7 | CV strategy and scoring — `StratifiedKFold(n_splits=5)`, macro-averaged metrics |
| 8 | Baseline cross-validation — untuned default tree, 5-fold CV across 4 metrics |
| 9 | Baseline boxplot — distribution of CV scores across folds |
| 10 | Grid search — `GridSearchCV` with F1-macro as the optimisation target |
| 11 | Top 20 configurations — ranked bar chart of mean CV F1 |
| 12 | Tuned model CV — re-evaluate best model on the same folds as baseline |
| 13 | Baseline vs. tuned comparison — side-by-side CV score boxplots |
| 14 | Test set evaluation — classification report on the held-out 20% |
| 15 | Confusion matrix — test set predictions vs. true labels |
| 16 | Tree visualisation — full `plot_tree` diagram of the best model |
| 17 | Gini feature importances — ranked bar chart from `feature_importances_` |
| 18 | Permutation importances — model-agnostic importance with error bars (30 repeats) |
| 19 | Depth sweep — accuracy vs. `max_depth` holding all other params fixed |
| 20 | Cost-complexity pruning path — accuracy vs. `ccp_alpha` |
| 21 | Final summary — side-by-side table of baseline vs. tuned; best params; top 3 features |

---

## Key Ideas

### Decision Tree Splitting

At each node, the tree selects the feature and threshold that minimises the chosen **impurity measure** across the resulting child nodes. Three criteria are available:

$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p_k^2$$

$$\text{Entropy}(t) = -\sum_{k=1}^{K} p_k \log_2 p_k$$

$$\text{Log-loss}(t) = -\sum_{k=1}^{K} p_k \log p_k$$

The split that most reduces weighted impurity across child nodes is chosen.

---

### Hyperparameter Search Space

| Parameter | Values searched | Effect |
|:----------|:----------------|:-------|
| `criterion` | gini, entropy, log_loss | Impurity measure used at each split |
| `splitter` | best, random | Whether to find the globally best split or a random one |
| `max_depth` | 2–10, None | Controls tree depth; primary driver of overfitting |
| `min_samples_split` | 2, 5, 10, 20 | Minimum samples needed to split a node |
| `min_samples_leaf` | 1, 2, 4, 8 | Minimum samples required at each leaf |
| `max_features` | None, sqrt, log2 | Feature subsampling at each split |
| `class_weight` | None, balanced | Adjusts for class imbalance |
| `ccp_alpha` | 0.0, 0.001, 0.005, 0.01 | Cost-complexity pruning strength |

The grid is validated by `validate_decision_tree_param_grid` before search to catch invalid values early.

---

### Gini Importance vs. Permutation Importance

| | Gini Importance | Permutation Importance |
|:--|:---------------|:-----------------------|
| **How** | Weighted total impurity reduction across all splits using that feature | Drop in accuracy when feature values are randomly shuffled |
| **Bias** | Can favour high-cardinality features | Unbiased; directly measures predictive contribution |
| **Requires test set** | No — computed from training structure | Yes — evaluated on test set |
| **Speed** | Instant (already computed during fit) | Slower (30 repeats × 13 features) |

---

### Cost-Complexity Pruning

Post-hoc pruning removes subtrees whose accuracy gain doesn't justify their complexity. The effective pruning strength is controlled by `ccp_alpha`:

$$\text{Score}_\alpha(T) = \text{Impurity}(T) + \alpha \cdot |T|$$

where $|T|$ is the number of leaves. Higher `alpha` → smaller, simpler trees. The pruning path sweep identifies the `alpha` that maximises test accuracy.


---

## How to Run

```bash
# 1. Clone or download this notebook
# 2. Install dependencies (if not already installed)
pip install numpy pandas matplotlib seaborn scikit-learn

# 3. Launch Jupyter
jupyter notebook decision_tree.ipynb
```

> **Python version:** 3.11+
> **Note:** The grid search explores a large parameter space (~9,200 combinations × 5 folds). Runtime depends on hardware — expect several minutes. `n_jobs=-1` uses all available CPU cores to parallelise the search.

---

