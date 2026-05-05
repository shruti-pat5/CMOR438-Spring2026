# Ensemble Methods

> **Topic:** Hard Voting, Bagging, Random Forests, AdaBoost, Gradient Boosting
> **Dataset:** Wine Dataset

---

## Overview

This notebook explores four families of **ensemble methods** — techniques that combine multiple base learners to produce a single, stronger model — applied to the Wine dataset (178 samples, 13 chemical features, 3 cultivar classes).

The core insight behind ensembles is that a collection of diverse, weak models often outperforms any single strong model. Each method achieves diversity differently:

- **Hard Voting** combines several *different* classifiers by majority vote.
- **Bagging** trains many copies of the *same* model on different bootstrap samples of the data.
- **Random Forests** extend bagging by also randomly subsampling features at each split, decorrelating the trees further.
- **Boosting** (AdaBoost and Gradient Boosting) trains models *sequentially*, each correcting the errors of the previous one.

A single Decision Tree baseline is established first, and every subsequent method is compared against it. The notebook concludes with a head-to-head bar chart ranking all methods by test accuracy.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | Numerical operations |
| `pandas` | Data loading and results tables |
| `matplotlib` / `seaborn` | All visualisations (scatter plots, confusion matrices, learning curves, bar charts) |
| `sklearn.datasets` | `load_wine` |
| `sklearn.model_selection` | `train_test_split`, `cross_val_score` |
| `sklearn.preprocessing` | `StandardScaler` — required for scale-sensitive base learners in the voting ensemble |
| `sklearn.pipeline` | `make_pipeline` — wraps scaling + classifier for KNN and Logistic Regression |
| `sklearn.tree` | `DecisionTreeClassifier` — base learner and single-tree baseline |
| `sklearn.neighbors` | `KNeighborsClassifier` — base learner for voting |
| `sklearn.linear_model` | `LogisticRegression` — base learner for voting |
| `sklearn.svm` | `SVC` — base learner for voting |
| `sklearn.ensemble` | `VotingClassifier`, `BaggingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `GradientBoostingClassifier` |
| `sklearn.metrics` | `accuracy_score`, `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay` |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine()` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features |
| **Target** | Cultivar class — 3 classes (class_0, class_1, class_2) |
| **Task type** | Multiclass Classification |
| **Train / Test split** | 67% / 33% (stratified) |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Setup | Imports and configuration |
| 2. Load & Explore | Class distribution, correlation heatmap |
| 3. Train / Test Split & Baseline | 67/33 stratified split; single `DecisionTreeClassifier(max_depth=3)` baseline |
| 4. Hard Voting | Four base learners (Decision Tree, KNN, Logistic Regression, SVM); individual accuracy vs. ensemble accuracy; confusion matrix |
| 5. Bagging | `BaggingClassifier` with 200 trees, OOB scoring, `max_samples=0.8`; accuracy vs. number of estimators curve |
| 6. Random Forests | `RandomForestClassifier` with 200 trees, `max_features='sqrt'`, OOB scoring; feature importance bar chart; `max_features` sweep |
| 7. AdaBoost | 200 depth-1 stumps, `learning_rate=0.5`; staged accuracy curve; learning rate sweep |
| 8. Gradient Boosting | 200 trees, `max_depth=3`, `learning_rate=0.1`, `subsample=0.8`; staged accuracy curve; learning rate sweep vs. AdaBoost |
| 9. AdaBoost vs. Gradient Boosting | Overlaid staged accuracy curves; 5-fold CV comparison |
| 10. Final Comparison | Annotated bar chart ranking all 6 configurations by test accuracy |

---

## Key Ideas

### Hard Voting

The simplest ensemble: train $m$ diverse classifiers independently and let them vote. The class receiving the most votes wins:

$$\hat{y} = \text{mode}\bigl(\hat{y}_1,\; \hat{y}_2,\; \ldots,\; \hat{y}_m\bigr)$$

Diversity is critical — four very different model families (tree, distance-based, linear, kernel) are combined. Scale-sensitive learners (KNN, Logistic Regression, SVM) are wrapped in `make_pipeline(StandardScaler(), ...)` since `VotingClassifier` trains all estimators on the same raw data.

---

### Bagging (Bootstrap Aggregating)

Trains $B$ copies of the same base estimator, each on a different bootstrap sample (random sample *with replacement*) of the training data, then aggregates by majority vote:

$$\hat{y} = \text{mode}\bigl(h_1(x), h_2(x), \ldots, h_B(x)\bigr)$$

Because each tree sees only ~63% of the training samples, the remaining ~37% act as a natural validation set — the **Out-of-Bag (OOB) score** — requiring no extra cross-validation.

---

### Random Forests

Extend bagging with one additional source of randomness: at each split in each tree, only a random subset of $p = \sqrt{\text{total features}} \approx 4$ features is considered. This **decorrelates** the trees — even the strongest feature cannot dominate every split in every tree — typically reducing variance further beyond plain bagging.

| Method | Bootstrap samples | Random feature subset per split |
|:-------|:-----------------:|:-------------------------------:|
| Bagging | ✓ | ✗ |
| Random Forest | ✓ | ✓ |

---

### AdaBoost

Trains models *sequentially*. After each round $t$, misclassified samples receive higher weight so the next model focuses on the hard examples:

$$\alpha_t = \frac{1}{2} \ln\!\left(\frac{1 - \varepsilon_t}{\varepsilon_t}\right), \quad \hat{y} = \text{sign}\!\left(\sum_t \alpha_t h_t(x)\right)$$

where $\varepsilon_t$ is the weighted error of the $t$-th weak learner and $\alpha_t$ is its vote weight. The base learner is a depth-1 decision stump.

---

### Gradient Boosting

Also sequential, but corrects errors by fitting each new tree to the **pseudo-residuals** — the negative gradient of the loss with respect to the current ensemble's predictions:

$$r_{ti} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{t-1}}, \quad F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$

The learning rate $\eta$ (shrinkage) scales each tree's contribution, trading convergence speed for generalisation. Setting `subsample < 1.0` introduces stochastic gradient boosting, adding further variance reduction.

---

### Staged Accuracy

Both boosting methods expose `staged_predict`, which yields predictions after each successive estimator is added. Plotting staged accuracy reveals how quickly the ensemble converges and whether it begins to overfit as more estimators are added.

