# CMOR 438: Data Science and Machine Learning

# Shruti Patankar

## Overview

This repository contains a custom machine learning package developed for **CMOR 438**.
The project implements classic **supervised and unsupervised learning algorithms from scratch** using NumPy, organized into a clean and modular Python package called **`ml_package`**.

The package is paired with structured **Jupyter notebooks** that demonstrate each algorithm on the Wine dataset.

---

## Project Highlights

- Fully custom implementations of core machine learning algorithms
- A well-structured Python package (`ml_package/`) with separate modules for processing and either linear or logistic regression
- Educational notebooks demonstrating each algorithm step-by-step
- A **pytest test suite** covering the linear and logistic regression models

---

## Capabilities

### Supervised Learning

Implemented in `ml_package/supervised/`:

- **Linear Regression** — OLS, Ridge (L2), and Lasso (L1) via coordinate descent
- **Logistic Regression** — Binary, multinomial softmax, and one-vs-rest; trained with gradient descent

### Unsupervised Learning

Demonstrated in `notebooks/unsupervised_learning/`:

- **k-Means Clustering**
- **Principal Component Analysis (PCA)**

### Data Processing Utilities

Implemented in `ml_package/processing/`:

- Feature standardization (`standardize`, `minmax_scale`, `maxabs_scale`)
- Row normalization (`l1_normalize_rows`, `l2_normalize_rows`)
- Dataset splitting (`train_test_split`, `train_val_test_split`)
- Classification and regression evaluation metrics (`accuracy_score`, `f1_score`, `r2_score`, `roc_auc_score`, and more)

---

## Repository Structure

```text
.
├── ml_package/
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # Scaling, normalisation, splitting
│   │   └── postprocessing.py       # Classification and regression metrics
│   │
│   ├── supervised/
│   │   ├── linear_regression.py    # LinearRegression, RidgeRegression, LassoRegression
│   │   └── logistic_regression.py  # LogisticRegression (binary, multinomial, OvR)
│   │
│   └── unit_test/
│       ├── linreg_test.py          # pytest suite for linear regression
│       └── logreg_test.py          # pytest suite for logistic regression
│
├── notebooks/
│   ├── supervised_learning/
│   │   ├── decision_tree/
│   │   ├── ensemble_methods/
│   │   ├── gradient_descent/
│   │   ├── knn/
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── perceptron/
│   │   └── regression_tree/
│   │
│   └── unsupervised_learning/
│       ├── kmeans/
│       └── pca/
│
├── LICENSE
└── README.md
```

---

## Notebooks

Each algorithm has a corresponding notebook folder under `notebooks/`. Notebooks cover:

- Dataset loading and exploration
- Preprocessing and scaling
- Training and evaluation
- Visualisation of predictions, decision boundaries, or cluster assignments
- Discussion of assumptions, behaviour, and limitations

| Folder | Topic |
|:-------|:------|
| `supervised_learning/linear_regression/` | OLS, Ridge, and Lasso Regression on the Wine dataset |
| `supervised_learning/logistic_regression/` | Logistic Regression (gradient descent, OvR, multinomial) |
| `supervised_learning/knn/` | k-Nearest Neighbours from scratch with wine recommender |
| `supervised_learning/perceptron/` | Perceptron from scratch with OvR multiclass strategy |
| `supervised_learning/decision_tree/` | Decision Tree classifier with GridSearchCV tuning |
| `supervised_learning/regression_tree/` | Regression Tree predicting alcohol content |
| `supervised_learning/ensemble_methods/` | Hard Voting, Bagging, Random Forests, AdaBoost, Gradient Boosting |
| `supervised_learning/gradient_descent/` | Batch GD, SGD, Mini-Batch, Momentum from scratch |
| `unsupervised_learning/kmeans/` | k-Means clustering from scratch with Elbow Method |
| `unsupervised_learning/pca/` | PCA from scratch via SVD; Scree plot; PCA as preprocessing |

---

## Testing

The linear and logistic regression models are covered by a **pytest** suite in `ml_package/unit_test/`.

| File | Coverage |
|:-----|:---------|
| `linreg_test.py` | `LinearRegression`, `RidgeRegression`, `LassoRegression`, all helper functions, sklearn cross-checks |
| `logreg_test.py` | `LogisticRegression` (binary, multinomial, OvR), L2 regularisation, standardisation, helper functions |

To run the tests:

```bash
cd ml_package/unit_test
pytest -v
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Example usage:

```python
from ml_package.supervised.linear_regression import LinearRegression, RidgeRegression, LassoRegression
from ml_package.supervised.logistic_regression import LogisticRegression
from ml_package.processing.preprocessing import standardize, train_test_split
from ml_package.processing.postprocessing import accuracy_score, r2_score
```

---

## Project Goals

- Deepen understanding of machine learning algorithms by implementing them from first principles
- Practice professional-quality Python package organisation
- Integrate testing, documentation, and examples into a single codebase
- Emphasise algorithmic assumptions, limitations, and interpretation
- Cover the full ML workflow: preprocessing → modelling → evaluation → visualisation

---

## Author

Shruti Patankar
Rice University — CMOR 438