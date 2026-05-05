# Logistic Regression

> **Topic:** Logistic Regression 
> **Dataset:** Wine Dataset

---

## Overview

This notebook applies **supervised classification** to the Wine dataset, predicting a wine's cultivar class (0, 1, or 2) from its 13 chemical features. It serves as a practical demonstration of the `ml_package` logistic regression API, which implements classification from scratch.

We train and compare two multiclass strategies — **Multinomial Softmax** (a single model with a softmax output layer) and **One-vs-Rest** (one binary sigmoid classifier per class) — and investigate the effect of **L2 regularization** across five penalty strengths. The notebook concludes with coefficient interpretation and a detailed look at predicted probabilities, including a confidence analysis of correct vs. incorrect predictions.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | Numerical computation and metric calculation |
| `pandas` | Data manipulation and results tables |
| `matplotlib` | Visualisation (confusion matrices, loss curves, bar charts) |
| `sklearn` | Loading the Wine dataset (`load_wine`) only |
| `ml_package` | From-scratch `LogisticRegression`, `train_test_split`, `standardize` |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine(as_frame=True)` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features (e.g. flavanoids, proline, alcohol, malic acid) |
| **Target** | Cultivar class — 3 classes (class_0, class_1, class_2) |
| **Task type** | Multiclass Classification |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Setup | Imports, path configuration, and `ml_package` loading |
| 2. Load the Wine Dataset | Load data, inspect shapes, class counts, and feature values |
| 3. Exploratory Data Analysis | Per-class feature means; scatter plot of flavanoids vs. proline |
| 4. Train / Test Split | 80 / 20 split using `ml_package.train_test_split` |
| 5. Standardize Features | Fit scaler on training set only; apply to test set to prevent data leakage |
| 6. Evaluation Helper Functions | Define `accuracy`, `confusion_matrix`, `plot_confusion_matrix`, and `evaluate` |
| 7. Multinomial Softmax Regression | Fit and evaluate a single softmax model; plot training loss curve |
| 8. One-vs-Rest Regression | Fit one binary classifier per class; evaluate and plot confusion matrix |
| 9. Effect of L2 Regularization | Sweep `l2_penalty` across [0, 0.01, 0.1, 1.0, 10.0]; plot accuracy vs. penalty |
| 10. Compare All Strategies | Head-to-head comparison of 4 configurations by train and test accuracy |
| 11. Interpret Coefficients | Bar charts of standardized coefficients per class for the best model |
| 12. Predicted Probabilities | Inspect `predict_proba` output; confidence of correct vs. incorrect predictions |
| 13. Prediction Example on One Wine | Walk through a single sample's predicted class and probability breakdown |
| 14. Summary | Recap of the workflow and coefficient interpretation note |

---

## Key Ideas

### Multinomial Softmax Regression

Trains a single model with one weight vector per class. The **softmax** function converts raw scores (logits) into a valid probability distribution across all $K$ classes:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x}}}$$

The model minimizes the **multiclass cross-entropy loss**:

$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik}$$

---

### One-vs-Rest (OvR) Classification

Trains $K$ independent **binary sigmoid** classifiers, one per class. For class $k$, the target is relabelled as 1 if $y = k$, else 0. The class with the highest sigmoid score wins:

$$\hat{y} = \arg\max_k \; \sigma(\mathbf{w}_k^\top \mathbf{x}), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$$

---

### L2 Regularization

An L2 penalty is added to the loss to shrink coefficients and reduce overfitting. The intercept is never penalized:

$$\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2} \sum_{j} \beta_j^2$$

Higher `l2_penalty` $\lambda$ → stronger shrinkage → smaller coefficient magnitudes.

---

### Coefficient Interpretation

Because all features are **standardized** before fitting, each coefficient represents:

> the expected change in log-odds for a **one-standard-deviation increase** in that feature, holding all others fixed.

Larger absolute values indicate more influential predictors for that class.

---

## Results & Key Findings

<!-- Fill in the table below after running all cells. -->

| Model | l2_penalty | Train Accuracy | Test Accuracy | Iterations |
|:------|:----------:|:--------------:|:-------------:|:----------:|
| Multinomial | 0.0 | | | |
| OvR | 0.0 | | | |
| Multinomial + L2 | 0.1 | | | |
| OvR + L2 | 0.1 | | | |

**L2 regularization sweep (Multinomial):**

| l2_penalty | Train Accuracy | Test Accuracy |
|:----------:|:--------------:|:-------------:|
| 0.0 | | |
| 0.01 | | |
| 0.1 | | |
| 1.0 | | |
| 10.0 | | |

**Notable observations:**

- Standardizing features before fitting is critical — logistic regression is scale-sensitive, and unscaled features with large ranges will dominate the gradient updates.
- The training loss curve for the Multinomial model shows how gradient descent converges over iterations.
- OvR and Multinomial tend to achieve similar accuracy on the Wine dataset, since the three classes are relatively well-separated in feature space (visible in the Flavanoids vs. Proline scatter plot).
- Moderate L2 regularization can improve test accuracy by preventing the model from overfitting to the training set.
- The confidence histogram shows that correct predictions tend to have higher maximum predicted probability than incorrect ones.

---

## How to Run

```bash
# 1. Clone or download this notebook and the ml_package directory
# 2. Install dependencies
pip install numpy pandas matplotlib scikit-learn

# 3. Ensure ml_package is on your Python path (the notebook does this automatically)

# 4. Launch Jupyter
jupyter notebook logistic_regression_wine.ipynb
```

> **Python version:** 3.10+
> **Note:** This notebook uses `ml_package` — ensure the package root is in the same directory as the notebook, or one level above it. The notebook patches `sys.path` automatically.

---
