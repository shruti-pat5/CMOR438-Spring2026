# Gradient Descent

> **Topic:** Batch Gradient Descent, Stochastic Gradient Descent, Mini-Batch Gradient Descent, and Momentum
> **Dataset:** Wine Dataset

---

## Overview

This notebook implements **gradient descent from scratch** and applies it to fit a **softmax logistic regression** classifier on the Wine dataset. Rather than using a pre-built solver, all three core variants of gradient descent are built using only NumPy, giving a ground-level understanding of how iterative optimisation works in machine learning.

We cover batch GD, SGD, and mini-batch GD side by side, investigate the effect of learning rate and momentum, and finish with a sanity check against `sklearn`'s `SGDClassifier`. The loss surface is also visualised on a simple 2D quadratic before touching real data, making the geometry of gradient descent concrete.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | All gradient descent implementations and matrix operations |
| `pandas` | Data loading and display |
| `matplotlib` / `seaborn` | Loss curves, accuracy curves, 2D/3D surface plots, confusion matrix |
| `sklearn` | Dataset loading, train/test split, `StandardScaler`, `SGDClassifier` (reference), `ConfusionMatrixDisplay` |

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
| 2. Load & Prepare the Wine Dataset | Load data, standardise features (fit on train only) |
| 3. Softmax Logistic Regression via GD | Model definition, cross-entropy loss, gradient formula, and helper functions (`softmax`, `one_hot`, `cross_entropy_loss`, `predict`) |
| 4. Batch Gradient Descent | Full-dataset gradient at each step; loss and accuracy curves |
| 5. Stochastic Gradient Descent (SGD) | Single random sample per step; noisy loss curve |
| 6. Mini-Batch Gradient Descent | Random batch of $B=16$ samples per step |
| 7. Comparing Batch, SGD, and Mini-Batch | Overlaid loss and accuracy curves for all three variants |
| 8. The Effect of Learning Rate | Sweep $\eta \in \{0.001, 0.01, 0.05, 0.1, 0.5, 1.0\}$; visualise convergence and divergence |
| 9. Momentum | Velocity accumulation to dampen oscillations; sweep $\beta \in \{0.0, 0.5, 0.7, 0.9, 0.99\}$ |
| 10. Evaluate the Best From-Scratch Model | Classification report and confusion matrix for the best variant |
| 11. Comparison: From-Scratch vs. sklearn | Sanity check against `sklearn`'s `SGDClassifier` |
| 12. Final Summary of All Variants | Ranked bar chart of test accuracy across all five configurations |

---

## Key Ideas

### The Gradient Descent Update Rule

Given a model with parameters $\boldsymbol{W}$ and a loss function $J(\boldsymbol{W})$, the update at each step is:

$$\boldsymbol{W}^{(t+1)} = \boldsymbol{W}^{(t)} - \eta \, \nabla_{\boldsymbol{W}} J(\boldsymbol{W}^{(t)})$$

where $\eta > 0$ is the **learning rate**. The gradient points in the direction of steepest ascent; subtracting it moves parameters downhill.

---

### Softmax and Cross-Entropy Loss

For a $K$-class problem with weight matrix $\mathbf{W} \in \mathbb{R}^{K \times p}$:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j=1}^{K} e^{\mathbf{w}_j^\top \mathbf{x}}} \quad \text{(softmax)}$$

$$J(\mathbf{W}) = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{ik} \log \hat{p}_{ik} \quad \text{(cross-entropy)}$$

$$\nabla_{\mathbf{W}} J = \frac{1}{n}(\hat{\mathbf{P}} - \mathbf{Y})^\top \mathbf{X} \quad \text{(gradient)}$$

---

### Three Gradient Descent Variants

| Variant | Gradient computed on | Noise | Typical use |
|:--------|:---------------------|:-----:|:------------|
| **Batch GD** | Full dataset ($n$ samples) | Low | Small datasets; stable convergence |
| **SGD** | 1 random sample | High | Large datasets; can escape local minima |
| **Mini-Batch GD** | $B$ random samples | Medium | Default for neural networks ($B \in [16, 256]$) |

---

### Momentum

Momentum accumulates a **velocity** $\mathbf{v}$ in the direction of persistent gradients:

$$\mathbf{v}^{(t+1)} = \beta \, \mathbf{v}^{(t)} + \eta \, \nabla J(\boldsymbol{W}^{(t)})$$
$$\boldsymbol{W}^{(t+1)} = \boldsymbol{W}^{(t)} - \mathbf{v}^{(t+1)}$$

When $\beta = 0$ this reduces to plain gradient descent. The typical value is $\beta = 0.9$.

---

### Learning Rate

The learning rate $\eta$ is the most important hyperparameter:

- **Too small** ($\eta = 0.001$) → very slow convergence; may never reach the minimum in a fixed number of epochs
- **Too large** ($\eta = 1.0$) → overshoots; loss may diverge or oscillate
- **Just right** ($\eta \approx 0.1$) → fast, stable convergence

---

## Results & Key Findings

<!-- Fill in the table below after running all cells. -->

| Model | η | Extra params | Test Accuracy |
|:------|:-:|:-------------|:-------------:|
| Batch GD (from scratch) | 0.1 | 500 epochs | |
| SGD (from scratch) | 0.01 | 100 epochs | |
| Mini-Batch GD (from scratch) | 0.05 | B=16, 200 epochs | |
| Batch GD + Momentum (from scratch) | 0.05 | β=0.9, 300 epochs | |
| sklearn SGDClassifier | 0.01 | log_loss, 300 iter | |

**Notable observations:**

- Standardising features before gradient descent is essential — without it, features on different scales cause the gradient to be dominated by large-magnitude columns, preventing convergence.
- The SGD loss curve is visibly noisier than Batch GD due to single-sample gradient estimates, but still converges to a similar final accuracy.
- The learning rate sweep shows a clear transition from underfitting ($\eta = 0.001$, barely moves) to divergence ($\eta = 1.0$, loss explodes); the sweet spot is in the $[0.05, 0.1]$ range.
- Momentum with $\beta = 0.9$ converges faster than plain Batch GD on the same data, and very high momentum ($\beta = 0.99$) can cause initial overshooting.
- The from-scratch Batch GD and sklearn's `SGDClassifier` produce comparable test accuracy, validating the correctness of the implementation.

---

## How to Run

```bash
# 1. Clone or download this notebook
# 2. Install dependencies (if not already installed)
pip install numpy pandas matplotlib seaborn scikit-learn

# 3. Launch Jupyter
jupyter notebook gradient_descent.ipynb
```

> **Python version:** 3.11+
> **Kernel:** `ml_env` (or any environment with the dependencies above installed)
> **Note:** `np.random.seed(42)` is set at the top of the notebook for reproducibility — re-running cells out of order may produce different results.

---

