# Perceptron

> **Topic:** Perceptron (from scratch), One-vs-Rest Multiclass, Binary Classification, Decision Boundaries, Weight Interpretation
> **Dataset:** Wine Dataset

---

## Overview

This notebook introduces the **Perceptron** — the simplest linear classifier and the conceptual ancestor of modern neural networks — applied to the Wine dataset. Because the Perceptron is inherently binary, two multiclass strategies are explored:

1. **One-vs-Rest (OvR):** train one binary Perceptron per class; at prediction time the class whose Perceptron fires with the largest net input wins.
2. **Binary subset:** filter the dataset to just two cultivar classes and train a single Perceptron directly.

Both the `Perceptron` and `OvRPerceptron` classes are **built from scratch using NumPy**, with no `sklearn` model classes used for training. The notebook then investigates how the learning rate affects convergence, visualises 2D decision boundaries, and interprets the final weights as a proxy for feature importance.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | All from-scratch model implementations and array operations |
| `pandas` | Data loading and results display |
| `matplotlib` | Visualisations (error curves, decision boundaries, weight bar charts, scatter matrix) |
| `sklearn.datasets` | `load_wine` — dataset loading only |
| `sklearn.model_selection` | `train_test_split` |
| `sklearn.preprocessing` | `StandardScaler` — feature standardisation |
| `sklearn.metrics` | `accuracy_score`, `confusion_matrix`, `ConfusionMatrixDisplay`, `classification_report` |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine()` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features |
| **Target** | Cultivar class — 3 classes (class_0, class_1, class_2) |
| **Task type** | Multiclass Classification (via OvR) and Binary Classification |
| **Train / Test split** | 80% / 20% (stratified) |
| **Label encoding** | OvR: +1 for the target class, −1 for all others |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Setup | Imports and plot configuration |
| 2. Load the Wine Dataset | Feature matrix, target vector, class distribution, descriptive statistics |
| 3. Exploratory Data Analysis | Class distribution bar chart, correlation heatmap, scatter matrix of top 5 features |
| 4. Train / Test Split | 80/20 stratified split |
| 5. Standardize Features | `StandardScaler` fit on training set only; verify zero mean and unit variance |
| 6. Perceptron Implementation | From-scratch `Perceptron` class with +1/−1 labels, weight init, `fit`, `predict`, `score`, `errors_` history |
| 7. One-vs-Rest Strategy | From-scratch `OvRPerceptron` wrapper; `fit`, `predict`, `score` |
| 8. Train the OvR Perceptron | `eta=0.1`, `epochs=1000`; print train and test accuracy |
| 9. Test Set Evaluation | Classification report (per-class precision, recall, F1); confusion matrix |
| 10. Training Error Curves | Per-class misclassification count per epoch for all three OvR classifiers |
| 11. Decision Boundaries (2-Feature) | Re-train OvR on Flavanoids and Proline only; mesh grid decision region plot |
| 12. Binary Perceptron (Class 0 vs. 1) | Filter to two classes, encode ±1, train single Perceptron, plot error curve |
| 13. Effect of Learning Rate (`eta`) | Sweep `eta` ∈ {0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0}; table and log-scale accuracy plot |
| 14. Weight Interpretation | Per-class weight bar charts for all 13 standardised features |
| 15. Summary | Results table (fill in after running) and key takeaways |

---

## Key Ideas

### The Perceptron Update Rule

The Perceptron is a linear threshold classifier. It predicts:

$$\hat{y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$$

and updates weights only when it makes a mistake:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot (\hat{y} - y) \cdot \mathbf{x}$$

where $\eta$ is the learning rate. When $\hat{y} = y$ the weights are unchanged; only errors drive learning. This is fundamentally different from gradient descent, which updates on every sample regardless of correctness.

---

### Convergence Guarantee

The **Perceptron Convergence Theorem** states that if the training data is **linearly separable**, the Perceptron is guaranteed to converge to a solution in a finite number of steps. If the data is not linearly separable, the algorithm will never converge and oscillates indefinitely. The training error curves (Section 10) make this concrete — a curve that reaches zero confirms linear separability for that OvR sub-problem.

---

### One-vs-Rest (OvR)

For a $K$-class problem the OvR strategy trains $K$ binary classifiers:

| Classifier | Positive (+1) | Negative (−1) |
|:-----------|:--------------|:--------------|
| Classifier 0 | class_0 | class_1 + class_2 |
| Classifier 1 | class_1 | class_0 + class_2 |
| Classifier 2 | class_2 | class_0 + class_1 |

At prediction time, each classifier computes its **net input** $\mathbf{w}_k^\top \mathbf{x} + b_k$ and the class with the largest value is predicted — even if no classifier is fully confident.

---

### Why Standardisation is Essential

The Perceptron update step adds $\eta \cdot \mathbf{x}$ to the weights. Features with large magnitudes (e.g. proline ≈ 278–1680) will produce weight updates orders of magnitude larger than features with small magnitudes (e.g. ash ≈ 1.4–3.2), causing the decision boundary to be driven almost entirely by the large-scale features. Standardising to zero mean and unit variance ensures all features contribute equally to the update.

---

### Weight Interpretation

Because features are standardised before training, the learned weight $w_j$ for feature $j$ reflects the **relative contribution** of that feature to separating the target class from the rest. A large positive $w_j$ means the feature strongly supports classifying a sample as the target class; a large negative $w_j$ means it strongly opposes it. This makes the weight bar charts (Section 14) directly interpretable as a form of feature importance.

