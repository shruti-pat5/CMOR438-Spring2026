# Principal Component Analysis (PCA) 

> **Topic:** PCA from Scratch (SVD), Scree Plot, Loading Scores, Dimensionality Reduction, PCA as Preprocessing
> **Dataset:** Wine Dataset

---

## Overview

This notebook applies **Principal Component Analysis (PCA)** to the Wine dataset, working through the full five-step pipeline both manually (via `numpy.linalg.svd`) and using `sklearn`. PCA is introduced as a tool for two distinct but related purposes:

1. **Dimensionality reduction** — compressing 13 chemical features into a small number of principal components while retaining as much variance as possible.
2. **Visualisation** — projecting 13-dimensional data onto 2D and 3D scatter plots that reveal the natural structure of the three cultivar classes.

The mathematical derivation — covariance matrix, eigendecomposition, and SVD — is carried through from first principles before `sklearn` is used to confirm identical results (up to arbitrary sign flips). The notebook concludes with a practical downstream task: sweeping $k = 1$–$13$ PCA components as input to a **Random Forest classifier** to find the minimum number of PCs needed to match full-feature performance.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | Manual SVD, covariance matrix computation, variance explained |
| `pandas` | Data loading, loading-score DataFrame, tidy PCA DataFrame |
| `matplotlib` / `seaborn` | Covariance heatmap, Scree plot, cumulative variance curve, 2D and 3D PC scatter plots, loading score bar charts |
| `sklearn.datasets` | `load_wine` — dataset loading only |
| `sklearn.preprocessing` | `preprocessing.scale` — z-score standardisation |
| `sklearn.decomposition` | `PCA` — sklearn reference implementation; also used for the downstream classification sweep |
| `sklearn.model_selection` | `train_test_split` — for the classification experiment in Section 12 |
| `sklearn.ensemble` | `RandomForestClassifier` — downstream classifier for the PCA preprocessing experiment |
| `sklearn.metrics` | `accuracy_score`, `classification_report` |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine()` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features, measured on very different scales |
| **Target** | Cultivar class — 3 classes (class_0, class_1, class_2) |
| **Task type** | Dimensionality Reduction + Visualisation (+ supervised classification in Section 12) |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Setup | Imports and configuration |
| 2. Load the Wine Dataset | Feature matrix, labels, feature names, descriptive statistics |
| 3. Step 1 — Standardise | Two versions: centred-only matrix `A` (for manual SVD) and z-scored `scaled_X` (for sklearn); verify zero mean and unit variance |
| 4. Step 2 — Covariance Matrix | Compute $S = \frac{1}{n-1}AA^\top$; visualise as a 13×13 heatmap |
| 5. Step 3 — SVD and PCs | `np.linalg.svd` decomposition; verify reconstruction $A = U\Sigma V^\top$; print PC1 loading vector |
| 6. Step 4 — Project onto PC1 & PC2 | Compute variance explained per PC; project data; coloured PC1 vs. PC2 scatter (from scratch) |
| 7. Step 5 — Scree Plot (sklearn) | Fit `PCA()` on `scaled_X`; side-by-side Scree bar chart and cumulative variance curve with 90%/95% thresholds |
| 8. Loading Scores | `pca.components_` table (features × PCs); horizontal bar charts for PC1 and PC2 |
| 9. Project Data & Build PCA DataFrame | `pca.transform`; tidy DataFrame with cultivar labels and colours |
| 10. Two-Component PCA Plot | sklearn PC1 vs. PC2 scatter; 3D PC1 vs. PC2 vs. PC3 scatter |
| 11. Verify Manual vs. sklearn | Column-by-column comparison; handle arbitrary sign flips |
| 12. PCA as Preprocessing | Random Forest on 13 raw features (baseline) vs. PCA + RF for $k = 1$–$13$; accuracy vs. $k$ plot; best $k$ classification report |
| 13. Summary | Five-step PCA pipeline recap; key findings; when to use PCA; limitations |

---

## Key Ideas

### The Five-Step PCA Pipeline

**Step 1 — Standardise**

Because PCA maximises variance, features with large numerical ranges dominate if left unscaled. Each feature is converted to a z-score:

$$z = \frac{x - \bar{x}}{\sigma}$$

Two versions are maintained in the notebook: centred-only `A` (for the manual SVD, which follows the lecture derivation) and fully z-scored `scaled_X` (for sklearn and the downstream task).

---

**Step 2 — Covariance Matrix**

$$S = \frac{1}{n-1} A A^\top \in \mathbb{R}^{13 \times 13}$$

Diagonal entries are variances; off-diagonal entries are covariances. Positive values indicate features that increase together; negative values indicate inverse relationships. The heatmap makes correlated feature clusters immediately visible.

---

**Step 3 — SVD and Principal Component Directions**

The principal component directions are computed via the **Singular Value Decomposition** of $A$:

$$A = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

The rows of $\mathbf{V}^\top$ (columns of $\mathbf{V}$) are the principal component directions — eigenvectors of $S$ ordered by decreasing eigenvalue. The reconstruction identity $A = U\Sigma V^\top$ is verified numerically.

---

**Step 4 — Variance Explained and Projection**

The $i$-th principal component explains:

$$\frac{\sigma_i^2}{\sigma_1^2 + \cdots + \sigma_m^2}$$

of the total variance. Projecting onto the first $k$ components:

$$X_{kD} = A \cdot [\mathbf{PC}_1 \;\; \mathbf{PC}_2 \;\; \cdots \;\; \mathbf{PC}_k]$$

yields a $k$-dimensional representation that captures as much variance as any $k$-dimensional linear projection can.

---

**Step 5 — Scree Plot**

The Scree plot visualises how much variance each component captures and where returns diminish. The elbow — where the curve bends sharply — identifies a natural cut-off. The cumulative variance curve shows how many components are needed to reach 90% or 95% of total variance.

---

### Loading Scores

Each PC is a linear combination of the original features:

$$\text{PC}_1 = w_1 \cdot \text{alcohol} + w_2 \cdot \text{malic\_acid} + \cdots + w_{13} \cdot \text{proline}$$

The coefficients $w_j$ are the **loading scores** stored in `pca.components_`. Large positive loadings mean the feature strongly pushes a sample toward positive PC values; large negative loadings push it negative. Because features are standardised before fitting, loading magnitudes are directly comparable across features.

---

### Sign Flip

PCA is defined only up to a sign flip in each principal component direction — both $\mathbf{v}$ and $-\mathbf{v}$ are valid eigenvectors. `sklearn` and the manual SVD may therefore produce projections that are mirror images of each other along one or more axes. Section 11 checks for this column-by-column and reports whether each PC agrees exactly or only up to a sign flip.

---

### PCA as Preprocessing for Classification

Reducing 13 features to $k$ PCs before training a classifier can:
- Remove noise captured by low-variance components.
- Reduce multicollinearity (correlated features are merged into shared PCs).
- Speed up training by shrinking the feature space.

Section 12 sweeps $k = 1$–$13$ to find the minimum number of PCs that matches (or surpasses) the full-feature Random Forest baseline, reporting the percentage of total variance captured at that optimal $k$.

---
