# $k$-Means Clustering 

> **Topic:** $k$-Means Clustering (from scratch), WCSS Elbow Method, ARI, Silhouette Score, PCA Visualisation
> **Dataset:** Wine Dataset

---

## Overview

This notebook introduces **$k$-Means Clustering** — one of the most widely used unsupervised learning algorithms — and applies it to the Wine dataset. Unlike every other notebook in this series, there are **no labels used during training**. The algorithm discovers natural groupings in the 13-dimensional chemical feature space on its own; the true cultivar labels are only revealed afterwards to evaluate how well the discovered clusters match reality.

The entire algorithm is **built from scratch using NumPy**: Euclidean distance, cluster assignment, centre update, and the full iterative loop. Convergence is visualised step-by-step (iterations 1, 2, 3, and 10) before the full 100-iteration run. The notebook then systematically addresses the question *"how do we choose k?"* using three complementary diagnostics — WCSS (elbow method), Adjusted Rand Index, and Silhouette Score — all swept over $k = 1$ to $10$. A final sanity check compares the from-scratch implementation against `sklearn`'s optimised `KMeans`.

---

## Libraries Used

| Library | Purpose |
|:--------|:--------|
| `numpy` | All from-scratch algorithm implementations |
| `pandas` | Data loading and display |
| `matplotlib` / `seaborn` | Visualisations (PCA scatter plots, convergence grids, elbow/ARI/silhouette charts) |
| `sklearn.datasets` | `load_wine` — dataset loading only |
| `sklearn.preprocessing` | `StandardScaler` — feature standardisation |
| `sklearn.decomposition` | `PCA` — 2D projection for visualisation only; clustering runs in full 13D space |
| `sklearn.cluster` | `KMeans` — reference implementation for the final sanity check |
| `sklearn.metrics` | `adjusted_rand_score`, `silhouette_score` — cluster quality evaluation |

---

## Dataset

| Property | Value |
|:---------|:------|
| **Name** | Wine Dataset |
| **Source** | `sklearn.datasets.load_wine()` |
| **Samples** | 178 |
| **Features** | 13 numeric chemical features |
| **True classes** | 3 cultivar classes (used only for post-hoc evaluation, not training) |
| **Task type** | Unsupervised Clustering |
| **Labels during training** | None |

---

## Notebook Structure

| Section | Description |
|:--------|:------------|
| 1. Setup | Imports and configuration |
| 2. Load the Wine Dataset | Feature matrix, withheld labels, class distribution, descriptive statistics |
| 3. Standardise the Features | `StandardScaler` — why scale matters for Euclidean distance; verify zero mean and unit variance |
| 4. Exploratory Visualisation (PCA) | Project to 2D via PCA; plot unlabelled data (as the algorithm sees it) then true labels side-by-side |
| 5. $k$-Means from Scratch | `distance`, `assign_label`, `assign_clusters`, `update_centers`, `k_means_clustering` |
| 6. Step-by-Step Convergence | Plot cluster assignments after iterations 1, 2, 3, and 10 in a 2×2 grid |
| 7. Full $k$-Means ($k = 3$) | 100-iteration run; side-by-side comparison of discovered clusters vs. true cultivar labels |
| 8. Evaluate Cluster Quality | ARI and Silhouette Score for the $k=3$ result |
| 9. Choosing $k$ — Elbow Method | Sweep $k = 1$–$10$; WCSS, ARI, and Silhouette Score plotted together; all three point to $k = 3$ |
| 10. Effect of $k$ on Assignments | 2×2 grid showing PCA projections for $k = 2, 3, 4, 5$ |
| 11. From-Scratch vs. sklearn | Side-by-side ARI and Silhouette comparison; PCA scatter for both implementations |
| 12. Summary | Algorithm recap, why standardisation matters, strengths and weaknesses of $k$-Means |

---

## Key Ideas

### The $k$-Means Algorithm

Starting from $k$ randomly chosen centres, the algorithm alternates between two steps until convergence:

1. **Assignment** — assign each point to its nearest centre by Euclidean distance:
$$c_i = \arg\min_j \| x_i - \mu_j \|^2$$

2. **Update** — recompute each centre as the mean of all points assigned to it:
$$\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i$$

Convergence is guaranteed because the Within-Cluster Sum of Squares (WCSS) can only decrease or stay the same at each step. However, the solution may be a **local minimum** — the result depends on the random initialisation. Production implementations (such as `sklearn`) run multiple restarts and keep the best result.

---

### Why standardization is essential

$k$-Means uses Euclidean distance, which is not scale-invariant. Proline (range ≈ 278–1680) would contribute distances ~100× larger than ash (range ≈ 1.4–3.2), effectively making the clustering a single-feature problem. Standardising to zero mean and unit variance ensures all 13 features contribute equally.

---

### Choosing $k$: Three Diagnostics

| Diagnostic | What it measures | Requires ground truth? |
|:-----------|:-----------------|:----------------------:|
| **WCSS (Elbow)** | Total squared distance from each point to its centre — lower is better | No |
| **Adjusted Rand Index (ARI)** | Overlap between discovered clusters and true labels; 1.0 = perfect, 0 = random | Yes |
| **Silhouette Score** | How similar each point is to its own cluster vs. the nearest other cluster; range −1 to 1, higher is better | No |

In practice, only WCSS and Silhouette are available without labels. ARI is included here as a learning tool because we happen to know the true cultivar classes. All three diagnostics correctly suggest $k = 3$ for this dataset.

---

### WCSS (Inertia)

$$\text{WCSS}(k) = \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2$$

WCSS always decreases as $k$ increases (more clusters → smaller distances). The **elbow** — the point where the rate of decrease sharply slows — indicates the natural number of clusters.

---

### PCA for Visualisation

The Wine dataset lives in $\mathbb{R}^{13}$, which cannot be plotted directly. PCA is used to project onto the first two principal components **for visualisation only** — the from-scratch $k$-Means algorithm runs in the full 13-dimensional standardised space. The 2D projection is not shown to the clustering algorithm at any point.

---
