# Estimator API and Data Fundamentals

---

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Estimator API Basics](#estimator-api-basics)
- [Data Representations](#data-representations)
- [Toy Datasets](#toy-datasets)
- [Real-World Datasets](#real-world-datasets)
- [Generated Datasets](#generated-datasets)

---

## Installation and Setup

### Requirements

Scikit-learn requires **Python 3.9+** and core scientific Python libraries:

| Dependency | Purpose |
|------------|---------|
| NumPy | Array operations, numerical computation |
| SciPy | Sparse matrices, scientific routines |
| joblib | Parallelization, model persistence |
| threadpoolctl | Thread pool control |

### Installation

```python
pip install scikit-learn
```

### Version and Environment Check

Use **show_versions()** to print the full environment including sklearn version and all dependencies:

```python
import sklearn
print(sklearn.__version__)
sklearn.show_versions()
```

### Verification

```python
from sklearn.linear_model import LinearRegression
import numpy as np
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])
model = LinearRegression().fit(X, y)
model.predict([[4]])  # Should return ~8
```

---

## Estimator API Basics

Scikit-learn uses a consistent **estimator API** across all algorithms. Every estimator implements the same interface.

### Core Methods

| Method | Purpose | Used By |
|--------|---------|---------|
| **fit(X, y)** | Learn from data | All estimators |
| **predict(X)** | Predict targets | Classifiers, regressors |
| **transform(X)** | Transform data | Transformers (preprocessing) |
| **score(X, y)** | Evaluate performance | Supervised estimators |

### fit()

**fit()** learns parameters from the training data. For supervised estimators, it takes both **X** (features) and **y** (targets). For unsupervised estimators, only **X** is required.

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
```

### predict()

**predict()** outputs predictions for new samples. Used by classifiers (class labels) and regressors (continuous values).

```python
y_pred = clf.predict(X_test)
```

### transform()

**transform()** applies a learned transformation. Used by **transformers** such as StandardScaler, PCA. The fitted parameters (e.g., mean, std) are applied to new data.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_scaled = scaler.transform(X_test)
```

### score()

**score()** returns a default metric: **accuracy** for classification, **R-squared** for regression.

```python
accuracy = clf.score(X_test, y_test)
```

### Convenience Methods

- **fit_transform(X)**: Equivalent to `fit(X)` followed by `transform(X)` for transformers.
- **fit_predict(X)**: Equivalent to `fit(X)` followed by `predict(X)` for clustering.

---

## Data Representations

### Feature Matrix X

The **feature matrix** is always a 2D array of shape **(n_samples, n_features)**. Samples are stored in rows; features in columns. Convention: **row-major (C-order)**.

```python
X.shape  # (150, 4) for iris: 150 samples, 4 features
X[0]     # First sample's feature vector
```

### Target Vector y

The **target vector** is a 1D array of shape **(n_samples,)**. For classification, values are integer class labels. For regression, values are continuous.

```python
y.shape   # (150,)
np.unique(y)  # Unique classes or value range
```

### Sparse Matrices

For high-dimensional sparse data (e.g., text), use **scipy.sparse** matrices. **CSR** (Compressed Sparse Row) is the most common format. Many estimators accept sparse input (e.g., SGDClassifier, MultinomialNB).

```python
from scipy import sparse
X_sparse = sparse.csr_matrix(X_dense)
# X_sparse.nnz gives number of non-zero elements
```

### Data Types

- **X**: float64 (or float32 for memory efficiency)
- **y**: int for classification, float for regression
- Sparse matrices supported where documented

---

## Toy Datasets

Built-in datasets are loaded via **load_*** functions. They return a **Bunch** object (dictionary-like) with keys: `data`, `target`, `feature_names`, `target_names`, `DESCR`.

### load_iris

Classic 3-class classification: 150 samples, 4 features (sepal/petal length and width).

```python
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
# iris.feature_names, iris.target_names
```

### load_digits

Handwritten digits 0-9: 1797 samples, 64 features (8x8 pixel images flattened).

```python
from sklearn.datasets import load_digits
digits = load_digits()
```

### load_wine

Wine classification: 178 samples, 13 chemical features, 3 classes.

### load_breast_cancer

Binary classification: malignant vs benign. 569 samples, 30 features.

### load_diabetes

Regression: 442 samples, 10 features. Target is disease progression.

### return_X_y

Use **return_X_y=True** to get only (data, target) as a tuple:

```python
X, y = load_iris(return_X_y=True)
```

---

## Real-World Datasets

These functions **download** data from the internet and cache it locally.

### fetch_openml

Fetches datasets from **OpenML**. Supports thousands of datasets (e.g., MNIST, Titanic).

```python
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
# mnist.data, mnist.target
```

Parameters:
- **as_frame**: Return pandas DataFrame when True
- **parser**: Use `"auto"` for compatibility

### fetch_20newsgroups

Text classification: 20 newsgroup categories. Returns raw text strings.

```python
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset="train", categories=["sci.med", "sci.space"])
# news.data: list of document strings
# news.target: class labels
# subset: "train" | "test" | "all"
# remove=("headers","footers","quotes") for cleaner text
```

### fetch_lfw_people

Labeled Faces in the Wild: face images for classification.

```python
from sklearn.datasets import fetch_lfw_people
lfw = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
# lfw.images: image arrays
# lfw.data: flattened features
```

### Cache Location

```python
from sklearn.datasets import get_data_home
get_data_home()  # Default: ~/scikit_learn_data
```

---

## Generated Datasets

Synthetic data for testing and demonstrations. Use **make_*** functions.

### make_classification

Generates random classification data with controllable structure.

```python
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_classes=3, n_clusters_per_class=1, random_state=42)
```

Key parameters: **n_informative**, **n_redundant**, **n_repeated**, **n_clusters_per_class**.

### make_regression

Generates random regression data.

```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=5, n_informative=3, noise=10)
```

### make_blobs

Generates isotropic Gaussian blobs for clustering.

```python
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
```

### make_moons

Two interleaving half circles; useful for non-linear classification.

```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.1)
```

### make_circles

Concentric circles. **factor** controls inner/outer radius ratio.

```python
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=200, noise=0.05, factor=0.5)
```

### return_X_y

All **make_*** functions support **return_X_y=True** for a simple (X, y) tuple.
