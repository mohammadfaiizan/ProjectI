# Results Analysis and Bayesian Optimization

---

## Table of Contents

- [Analyzing Search Results](#analyzing-search-results)
- [BayesSearchCV and scikit-optimize](#bayessearchcv-and-scikit-optimize)
- [Optuna Integration](#optuna-integration)
- [Visualizing Search Results](#visualizing-search-results)

---

## Analyzing Search Results

### cv_results_

**cv_results_** is a dict of arrays with one entry per parameter combination. Keys include **params**, **mean_test_score**, **std_test_score**, **mean_train_score**, **rank_test_score**, and fold-level scores.

```python
grid.fit(X_train, y_train)
results = grid.cv_results_
print(results["params"][:3])
print(results["mean_test_score"])
print(results["std_test_score"])
```

### best_params_ and best_estimator_

**best_params_** holds the best parameter dict. **best_estimator_** is the refitted model (when `refit=True`). **best_index_** is the index into **cv_results_**.

```python
best_params = grid.best_params_
best_model = grid.best_estimator_
best_idx = grid.best_index_
```

### Top-K Parameter Combinations

Sort by **mean_test_score** to find top candidates:

```python
idx = np.argsort(grid.cv_results_["mean_test_score"])[::-1][:5]
for i in idx:
    print(grid.cv_results_["params"][i], grid.cv_results_["mean_test_score"][i])
```

### Train vs Test Gap

Compare **mean_train_score** and **mean_test_score** to detect overfitting. Large gap suggests overfitting to the validation folds.

```python
gap = (grid.cv_results_["mean_train_score"][grid.best_index_] -
       grid.cv_results_["mean_test_score"][grid.best_index_])
```

---

## BayesSearchCV and scikit-optimize

### Overview

**BayesSearchCV** (from **scikit-optimize**) uses Bayesian optimization to choose parameter combinations. It typically requires fewer evaluations than random search for similar quality.

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

search_spaces = {
    "C": Real(1e-2, 1e2, prior="log-uniform"),
    "gamma": Real(1e-4, 1e-1, prior="log-uniform"),
    "kernel": Categorical(["rbf", "linear"]),
}
bayes_search = BayesSearchCV(
    SVC(),
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    random_state=42,
)
bayes_search.fit(X_train, y_train)
```

### Search Spaces

| Space | Description |
|-------|-------------|
| **Real(low, high, prior)** | Continuous; prior="log-uniform" for scale params |
| **Integer(low, high)** | Discrete integers |
| **Categorical(choices)** | List of options |

### prior="log-uniform"

For parameters like **C** and **gamma**, use **prior="log-uniform"** to sample uniformly in log-space.

### Installation

```bash
pip install scikit-optimize
```

---

## Optuna Integration

### Pattern

**Optuna** is a general-purpose hyperparameter optimization library. Integrate with sklearn by defining an objective that suggests parameters and returns a cross-validation score.

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    C = trial.suggest_float("C", 1e-2, 1e2, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1e-1, log=True)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    clf = SVC(C=C, gamma=gamma, kernel=kernel)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
print(study.best_value)
```

### suggest_float with log=True

**suggest_float(name, low, high, log=True)** samples in log-space, analogous to **loguniform**.

### TPE Sampler

Use **TPESampler** for efficient Bayesian-style search:

```python
from optuna.samplers import TPESampler

study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
```

### Installation

```bash
pip install optuna
```

---

## Visualizing Search Results

### Heatmaps

For two parameters, pivot **cv_results_** and plot a heatmap. Filter by other parameters (e.g., kernel) if needed.

```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.DataFrame(grid.cv_results_)
subset = results[results["param_kernel"] == "rbf"]
pivot = subset.pivot_table(
    values="mean_test_score",
    index="param_gamma",
    columns="param_C",
)
plt.imshow(pivot.values, cmap="viridis")
plt.colorbar()
plt.xticks(range(len(pivot.columns)), pivot.columns)
plt.yticks(range(len(pivot.index)), pivot.index)
plt.xlabel("C")
plt.ylabel("gamma")
plt.show()
```

### Parallel Coordinates

**Parallel coordinates** plot each parameter and score as an axis. Each line is one parameter combination. Use **pd.plotting.parallel_coordinates** or **plotly** for interactive plots.

```python
import pandas as pd

df = pd.DataFrame(grid.cv_results_)
# Select param and score columns, then plot
```

### Validation Curves

For a single parameter, plot **mean_test_score** vs parameter value. Use **validation_curve** for this purpose, or extract from **cv_results_**.

### Ranking Table

Sort by **rank_test_score** and display top combinations in a table for quick inspection.

---
