"""
Scikit-learn performance: n_jobs parallelism, warm_start, efficient computation
"""

import numpy as np
import time
from sklearn.datasets import load_digits, make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def main():
    print("=" * 60)
    print("Performance Optimization: n_jobs, warm_start, efficiency")
    print("=" * 60)

    print("\n[1] n_jobs parallelism:")
    X, y = load_digits(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    scores = cross_val_score(clf, X, y, cv=3, n_jobs=-1)
    t1 = time.perf_counter()
    print(f"    CV with n_jobs=-1: {t1-t0:.3f}s, mean={scores.mean():.4f}")

    print("\n[2] warm_start for incremental training:")
    gb = GradientBoostingClassifier(n_estimators=10, warm_start=True, random_state=42)
    gb.fit(X[:500], y[:500])
    gb.n_estimators = 20
    gb.fit(X[:500], y[:500])
    print(f"    n_estimators after warm_start: {gb.n_estimators}")
    print(f"    Fitted: {gb.n_estimators_}")

    print("\n[3] Pipeline with n_jobs in GridSearchCV:")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", random_state=42))
    ])
    param_grid = {"clf__C": [0.1, 1], "clf__gamma": ["scale", "auto"]}
    gs = GridSearchCV(pipe, param_grid, cv=2, n_jobs=-1, verbose=0)
    gs.fit(X[:200], y[:200])
    print(f"    Best params: {gs.best_params_}")
    print(f"    Best score: {gs.best_score_:.4f}")

    print("\n[4] Sparse vs dense for text-like data:")
    from scipy.sparse import csr_matrix
    X_dense = np.random.rand(1000, 100)
    X_sparse = csr_matrix(X_dense)
    clf2 = RandomForestClassifier(n_estimators=20, random_state=42)
    t_d = time.perf_counter()
    clf2.fit(X_dense, np.random.randint(0, 2, 1000))
    t_d = time.perf_counter() - t_d
    print(f"    Dense fit time: {t_d:.4f}s")
    print("    Note: RF prefers dense; use HashingVectorizer for sparse text")

    print("\n[5] Efficient preprocessing in pipeline:")
    pipe2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=30, n_jobs=-1, random_state=42))
    ])
    pipe2.fit(X, y)
    print(f"    Single fit+transform+predict chain: OK")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
