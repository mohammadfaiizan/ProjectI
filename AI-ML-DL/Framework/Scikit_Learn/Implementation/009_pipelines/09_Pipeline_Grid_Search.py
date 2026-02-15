"""
Scikit-learn Pipeline with GridSearchCV: param naming __ convention
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Pipeline GridSearchCV: param naming __ convention")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=42)),
        ("clf", SVC(random_state=42)),
    ])

    param_grid = {
        "pca__n_components": [2, 3, 4],
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", "auto"],
    }

    print("\n[1] GridSearchCV with pipeline:")
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"    Best score (CV): {gs.best_score_:.4f}")
    print(f"    Best params: {gs.best_params_}")

    print("\n[2] __ convention for nested params:")
    print("    pca__n_components -> PCA.n_components")
    print("    clf__C -> SVC.C")
    print("    clf__gamma -> SVC.gamma")

    print("\n[3] Best estimator predictions:")
    pred = gs.predict(X_test)
    print(f"    Test accuracy: {accuracy_score(y_test, pred):.4f}")

    print("\n[4] cv_results_ (top 3):")
    results = gs.cv_results_
    for i in range(min(3, len(results["params"]))):
        print(f"    {results['params'][i]} -> {results['mean_test_score'][i]:.4f}")

    print("\n[5] Access best pipeline steps:")
    best_pca = gs.best_estimator_.named_steps["pca"]
    print(f"    Best n_components: {best_pca.n_components_}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
