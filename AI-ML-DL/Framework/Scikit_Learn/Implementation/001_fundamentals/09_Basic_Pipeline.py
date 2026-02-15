"""
Scikit-learn Basic Pipeline
Pipeline with StandardScaler and LogisticRegression
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def main():
    print("=" * 60)
    print("Basic Pipeline: Preprocessing + Estimator")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("\n[1] Creating Pipeline:")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    print(f"    pipe.steps: {[s[0] for s in pipe.steps]}")

    print("\n[2] fit() - fits all steps in sequence:")
    pipe.fit(X_train, y_train)
    print("    scaler.fit_transform(X_train) -> clf.fit(scaled_X, y_train)")

    print("\n[3] predict() - transforms then predicts:")
    y_pred = pipe.predict(X_test)
    print(f"    Predictions (first 10): {y_pred[:10]}")

    print("\n[4] score() - end-to-end evaluation:")
    acc = pipe.score(X_test, y_test)
    print(f"    Accuracy: {acc:.4f}")

    print("\n[5] Accessing steps by name:")
    scaler = pipe["scaler"]
    clf = pipe["clf"]
    print(f"    pipe['scaler']: {type(scaler).__name__}")
    print(f"    pipe['clf']: {type(clf).__name__}")
    print(f"    Scaler mean_: {scaler.mean_}")

    print("\n[6] Pipeline with fit_transform flow:")
    print("    Each step's fit_transform feeds into next step's fit")
    print("    Last step only needs fit (classifier/regressor)")

    print("\n[7] Named steps for clarity:")
    print("    ('scaler', StandardScaler()) - human-readable step names")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
