"""
Scikit-learn pipeline inspection: named_steps, get_params, set_params, indexing
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("Pipeline Inspection: named_steps, get_params, set_params")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])

    print("\n[1] named_steps:")
    for name, step in pipe.named_steps.items():
        print(f"    {name}: {type(step).__name__}")

    print("\n[2] get_params (flat):")
    params = pipe.get_params()
    for k in ["scaler", "clf", "clf__C", "clf__max_iter"]:
        if k in params:
            print(f"    {k}: {params[k]}")

    print("\n[3] set_params:")
    pipe.set_params(clf__C=0.1, clf__max_iter=500)
    print(f"    clf.C after set_params: {pipe.named_steps['clf'].C}")
    print(f"    clf.max_iter after set_params: {pipe.named_steps['clf'].max_iter}")

    print("\n[4] Indexing pipe[step_name]:")
    scaler = pipe["scaler"]
    clf = pipe["clf"]
    print(f"    pipe['scaler'] type: {type(scaler).__name__}")
    print(f"    pipe['clf'] type: {type(clf).__name__}")

    print("\n[5] steps attribute:")
    for i, (name, step) in enumerate(pipe.steps):
        print(f"    Step {i}: {name} -> {type(step).__name__}")

    print("\n[6] Nested param access via get_params:")
    pipe.set_params(clf__solver="lbfgs")
    print(f"    clf.solver: {pipe.get_params()['clf__solver']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
