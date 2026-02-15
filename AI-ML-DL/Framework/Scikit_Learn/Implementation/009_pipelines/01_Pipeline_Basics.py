"""
Scikit-learn Pipeline basics: Pipeline, make_pipeline, named_steps access
"""

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Pipeline Basics: Pipeline, make_pipeline, named_steps")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] Pipeline with explicit step names:")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred):.4f}")

    print("\n[2] make_pipeline (auto-generated names):")
    pipe_auto = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, random_state=42))
    pipe_auto.fit(X_train, y_train)
    pred_auto = pipe_auto.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_auto):.4f}")
    print(f"    Step names: {[s[0] for s in pipe_auto.steps]}")

    print("\n[3] named_steps access:")
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    print(f"    Scaler mean (first feature): {scaler.mean_[0]:.4f}")
    print(f"    Classifier classes: {clf.classes_}")

    print("\n[4] Indexing by position:")
    step_0 = pipe[0]
    step_1 = pipe[1]
    print(f"    Step 0 type: {type(step_0).__name__}")
    print(f"    Step 1 type: {type(step_1).__name__}")

    print("\n[5] Slicing pipeline:")
    pipe_preprocess = pipe[:-1]
    X_transformed = pipe_preprocess.transform(X_train)
    print(f"    Preprocessing only output shape: {X_transformed.shape}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
