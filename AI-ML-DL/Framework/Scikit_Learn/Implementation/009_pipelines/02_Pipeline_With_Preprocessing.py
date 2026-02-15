"""
Scikit-learn Pipeline with preprocessing: Scaler + model pipeline, fit/predict
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Pipeline with Preprocessing: Scaler + Model")
    print("=" * 60)

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] StandardScaler + LogisticRegression:")
    pipe_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe_lr.fit(X_train, y_train)
    pred_lr = pipe_lr.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_lr):.4f}")

    print("\n[2] MinMaxScaler + SVC:")
    pipe_svc = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", SVC(kernel="rbf", random_state=42)),
    ])
    pipe_svc.fit(X_train, y_train)
    pred_svc = pipe_svc.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_svc):.4f}")

    print("\n[3] fit vs fit_transform flow:")
    pipe_lr.fit(X_train, y_train)
    X_test_scaled = pipe_lr.named_steps["scaler"].transform(X_test)
    pred_direct = pipe_lr.named_steps["clf"].predict(X_test_scaled)
    print(f"    Manual flow accuracy: {accuracy_score(y_test, pred_direct):.4f}")
    print(f"    Pipeline predict matches: {np.allclose(pred_lr, pred_direct)}")

    print("\n[4] predict_proba through pipeline:")
    proba = pipe_lr.predict_proba(X_test)
    print(f"    predict_proba shape: {proba.shape}")
    print(f"    First sample probabilities: {proba[0].round(4)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
