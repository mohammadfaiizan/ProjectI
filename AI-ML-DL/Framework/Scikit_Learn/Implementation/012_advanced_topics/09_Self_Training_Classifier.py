"""
Scikit-learn SelfTrainingClassifier: threshold, criterion, max_iter
"""

import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("SelfTrainingClassifier: threshold, criterion, max_iter")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rng = np.random.RandomState(42)
    n_unlabeled = 60
    unlabeled_idx = rng.choice(len(X_train), n_unlabeled, replace=False)
    y_train_semi = y_train.copy()
    y_train_semi[unlabeled_idx] = -1

    print("\n[1] SelfTrainingClassifier with threshold:")
    base = LogisticRegression(max_iter=500, random_state=42)
    st = SelfTrainingClassifier(base, threshold=0.9, criterion="threshold")
    st.fit(X_train, y_train_semi)
    pred = st.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"    Labeled samples: {st.labeled_iter_}")

    print("\n[2] SelfTrainingClassifier with k_best criterion:")
    st_k = SelfTrainingClassifier(base, k=10, criterion="k_best")
    st_k.fit(X_train, y_train_semi)
    pred_k = st_k.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_k):.4f}")

    print("\n[3] max_iter to limit self-training rounds:")
    st_lim = SelfTrainingClassifier(base, threshold=0.85, max_iter=5)
    st_lim.fit(X_train, y_train_semi)
    print(f"    Max iterations: 5, labeled_iter: {st_lim.labeled_iter_}")

    print("\n[4] SelfTrainingClassifier with DecisionTree (no predict_proba):")
    st_dt = SelfTrainingClassifier(
        DecisionTreeClassifier(random_state=42),
        threshold=0.9,
        criterion="threshold",
    )
    st_dt.fit(X_train, y_train_semi)
    pred_dt = st_dt.predict(X_test)
    print(f"    Accuracy (DT base): {accuracy_score(y_test, pred_dt):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
