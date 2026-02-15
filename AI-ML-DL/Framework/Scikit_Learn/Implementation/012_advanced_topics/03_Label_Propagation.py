"""
Scikit-learn LabelPropagation: semi-supervised learning with RBF and kNN kernels
"""

import numpy as np
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("LabelPropagation: semi-supervised with RBF and kNN kernels")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rng = np.random.RandomState(42)
    n_unlabeled = 60
    unlabeled_idx = rng.choice(len(X_train), n_unlabeled, replace=False)
    y_train_semi = y_train.copy()
    y_train_semi[unlabeled_idx] = -1

    print("\n[1] LabelPropagation with RBF kernel:")
    lp_rbf = LabelPropagation(kernel="rbf", gamma=0.25)
    lp_rbf.fit(X_train, y_train_semi)
    pred = lp_rbf.predict(X_test)
    print(f"    gamma=0.25: Accuracy = {accuracy_score(y_test, pred):.4f}")

    print("\n[2] LabelPropagation with kNN kernel:")
    lp_knn = LabelPropagation(kernel="knn", n_neighbors=7)
    lp_knn.fit(X_train, y_train_semi)
    pred_knn = lp_knn.predict(X_test)
    print(f"    n_neighbors=7: Accuracy = {accuracy_score(y_test, pred_knn):.4f}")

    print("\n[3] Effect of gamma (RBF):")
    for gamma in [0.1, 0.25, 0.5, 1.0]:
        lp = LabelPropagation(kernel="rbf", gamma=gamma)
        lp.fit(X_train, y_train_semi)
        acc = accuracy_score(y_test, lp.predict(X_test))
        print(f"    gamma={gamma}: Accuracy = {acc:.4f}")

    print("\n[4] label_distributions_ (soft labels):")
    lp = LabelPropagation(kernel="rbf", gamma=0.25)
    lp.fit(X_train, y_train_semi)
    print(f"    Shape: {lp.label_distributions_.shape}")
    print(f"    Sample (first unlabeled): {lp.label_distributions_[unlabeled_idx[0]]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()