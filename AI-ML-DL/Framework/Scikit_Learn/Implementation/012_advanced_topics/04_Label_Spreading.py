"""
Scikit-learn LabelSpreading: semi-supervised with alpha regularization
"""

import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("LabelSpreading: semi-supervised with alpha regularization")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rng = np.random.RandomState(42)
    n_unlabeled = 60
    unlabeled_idx = rng.choice(len(X_train), n_unlabeled, replace=False)
    y_train_semi = y_train.copy()
    y_train_semi[unlabeled_idx] = -1

    print("\n[1] LabelSpreading with RBF kernel:")
    ls = LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2)
    ls.fit(X_train, y_train_semi)
    pred = ls.predict(X_test)
    print(f"    alpha=0.2: Accuracy = {accuracy_score(y_test, pred):.4f}")

    print("\n[2] Effect of alpha (clamping factor):")
    for alpha in [0.1, 0.2, 0.5, 0.8]:
        ls = LabelSpreading(kernel="rbf", gamma=0.25, alpha=alpha)
        ls.fit(X_train, y_train_semi)
        acc = accuracy_score(y_test, ls.predict(X_test))
        print(f"    alpha={alpha}: Accuracy = {acc:.4f}")

    print("\n[3] LabelSpreading with kNN kernel:")
    ls_knn = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)
    ls_knn.fit(X_train, y_train_semi)
    pred_knn = ls_knn.predict(X_test)
    print(f"    n_neighbors=7: Accuracy = {accuracy_score(y_test, pred_knn):.4f}")

    print("\n[4] transduction_ (fitted labels for training set):")
    ls = LabelSpreading(kernel="rbf", alpha=0.2)
    ls.fit(X_train, y_train_semi)
    print(f"    transduction_ shape: {ls.transduction_.shape}")
    print(f"    Match with true (labeled only): {np.mean(ls.transduction_[y_train_semi >= 0] == y_train[y_train_semi >= 0]):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()