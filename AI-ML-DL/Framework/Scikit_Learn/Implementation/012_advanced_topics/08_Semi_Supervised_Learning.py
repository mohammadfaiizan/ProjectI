"""
Scikit-learn semi-supervised learning: LabelPropagation, LabelSpreading
"""

import numpy as np
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Semi-Supervised Learning: LabelPropagation, LabelSpreading")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    rng = np.random.RandomState(42)
    n_unlabeled = 70
    unlabeled_idx = rng.choice(len(X_train), n_unlabeled, replace=False)
    y_train_semi = y_train.copy()
    y_train_semi[unlabeled_idx] = -1

    print("\n[1] LabelPropagation (rbf kernel, gamma):")
    lp = LabelPropagation(kernel="rbf", gamma=0.25)
    lp.fit(X_train, y_train_semi)
    pred_lp = lp.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_lp):.4f}")
    print(f"    Labeled: {np.sum(y_train_semi >= 0)}, Unlabeled: {n_unlabeled}")

    print("\n[2] LabelSpreading (rbf kernel, alpha for regularization):")
    ls = LabelSpreading(kernel="rbf", gamma=0.25, alpha=0.2)
    ls.fit(X_train, y_train_semi)
    pred_ls = ls.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_ls):.4f}")

    print("\n[3] LabelSpreading with kNN kernel (n_neighbors):")
    ls_knn = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)
    ls_knn.fit(X_train, y_train_semi)
    pred_knn = ls_knn.predict(X_test)
    print(f"    Accuracy: {accuracy_score(y_test, pred_knn):.4f}")

    print("\n[4] Compare with fully supervised baseline:")
    lp_full = LabelPropagation(kernel="rbf", gamma=0.25)
    lp_full.fit(X_train, y_train)
    print(f"    Full labels accuracy: {accuracy_score(y_test, lp_full.predict(X_test)):.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
