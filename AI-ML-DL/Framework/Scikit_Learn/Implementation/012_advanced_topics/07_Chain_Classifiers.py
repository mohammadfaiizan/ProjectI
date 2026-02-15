"""
Scikit-learn ClassifierChain for multi-label classification, order parameter
"""

import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss


def main():
    print("=" * 60)
    print("ClassifierChain: multi-label with label dependencies")
    print("=" * 60)

    X, y = make_multilabel_classification(
        n_samples=300, n_features=20, n_labels=4, n_classes=6, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] ClassifierChain with default order [0,1,2,3]:")
    chain = ClassifierChain(LogisticRegression(max_iter=500, random_state=42), order=[0, 1, 2, 3])
    chain.fit(X_train, y_train)
    pred = chain.predict(X_test)
    f1 = f1_score(y_test, pred, average="samples")
    print(f"    F1 (samples): {f1:.4f}")
    print(f"    Hamming loss: {hamming_loss(y_test, pred):.4f}")

    print("\n[2] ClassifierChain with random order:")
    chain_rand = ClassifierChain(LogisticRegression(max_iter=500, random_state=42), order=None)
    chain_rand.fit(X_train, y_train)
    pred_rand = chain_rand.predict(X_test)
    f1_rand = f1_score(y_test, pred_rand, average="samples")
    print(f"    F1 (samples): {f1_rand:.4f}")
    print(f"    order: {chain_rand.order_}")

    print("\n[3] cv parameter for multiple random chains (ensemble):")
    chain_cv = ClassifierChain(
        LogisticRegression(max_iter=500, random_state=42), order=None, cv=3
    )
    chain_cv.fit(X_train, y_train)
    pred_cv = chain_cv.predict(X_test)
    print(f"    F1 (samples): {f1_score(y_test, pred_cv, average='samples'):.4f}")

    print("\n[4] Access chain estimators:")
    for i, est in enumerate(chain.estimators_):
        print(f"    Step {i}: {type(est).__name__}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
