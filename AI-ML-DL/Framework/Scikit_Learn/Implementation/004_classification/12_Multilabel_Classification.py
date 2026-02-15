"""
Scikit-learn Multilabel Classification: MultiLabelBinarizer, multi-label strategies
"""

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, jaccard_score


def main():
    print("=" * 60)
    print("Multilabel Classification: MultiLabelBinarizer, strategies")
    print("=" * 60)

    X, y = make_multilabel_classification(
        n_samples=200, n_features=10, n_classes=5, n_labels=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n[1] MultiLabelBinarizer - encode multi-label targets:")
    mlb = MultiLabelBinarizer()
    y_binary = mlb.fit_transform([(0, 1), (1, 2), (0,), (2,)])
    print("    Input: [(0,1), (1,2), (0,), (2,)]")
    print("    Output:\n", y_binary)
    print("    classes_:", mlb.classes_)

    print("\n[2] MultiOutputClassifier - independent classifier per label:")
    moc = MultiOutputClassifier(LogisticRegression(random_state=42))
    moc.fit(X_train, y_train)
    y_pred = moc.predict(X_test)
    print(f"    Hamming loss: {hamming_loss(y_test, y_pred):.4f}")
    print(f"    Jaccard score: {jaccard_score(y_test, y_pred, average='samples'):.4f}")

    print("\n[3] ClassifierChain - chain classifiers, use prior predictions:")
    cc = ClassifierChain(LogisticRegression(random_state=42), order=None)
    cc.fit(X_train, y_train)
    y_pred_cc = cc.predict(X_test)
    print(f"    Hamming loss: {hamming_loss(y_test, y_pred_cc):.4f}")
    print(f"    Jaccard score: {jaccard_score(y_test, y_pred_cc, average='samples'):.4f}")

    print("\n[4] ClassifierChain with ordered chain:")
    cc_ord = ClassifierChain(LogisticRegression(random_state=42), order=[0, 1, 2, 3, 4])
    cc_ord.fit(X_train, y_train)
    print(f"    Order: {cc_ord.order}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
