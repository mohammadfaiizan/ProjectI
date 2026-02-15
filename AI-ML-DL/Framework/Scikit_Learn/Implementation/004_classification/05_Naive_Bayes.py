"""
Scikit-learn Naive Bayes: GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Naive Bayes: GaussianNB, MultinomialNB, BernoulliNB, ComplementNB")
    print("=" * 60)

    X_iris, y_iris = load_iris(return_X_y=True)
    X_digits, y_digits = load_digits(return_X_y=True)

    X_iris_train, X_iris_test, y_iris_train, y_iris_test = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=42
    )
    X_dig_train, X_dig_test, y_dig_train, y_dig_test = train_test_split(
        X_digits, y_digits, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_dig_train_scaled = scaler.fit_transform(X_dig_train)
    X_dig_test_scaled = scaler.transform(X_dig_test)

    print("\n[1] GaussianNB - continuous features (Iris):")
    gnb = GaussianNB()
    gnb.fit(X_iris_train, y_iris_train)
    print(f"    Accuracy: {accuracy_score(y_iris_test, gnb.predict(X_iris_test)):.4f}")

    print("\n[2] MultinomialNB - count data (Digits):")
    mnb = MultinomialNB()
    mnb.fit(X_dig_train_scaled, y_dig_train)
    print(f"    Accuracy: {accuracy_score(y_dig_test, mnb.predict(X_dig_test_scaled)):.4f}")

    print("\n[3] BernoulliNB - binary features:")
    bnb = BernoulliNB(alpha=0.1)
    X_binary = (X_dig_train_scaled > 0.5).astype(int)
    X_binary_test = (X_dig_test_scaled > 0.5).astype(int)
    bnb.fit(X_binary, y_dig_train)
    print(f"    Accuracy: {accuracy_score(y_dig_test, bnb.predict(X_binary_test)):.4f}")

    print("\n[4] ComplementNB - for imbalanced classes:")
    cnb = ComplementNB()
    cnb.fit(X_dig_train_scaled, y_dig_train)
    print(f"    Accuracy: {accuracy_score(y_dig_test, cnb.predict(X_dig_test_scaled)):.4f}")

    print("\n[5] Comparison on Digits:")
    for name, model in [("GaussianNB", GaussianNB()), ("MultinomialNB", MultinomialNB())]:
        m = model
        m.fit(X_dig_train_scaled, y_dig_train)
        acc = accuracy_score(y_dig_test, m.predict(X_dig_test_scaled))
        print(f"    {name}: {acc:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
