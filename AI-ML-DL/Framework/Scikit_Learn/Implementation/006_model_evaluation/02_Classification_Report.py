"""
Scikit-learn Classification Report: classification_report, per-class metrics, support
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():
    print("=" * 60)
    print("Classification Report: per-class metrics and support")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    target_names = ["setosa", "versicolor", "virginica"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)

    print("\n[1] Default classification_report:")
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)

    print("\n[2] With zero_division=0:")
    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    print(report)

    print("\n[3] output_dict=True:")
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print("    Keys:", list(report_dict.keys()))
    print("    Macro avg precision:", report_dict["macro avg"]["precision"])
    print("    Weighted avg f1-score:", report_dict["weighted avg"]["f1-score"])

    print("\n[4] digits parameter (2 decimal places):")
    report = classification_report(y_test, y_pred, target_names=target_names, digits=2)
    print(report)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
