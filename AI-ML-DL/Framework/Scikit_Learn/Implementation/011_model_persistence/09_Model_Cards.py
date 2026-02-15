"""
Scikit-learn model cards: model documentation, content structure
"""

import json
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def main():
    print("=" * 60)
    print("Model Cards: documentation structure")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    acc = (model.predict(X_test) == y_test).mean()

    print("\n[1] Model card structure:")
    model_card = {
        "model_details": {
            "name": "iris_rf_classifier",
            "version": "1.0.0",
            "type": "RandomForestClassifier",
            "date": datetime.now().isoformat(),
        },
        "intended_use": {
            "primary_use": "Iris species classification",
            "out_of_scope": "Other flower species",
        },
        "training_data": {
            "dataset": "Iris",
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X_train.shape[1],
        },
        "metrics": {
            "test_accuracy": float(acc),
        },
        "limitations": [
            "Trained on Iris only",
            "Assumes 4 numeric features",
        ],
        "ethical_considerations": "Low risk, botanical classification",
    }
    print("    model_details, intended_use, training_data, metrics, limitations")

    print("\n[2] Save model card:")
    with open("model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)
    print("    Saved model_card.json")

    print("\n[3] Key sections:")
    for section in model_card:
        print(f"    - {section}")

    print("\n[4] Model card best practices:")
    print("    - Document intended use and limitations")
    print("    - Include training data description")
    print("    - Report metrics and evaluation conditions")
    print("    - Note ethical considerations")

    import os
    if os.path.exists("model_card.json"):
        os.remove("model_card.json")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
