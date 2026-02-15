"""
Scikit-learn Validation Curve: validation_curve, ValidationCurveDisplay
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import validation_curve, ValidationCurveDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("Validation Curve")
    print("=" * 60)

    X, y = load_digits(return_X_y=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(random_state=42)),
    ])

    print("\n[1] validation_curve (param_name='clf__gamma'):")
    param_range = np.logspace(-6, -1, 5)
    train_scores, test_scores = validation_curve(
        pipe, X, y, param_name="clf__gamma", param_range=param_range, cv=5,
    )
    print(f"    param_range: {param_range}")
    print(f"    train_scores shape: {train_scores.shape}")
    print(f"    test_scores shape: {test_scores.shape}")

    print("\n[2] Mean scores per gamma:")
    for i, gamma in enumerate(param_range):
        tr_mean = train_scores[i].mean()
        te_mean = test_scores[i].mean()
        print(f"    gamma={gamma:.2e}: train={tr_mean:.4f}, test={te_mean:.4f}")

    print("\n[3] ValidationCurveDisplay.from_estimator:")
    fig, ax = plt.subplots()
    ValidationCurveDisplay.from_estimator(
        pipe, X, y, param_name="clf__gamma", param_range=param_range, cv=5, ax=ax,
    )
    plt.tight_layout()
    plt.savefig("validation_curve.png", dpi=100)
    plt.close()
    print("    Saved validation_curve.png")

    print("\n[4] validation_curve (param_name='clf__C'):")
    param_range_c = [0.01, 0.1, 1.0, 10.0, 100.0]
    train_scores, test_scores = validation_curve(
        pipe, X, y, param_name="clf__C", param_range=param_range_c, cv=3,
    )
    best_idx = test_scores.mean(axis=1).argmax()
    print(f"    Best C: {param_range_c[best_idx]} (test mean={test_scores.mean(axis=1)[best_idx]:.4f})")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
