"""
Scikit-learn model interpretation: SHAP TreeExplainer, LIME concepts
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def main():
    print("=" * 60)
    print("SHAP TreeExplainer and LIME concepts with sklearn models")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

    print("\n[1] Tree-based model for SHAP:")
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    print(f"    Model: {clf.__class__.__name__}")

    if SHAP_AVAILABLE:
        print("\n[2] SHAP TreeExplainer:")
        explainer = shap.TreeExplainer(clf, X_train)
        shap_values = explainer.shap_values(X_test[:5])
        if isinstance(shap_values, list):
            print(f"    Multiclass: {len(shap_values)} arrays")
            print(f"    Class 0 SHAP shape: {shap_values[0].shape}")
        else:
            print(f"    SHAP values shape: {shap_values.shape}")
        print(f"    Base value: {explainer.expected_value}")

        print("\n[3] Summary plot (computed):")
        sv = shap_values[0] if isinstance(shap_values, list) else shap_values
        mean_abs = np.abs(sv).mean(axis=0)
        for i, name in enumerate(feature_names):
            print(f"    {name}: mean |SHAP| = {mean_abs[i]:.4f}")
    else:
        print("\n[2] SHAP not installed. pip install shap")
        print("    LIME: pip install lime")

    print("\n[4] LIME concepts (local linear approximation):")
    print("    - Perturb instances around a point")
    print("    - Fit weighted linear model to predictions")
    print("    - Coefficients = local feature importance")
    print("    - Use lime.lime_tabular.LimeTabularExplainer for sklearn")

    print("\n[5] Built-in feature_importances_ (sklearn):")
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    imp = dt.feature_importances_
    for i, name in enumerate(feature_names):
        print(f"    {name}: {imp[i]:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
