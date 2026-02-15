"""
Scikit-learn Feature importance across ensemble methods
"""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    print("=" * 60)
    print("Feature Importance Across Ensemble Methods")
    print("=" * 60)

    X, y = load_breast_cancer(return_X_y=True)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    }

    print("\n[1] Feature importance by model:")
    print("-" * 60)
    for name, model in models.items():
        model.fit(X_train, y_train)
        imp = model.feature_importances_
        top5_idx = np.argsort(imp)[-5:][::-1]
        print(f"\n{name} - Top 5 features:")
        for idx in top5_idx:
            print(f"    {feature_names[idx]}: {imp[idx]:.4f}")

    print("\n[2] Correlation of importance across models:")
    importances = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        importances[name] = model.feature_importances_
    rf_imp = importances["RandomForest"]
    gb_imp = importances["GradientBoosting"]
    corr = np.corrcoef(rf_imp, gb_imp)[0, 1]
    print(f"    RandomForest vs GradientBoosting correlation: {corr:.4f}")

    print("\n[3] Most consistently important features:")
    imp_matrix = np.column_stack([importances[n] for n in models])
    mean_imp = imp_matrix.mean(axis=1)
    top5 = np.argsort(mean_imp)[-5:][::-1]
    for idx in top5:
        print(f"    {feature_names[idx]}: mean={mean_imp[idx]:.4f}")

    print("\n[4] Accuracy with top-k features:")
    for k in [5, 10, 20]:
        top_k_idx = np.argsort(mean_imp)[-k:]
        X_train_k = X_train[:, top_k_idx]
        X_test_k = X_test[:, top_k_idx]
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_k, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test_k))
        print(f"    Top {k} features: Accuracy = {acc:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
