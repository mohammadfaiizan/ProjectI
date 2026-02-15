"""
Comparing sklearn with XGBoost, LightGBM, CatBoost concepts
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


def main():
    print("=" * 60)
    print("Comparison: sklearn vs XGBoost, LightGBM, CatBoost")
    print("=" * 60)

    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    print("\n[1] sklearn GradientBoostingClassifier:")
    gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    acc_gb = accuracy_score(y_test, gb.predict(X_test))
    print(f"    Accuracy: {acc_gb:.4f}")
    print(f"    API: fit(X, y), predict(X)")

    print("\n[2] sklearn RandomForestClassifier:")
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    print(f"    Accuracy: {acc_rf:.4f}")

    if XGB_AVAILABLE:
        print("\n[3] XGBoost (sklearn API):")
        xgb_clf = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42)
        xgb_clf.fit(X_train, y_train)
        acc_xgb = accuracy_score(y_test, xgb_clf.predict(X_test))
        print(f"    Accuracy: {acc_xgb:.4f}")
        print(f"    Compatible with sklearn pipelines")
    else:
        print("\n[3] XGBoost: pip install xgboost")

    if LGB_AVAILABLE:
        print("\n[4] LightGBM (sklearn API):")
        lgb_clf = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42)
        lgb_clf.fit(X_train, y_train)
        lgb_clf.fit(X_train, y_train)
        acc_lgb = accuracy_score(y_test, lgb_clf.predict(X_test))
        print(f"    Accuracy: {acc_lgb:.4f}")
    else:
        print("\n[4] LightGBM: pip install lightgbm")

    print("\n[5] Framework comparison summary:")
    print("    | Framework  | Categorical | Sparse | Speed  | sklearn API |")
    print("    |------------|-------------|--------|-------|-------------|")
    print("    | sklearn GB | No          | No     | Slow  | Native      |")
    print("    | XGBoost    | Yes         | Yes    | Fast  | Yes         |")
    print("    | LightGBM   | Yes         | Yes    | Fast  | Yes         |")
    print("    | CatBoost   | Native      | Yes    | Fast  | Yes         |")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
