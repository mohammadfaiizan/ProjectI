"""
Scikit-learn Pandas integration: set_output("pandas"), DataFrame in/out, feature names
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("Pandas Integration: set_output, DataFrame, feature names")
    print("=" * 60)

    print("\n[1] set_output('pandas') on StandardScaler:")
    X, y = load_iris(return_X_y=True)
    scaler = StandardScaler().set_output(transform="pandas")
    X_scaled = scaler.fit_transform(pd.DataFrame(X, columns=["a", "b", "c", "d"]))
    print(f"    Type: {type(X_scaled).__name__}")
    print(f"    Columns: {list(X_scaled.columns)}")
    print(f"    Shape: {X_scaled.shape}")

    print("\n[2] Pipeline with pandas output:")
    pipe = Pipeline([
        ("scaler", StandardScaler().set_output(transform="pandas")),
        ("clf", LogisticRegression(max_iter=200, random_state=42))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    df_train = pd.DataFrame(X_train, columns=["a", "b", "c", "d"])
    pipe.fit(df_train, y_train)
    pred = pipe.predict(pd.DataFrame(X_test, columns=["a", "b", "c", "d"]))
    print(f"    Predictions type: {type(pred).__name__}")
    print(f"    Accuracy: {(pred == y_test).mean():.4f}")

    print("\n[3] ColumnTransformer with feature names:")
    df = pd.DataFrame({
        "num1": np.random.randn(100),
        "num2": np.random.randn(100),
        "cat": np.random.choice(["A", "B", "C"], 100)
    })
    ct = ColumnTransformer([
        ("num", StandardScaler(), ["num1", "num2"]),
        ("cat", OneHotEncoder(drop="first"), ["cat"])
    ], verbose_feature_names_out=False).set_output(transform="pandas")
    X_trans = ct.fit_transform(df)
    print(f"    Output columns: {list(X_trans.columns)}")
    print(f"    Feature names: {ct.get_feature_names_out()}")

    print("\n[4] Global config set_config:")
    from sklearn import set_config
    set_config(transform_output="pandas")
    scaler2 = StandardScaler()
    X_s = scaler2.fit_transform(X[:5])
    print(f"    With global pandas: {type(X_s).__name__}")
    set_config(transform_output="default")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
