"""
Scikit-learn NumPy interop: array handling, memory layout, C vs F order
"""

import numpy as np
from sklearn.datasets import load_iris, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    print("=" * 60)
    print("NumPy Interop: array handling, memory layout, C vs F order")
    print("=" * 60)

    print("\n[1] NumPy array input/output:")
    X, y = load_iris(return_X_y=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"    Input dtype: {X.dtype}, Output dtype: {X_scaled.dtype}")
    print(f"    Input shape: {X.shape}, Output shape: {X_scaled.shape}")

    print("\n[2] C-contiguous vs F-contiguous:")
    X_c = np.ascontiguousarray(X)
    X_f = np.asfortranarray(X)
    print(f"    C-order flags: {X_c.flags['C_CONTIGUOUS']}")
    print(f"    F-order flags: {X_f.flags['F_CONTIGUOUS']}")
    scaler.fit(X_c)
    pred_c = scaler.transform(X_c)
    pred_f = scaler.transform(X_f)
    print(f"    Same result: {np.allclose(pred_c, pred_f)}")

    print("\n[3] Memory layout and copy behavior:")
    X_sub = X[:10].copy()
    X_sub.setflags(write=False)
    X_out = scaler.transform(X_sub)
    print(f"    Output is copy: {X_out.flags['OWNDATA']}")
    print(f"    Output C-contiguous: {X_out.flags['C_CONTIGUOUS']}")

    print("\n[4] Mixed dtypes and float64 conversion:")
    X_float32 = X.astype(np.float32)
    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X_float32, y)
    print(f"    Coef dtype: {clf.coef_.dtype}")
    X_pred = X_float32[:5]
    pred = clf.predict(X_pred)
    print(f"    Predict works with float32: {len(pred)}")

    print("\n[5] 1D vs 2D input handling:")
    x_1d = X[0]
    x_2d = X[:1]
    pred_1d = clf.predict(x_1d.reshape(1, -1))
    pred_2d = clf.predict(x_2d)
    print(f"    1D reshaped pred: {pred_1d[0]}")
    print(f"    2D pred: {pred_2d[0]}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
