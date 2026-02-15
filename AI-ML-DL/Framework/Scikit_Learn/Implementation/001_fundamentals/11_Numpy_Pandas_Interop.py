"""
Scikit-learn NumPy and Pandas Interoperability
NumPy arrays in/out, set_output("pandas"), DataFrame support
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
def main():
    print("=" * 60)
    print("NumPy and Pandas Interoperability")
    print("=" * 60)

    iris = load_iris()
    X_np = iris.data
    X_df = pd.DataFrame(X_np, columns=iris.feature_names)

    print("\n[1] NumPy arrays - default input/output:")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    print(f"    Input: ndarray {X_np.shape}")
    print(f"    Output type: {type(X_scaled)}")
    print(f"    Output shape: {X_scaled.shape}")

    print("\n[2] set_output('pandas') - DataFrame output:")
    scaler_pd = StandardScaler().set_output(transform="pandas")
    X_scaled_pd = scaler_pd.fit_transform(X_df)
    print(f"    Output type: {type(X_scaled_pd)}")
    print(f"    Columns: {list(X_scaled_pd.columns)}")
    print(f"    First row:\n{X_scaled_pd.iloc[:1]}")

    print("\n[3] Pandas DataFrame as input:")
    clf_scaler = StandardScaler()
    clf_scaler.fit(X_df)
    out = clf_scaler.transform(X_df)
    print(f"    DataFrame input accepted")
    print(f"    Output preserves structure when set_output('pandas')")

    print("\n[4] PCA with pandas output:")
    pca = PCA(n_components=2).set_output(transform="pandas")
    X_pca = pca.fit_transform(X_df)
    print(f"    PCA output columns: {list(X_pca.columns)}")

    print("\n[5] config_context for temporary setting:")
    from sklearn import config_context
    with config_context(transform_output="pandas"):
        scaler_ctx = StandardScaler()
        out_ctx = scaler_ctx.fit_transform(X_df)
    print(f"    Within config_context: output is {type(out_ctx).__name__}")

    print("\n[6] Estimators that support set_output:")
    print("    StandardScaler, MinMaxScaler, PCA, SimpleImputer, etc.")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
