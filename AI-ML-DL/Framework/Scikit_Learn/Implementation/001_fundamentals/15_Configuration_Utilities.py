"""
Scikit-learn Configuration Utilities
sklearn.set_config, config_context, show_versions
"""

import sklearn
from sklearn import set_config, config_context
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd


def main():
    print("=" * 60)
    print("Configuration Utilities")
    print("=" * 60)

    print("\n[1] set_config - global configuration:")
    set_config(assume_finite=True)
    print("    assume_finite=True skips nan/inf checks (faster)")

    print("\n[2] set_config - transform_output:")
    set_config(transform_output="pandas")
    scaler = StandardScaler()
    X, _ = load_iris(return_X_y=True)
    X_df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    out = scaler.fit_transform(X_df)
    print(f"    transform_output='pandas': output type = {type(out).__name__}")
    set_config(transform_output="default")

    print("\n[3] config_context - temporary settings:")
    with config_context(assume_finite=True):
        print("    Inside context: assume_finite=True")
    print("    Outside: previous config restored")

    print("\n[4] config_context - transform_output:")
    with config_context(transform_output="pandas"):
        scaler2 = StandardScaler()
        out2 = scaler2.fit_transform(X_df)
        print(f"    Within context: {type(out2).__name__}")
    out3 = StandardScaler().fit_transform(X_df)
    print(f"    After context: {type(out3).__name__}")

    print("\n[5] show_versions - full environment:")
    print("    sklearn.show_versions() prints:")
    sklearn.show_versions()

    print("\n[6] display - HTML representation (Jupyter):")
    print("    set_config(display='diagram') for Pipeline viz")
    set_config(display="text")
    print("    display='text' | 'diagram'")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
