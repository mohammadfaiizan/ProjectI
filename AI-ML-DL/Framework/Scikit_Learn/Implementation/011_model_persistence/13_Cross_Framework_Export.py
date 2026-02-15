"""
Scikit-learn cross-framework export: ONNX, PMML concepts
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False


def main():
    print("=" * 60)
    print("Cross-Framework Export: ONNX, PMML")
    print("=" * 60)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)

    print("\n[1] ONNX export (framework-agnostic):")
    if SKL2ONNX_AVAILABLE:
        initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        with open("model.onnx", "wb") as f:
            f.write(onnx_model.SerializeToString())
        print("    Saved model.onnx (runnable in ONNX Runtime, TensorFlow, etc.)")
    else:
        print("    skl2onnx not installed")

    print("\n[2] ONNX consumers:")
    print("    - ONNX Runtime (Python, C#, Java)")
    print("    - TensorFlow (via onnx-tf)")
    print("    - PyTorch (via onnx)")
    print("    - Edge devices (ONNX Mobile)")

    print("\n[3] PMML (Predictive Model Markup Language):")
    print("    - XML-based, legacy support")
    print("    - sklearn2pmml, pypmml packages")
    print("    - Used in Java/SAS ecosystems")

    print("\n[4] Export format comparison:")
    print("    | Format | Use Case          | Ecosystem |")
    print("    | joblib | sklearn-only      | Python    |")
    print("    | ONNX   | Cross-framework   | Broad     |")
    print("    | PMML   | Enterprise/Java  | Legacy    |")

    print("\n[5] When to use ONNX:")
    print("    - Deploy to non-Python runtime")
    print("    - Need inference optimization")
    print("    - Mobile or edge deployment")

    import os
    if os.path.exists("model.onnx"):
        os.remove("model.onnx")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
