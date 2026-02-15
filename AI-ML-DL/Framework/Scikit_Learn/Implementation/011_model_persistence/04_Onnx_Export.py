"""
Scikit-learn ONNX export: skl2onnx concepts, convert_sklearn, to_onnx
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False


def main():
    print("=" * 60)
    print("ONNX Export: skl2onnx, convert_sklearn, to_onnx")
    print("=" * 60)

    if not SKL2ONNX_AVAILABLE:
        print("\n[!] skl2onnx not installed. Run: pip install skl2onnx onnx")
        return

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    print(f"\n[1] Model trained, test accuracy: {(model.predict(X_test) == y_test).mean():.4f}")

    print("\n[2] Define initial types (input shape):")
    n_features = X_train.shape[1]
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    print(f"    Input: FloatTensorType([None, {n_features}])")

    print("\n[3] convert_sklearn to ONNX:")
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    with open("model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("    Saved model.onnx")

    print("\n[4] Load and run with onnxruntime:")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession("model.onnx")
        inp = X_test[:5].astype(np.float32)
        pred_onnx = sess.run(None, {"float_input": inp})[0]
        pred_skl = model.predict(X_test[:5])
        print(f"    ONNX predictions: {pred_onnx}")
        print(f"    sklearn predictions: {pred_skl}")
        print(f"    Match: {(pred_onnx == pred_skl).all()}")
    except ImportError:
        print("    onnxruntime not installed, skipping inference")

    import os
    if os.path.exists("model.onnx"):
        os.remove("model.onnx")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
