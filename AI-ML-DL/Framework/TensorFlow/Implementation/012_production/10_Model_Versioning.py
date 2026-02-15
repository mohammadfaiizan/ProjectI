"""
Model versioning strategies, metadata.
"""
import tensorflow as tf
import os
import json

def main():
    print("=" * 50)
    print("Model Versioning")
    print("=" * 50)

    base_dir = os.path.join(os.path.dirname(__file__), "versioned_models")
    if os.path.exists(base_dir):
        import shutil
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    for v in [1, 2, 3]:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        model_path = os.path.join(base_dir, str(v))
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        tf.saved_model.save(model, model_path)
        metadata = {
            "version": v,
            "description": f"Model version {v}",
            "created": "2024-01-01",
            "metrics": {"accuracy": 0.9 + v * 0.01}
        }
        with open(os.path.join(model_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Version {v} saved with metadata")

    versions = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], key=int)
    print(f"Available versions: {versions}")

    latest = max(versions, key=int)
    meta_path = os.path.join(base_dir, latest, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Latest metadata: {meta}")

    print("\nVersioning strategies:")
    print("  - Timestamped folders: model/20240101_120000/")
    print("  - Semantic versioning: model/1.2.3/")
    print("  - Numeric: model/1/, model/2/")

    print("Model versioning demo complete.")

if __name__ == "__main__":
    main()
