"""
REST and gRPC API for TF Serving.
"""
import tensorflow as tf
import json
import os

def main():
    print("=" * 50)
    print("TF Serving REST and gRPC API")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    save_path = os.path.join(os.path.dirname(__file__), "serving_model")
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    tf.saved_model.save(model, save_path)
    print("Model saved for serving")

    sample = tf.random.normal((2, 16)).numpy().tolist()
    rest_payload = {"instances": sample}
    print("REST request payload (instances):")
    print(json.dumps({"instances": [[0.0] * 16]}, indent=2)[:200] + "...")

    print("\nREST endpoint: POST http://localhost:8501/v1/models/my_model:predict")
    print("  Body: {\"instances\": [[...], [...]]}")

    print("\ngRPC endpoint: localhost:8500")
    print("  PredictionService.Predict")

    try:
        import requests
        resp = requests.post(
            "http://localhost:8501/v1/models/my_model:predict",
            json=rest_payload,
            timeout=2
        )
        print(f"\nREST response status: {resp.status_code}")
        if resp.ok:
            pred = resp.json()
            print(f"Predictions keys: {list(pred.keys())}")
    except Exception as e:
        print(f"\nREST request (TF Serving not running): {e}")

    print("TF Serving API demo complete.")

if __name__ == "__main__":
    main()
