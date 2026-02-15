"""
TF Serving with Docker, model directory structure.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("TF Serving Setup - Model Directory Structure")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    x = tf.random.normal((50, 16))
    y = tf.random.uniform((50,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)

    base_dir = os.path.join(os.path.dirname(__file__), "tf_serving_models")
    model_name = "my_model"
    version = 1
    model_path = os.path.join(base_dir, model_name, str(version))
    if os.path.exists(base_dir):
        import shutil
        shutil.rmtree(base_dir)
    os.makedirs(model_path, exist_ok=True)

    tf.saved_model.save(model, model_path)
    print(f"Model saved at: {model_path}")
    print(f"Structure: {model_name}/")
    print(f"           {version}/")
    print("             saved_model.pb")
    print("             variables/")

    pb_path = os.path.join(model_path, "saved_model.pb")
    print(f"saved_model.pb exists: {os.path.exists(pb_path)}")

    print("\nDocker run command:")
    print("  docker run -p 8501:8501 -p 8500:8500 \\")
    print(f"    -v \"{os.path.abspath(base_dir)}:/models/{model_name}\" \\")
    print("    -e MODEL_NAME=my_model \\")
    print("    tensorflow/serving")

    print("\nTF Serving setup demo complete.")

if __name__ == "__main__":
    main()
