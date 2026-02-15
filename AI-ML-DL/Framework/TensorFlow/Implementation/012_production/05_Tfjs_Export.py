"""
TensorFlow.js conversion concepts and export.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("TensorFlow.js Export Concepts")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    x = tf.random.normal((50, 16))
    y = tf.random.uniform((50,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)
    print("Keras model built")

    save_dir = os.path.join(os.path.dirname(__file__), "tfjs_model")
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model.save(os.path.join(save_dir, "keras"))
    print(f"Model saved for TF.js at {save_dir}/keras")

    try:
        import tensorflowjs as tfjs
        tfjs_out = os.path.join(save_dir, "tfjs")
        tfjs.converters.convert_tf_saved_model(
            os.path.join(save_dir, "keras"),
            tfjs_out
        )
        print(f"TF.js conversion successful: {tfjs_out}")
    except ImportError:
        print("tensorflowjs not installed. Install with: pip install tensorflowjs")
        print("Alternative: use tensorflowjs_converter CLI:")
        print("  tensorflowjs_converter --input_format=keras path/to/model path/to/output")

    print("TF.js export concepts demo complete.")

if __name__ == "__main__":
    main()
