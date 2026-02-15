"""
tf.saved_model.save, tf.saved_model.load, signatures.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("SavedModel Format - Save and Load")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x = tf.random.normal((100, 32))
    y = tf.random.uniform((100,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)
    print("Model trained")

    save_path = os.path.join(os.path.dirname(__file__), "saved_model_01")
    if os.path.exists(save_path):
        import shutil
        shutil.rmtree(save_path)
    tf.saved_model.save(model, save_path)
    print(f"Model saved to {save_path}")

    loaded = tf.saved_model.load(save_path)
    print(f"Loaded object type: {type(loaded)}")

    infer = loaded.signatures.get('serving_default')
    if infer is None:
        infer = list(loaded.signatures.values())[0]
    print(f"Available signatures: {list(loaded.signatures.keys())}")

    sample_input = tf.random.normal((2, 32))
    result = infer(tf.constant(sample_input))
    output_key = list(result.keys())[0]
    print(f"Output shape: {result[output_key].shape}")

    keras_loaded = tf.keras.models.load_model(save_path)
    pred = keras_loaded.predict(sample_input, verbose=0)
    print(f"Keras reload prediction shape: {pred.shape}")

    print("SavedModel format demo complete.")

if __name__ == "__main__":
    main()
