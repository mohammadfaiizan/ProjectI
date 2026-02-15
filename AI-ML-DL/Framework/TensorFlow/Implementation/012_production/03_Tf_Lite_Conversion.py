"""
tf.lite.TFLiteConverter from_saved_model, from_keras_model.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("TF Lite Conversion")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    x = tf.random.normal((50, 16))
    y = tf.random.uniform((50,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)
    print("Keras model built and trained")

    converter_keras = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_keras = converter_keras.convert()
    print(f"From Keras: tflite size {len(tflite_keras)} bytes")

    save_dir = os.path.join(os.path.dirname(__file__), "saved_for_tflite")
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    model.save(save_dir)
    converter_saved = tf.lite.TFLiteConverter.from_saved_model(save_dir)
    tflite_saved = converter_saved.convert()
    print(f"From SavedModel: tflite size {len(tflite_saved)} bytes")

    tflite_path = os.path.join(os.path.dirname(__file__), "model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_keras)
    print(f"TFLite model written to {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f"Input shape: {input_details[0]['shape']}, Output shape: {output_details[0]['shape']}")

    test_input = tf.random.normal((1, 16))
    interpreter.set_tensor(input_details[0]['index'], test_input.numpy())
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    print(f"Inference output shape: {output.shape}")

    print("TF Lite conversion demo complete.")

if __name__ == "__main__":
    main()
