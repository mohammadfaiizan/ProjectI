"""
Dynamic range, full integer, float16 quantization.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("TF Lite Quantization")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    x = tf.random.normal((100, 16))
    y = tf.random.uniform((100,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)

    save_dir = os.path.join(os.path.dirname(__file__), "saved_quant")
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)
    model.save(save_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(save_dir)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()
    print(f"Dynamic range quantization: {len(tflite_dynamic)} bytes")

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_float16 = converter.convert()
    print(f"Float16 quantization: {len(tflite_float16)} bytes")

    def representative_dataset():
        for _ in range(10):
            yield [tf.random.normal((1, 16)).numpy().astype(tf.float32)]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    try:
        tflite_full_int = converter.convert()
        print(f"Full integer quantization: {len(tflite_full_int)} bytes")
    except Exception as e:
        print(f"Full integer (may need more ops): {e}")

    out_dir = os.path.dirname(__file__)
    with open(os.path.join(out_dir, "model_dynamic.tflite"), 'wb') as f:
        f.write(tflite_dynamic)
    with open(os.path.join(out_dir, "model_float16.tflite"), 'wb') as f:
        f.write(tflite_float16)
    print("Quantized models saved")

    print("TF Lite quantization demo complete.")

if __name__ == "__main__":
    main()
