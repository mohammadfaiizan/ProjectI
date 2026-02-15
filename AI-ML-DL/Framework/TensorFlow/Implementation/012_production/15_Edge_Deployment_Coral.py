"""
Edge deployment with TF Lite, Coral/Edge TPU.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("Edge Deployment - TF Lite and Coral")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    x = tf.random.normal((50, 16))
    y = tf.random.uniform((50,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    tflite_path = os.path.join(os.path.dirname(__file__), "edge_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved: {tflite_path}")

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("TFLite interpreter ready for edge inference")

    print("\nEdge deployment options:")
    print("  - TF Lite: mobile, embedded, Raspberry Pi")
    print("  - Coral Edge TPU: compile with edgetpu_compiler for .tflite")
    print("  - Coral devices: USB Accelerator, Dev Board")

    print("\nCoral conversion (requires edgetpu_compiler):")
    print("  edgetpu_compiler edge_model.tflite -o edge_model_edgetpu.tflite")

    print("\nEdge deployment demo complete.")

if __name__ == "__main__":
    main()
