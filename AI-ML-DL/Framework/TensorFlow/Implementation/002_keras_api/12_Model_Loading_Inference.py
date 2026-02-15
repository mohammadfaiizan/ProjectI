"""
tf.keras.models.load_model, load_weights, inference.
"""
import tensorflow as tf
import numpy as np
import os
import tempfile

def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    tmpdir = tempfile.mkdtemp()
    model_path = os.path.join(tmpdir, 'model.keras')
    model.save(model_path)

    loaded = tf.keras.models.load_model(model_path)
    print("Model loaded via load_model.")

    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test = x_test.astype(np.float32) / 255.0
    x_test = x_test.reshape(-1, 784)[:10]

    preds = loaded.predict(x_test, verbose=0)
    classes = np.argmax(preds, axis=1)
    print(f"Inference output shape: {preds.shape}")
    print(f"Predicted classes: {classes}")

    single_pred = loaded.predict(x_test[:1], verbose=0)
    print(f"Single sample prediction shape: {single_pred.shape}")

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    weights_path = os.path.join(tmpdir, 'weights.weights.h5')
    model.save_weights(weights_path)
    model2.load_weights(weights_path)
    out2 = model2.predict(x_test[:2], verbose=0)
    out1 = loaded.predict(x_test[:2], verbose=0)
    print(f"load_weights inference match: {np.allclose(out1, out2)}")
    print("Model loading and inference verified.")

if __name__ == "__main__":
    main()
