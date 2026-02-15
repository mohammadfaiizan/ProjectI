"""
SavedModel, HDF5, weights-only (save, save_weights).
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
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    tmpdir = tempfile.mkdtemp()
    savedmodel_path = os.path.join(tmpdir, 'saved_model')
    h5_path = os.path.join(tmpdir, 'model.h5')
    weights_path = os.path.join(tmpdir, 'weights.weights.h5')

    model.save(savedmodel_path, save_format='tf')
    print(f"SavedModel saved to {savedmodel_path}")

    model.save(h5_path, save_format='h5')
    print(f"HDF5 model saved to {h5_path}")

    model.save_weights(weights_path)
    print(f"Weights-only saved to {weights_path}")

    x = tf.random.normal((2, 784))
    out_orig = model(x)

    loaded_savedmodel = tf.keras.models.load_model(savedmodel_path)
    out_sm = loaded_savedmodel(x)
    print(f"SavedModel load match: {tf.reduce_all(tf.abs(out_orig - out_sm) < 1e-5)}")

    loaded_h5 = tf.keras.models.load_model(h5_path)
    out_h5 = loaded_h5(x)
    print(f"HDF5 load match: {tf.reduce_all(tf.abs(out_orig - out_h5) < 1e-5)}")

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model2.load_weights(weights_path)
    out_w = model2(x)
    print(f"Weights-only load match: {tf.reduce_all(tf.abs(out_orig - out_w) < 1e-5)}")
    print("Model saving formats verified.")

if __name__ == "__main__":
    main()
