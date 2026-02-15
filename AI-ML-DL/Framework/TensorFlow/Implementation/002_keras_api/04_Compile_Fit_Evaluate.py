"""
model.compile(optimizer, loss, metrics), model.fit(), model.evaluate(), model.predict()
"""
import tensorflow as tf
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)
    x_test = x_test.reshape(-1, 784)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train, y_train = x_train[:1000], y_train[:1000]
    x_test, y_test = x_test[:200], y_test[:200]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled with optimizer, loss, metrics.")

    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )
    print(f"\nTraining completed. Final loss: {history.history['loss'][-1]:.4f}")

    results = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nEvaluate: loss={results[0]:.4f}, accuracy={results[1]:.4f}")

    preds = model.predict(x_test[:5])
    print(f"\nPredict output shape: {preds.shape}")
    print(f"Sample predictions (argmax): {np.argmax(preds, axis=1)}")
    print("Compile, fit, evaluate, predict verified.")

if __name__ == "__main__":
    main()
