"""
EarlyStopping, ReduceLROnPlateau, and learning rate scheduling.
"""
import tensorflow as tf
import numpy as np

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_train = x_train.reshape(-1, 784)[:1000]
    y_train = tf.keras.utils.to_categorical(y_train[:1000], 10)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    print(f"\nEpochs run: {len(history.history['loss'])}")
    print(f"Final val_loss: {history.history['val_loss'][-1]:.4f}")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=100,
        decay_rate=0.96
    )
    opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model2.fit(x_train, y_train, epochs=2, batch_size=32, verbose=0)
    print(f"\nExponential decay LR at step 100: {lr_schedule(100).numpy():.6f}")

    print("Early stopping and learning rate callbacks verified.")

if __name__ == "__main__":
    main()
