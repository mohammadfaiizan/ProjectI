"""
Adam, AdamW (via weight_decay), Adamax.
"""
import tensorflow as tf

def main():
    adam = tf.keras.optimizers.Adam(learning_rate=0.001)
    print(f"Adam: lr={adam.learning_rate.numpy()}, beta_1={adam.beta_1}, beta_2={adam.beta_2}")

    adam_custom = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    print(f"Adam (custom): epsilon={adam_custom.epsilon}")

    adamw = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)
    print(f"AdamW: weight_decay={adamw.weight_decay}")

    adamax = tf.keras.optimizers.Adamax(learning_rate=0.002)
    print(f"Adamax: lr={adamax.learning_rate.numpy()}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x = tf.random.normal((64, 10))
    y = tf.random.uniform((64,), maxval=10, dtype=tf.int32)
    history = model.fit(x, y, epochs=2, verbose=0)
    print(f"Adam training loss: {history.history['loss'][-1]:.4f}")

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model2.compile(optimizer=adamw, loss='sparse_categorical_crossentropy')
    model2.fit(tf.random.normal((32, 5)), tf.random.uniform((32,), maxval=5, dtype=tf.int32), epochs=2, verbose=0)
    print(f"AdamW training completed.")

    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model3.compile(optimizer=adamax, loss='sparse_categorical_crossentropy')
    model3.fit(tf.random.normal((16, 4)), tf.random.uniform((16,), maxval=3, dtype=tf.int32), epochs=2, verbose=0)
    print(f"Adamax training completed.")
    print("Adam family optimizers verified.")

if __name__ == "__main__":
    main()
