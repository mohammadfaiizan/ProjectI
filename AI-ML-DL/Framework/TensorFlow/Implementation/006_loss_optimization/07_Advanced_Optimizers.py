"""
RMSprop, Adagrad, Adadelta, Ftrl, Nadam.
"""
import tensorflow as tf

def main():
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
    print(f"RMSprop: rho={rmsprop.rho}")

    adagrad = tf.keras.optimizers.Adagrad(learning_rate=0.01, initial_accumulator_value=0.1)
    print(f"Adagrad: initial_accumulator_value={adagrad.initial_accumulator_value}")

    adadelta = tf.keras.optimizers.Adadelta(learning_rate=1.0, rho=0.95)
    print(f"Adadelta: rho={adadelta.rho}")

    ftrl = tf.keras.optimizers.Ftrl(learning_rate=0.1, l1_regularization_strength=0.01, l2_regularization_strength=0.01)
    print(f"Ftrl: l1={ftrl.l1_regularization_strength}, l2={ftrl.l2_regularization_strength}")

    nadam = tf.keras.optimizers.Nadam(learning_rate=0.001)
    print(f"Nadam: lr={nadam.learning_rate.numpy()}")

    model_rms = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    model_rms.compile(optimizer=rmsprop, loss='mse')
    model_rms.fit(tf.random.normal((32, 8)), tf.random.normal((32, 1)), epochs=2, verbose=0)
    print(f"RMSprop training completed.")

    model_ada = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model_ada.compile(optimizer=adagrad, loss='mse')
    model_ada.fit(tf.random.normal((16, 4)), tf.random.normal((16, 1)), epochs=2, verbose=0)
    print(f"Adagrad training completed.")

    model_nadam = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model_nadam.compile(optimizer=nadam, loss='sparse_categorical_crossentropy')
    model_nadam.fit(tf.random.normal((24, 6)), tf.random.uniform((24,), maxval=3, dtype=tf.int32), epochs=2, verbose=0)
    print(f"Nadam training completed.")

    model_ftrl = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model_ftrl.compile(optimizer=ftrl, loss='binary_crossentropy')
    model_ftrl.fit(tf.random.normal((32, 10)), tf.random.uniform((32, 1), maxval=2, dtype=tf.float32), epochs=2, verbose=0)
    print(f"Ftrl training completed.")
    print("Advanced optimizers verified.")

if __name__ == "__main__":
    main()
