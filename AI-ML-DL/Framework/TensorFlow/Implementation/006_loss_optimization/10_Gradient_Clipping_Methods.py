"""
clipnorm, clipvalue, global_clipnorm in optimizers.
"""
import tensorflow as tf

def main():
    sgd_clipnorm = tf.keras.optimizers.SGD(learning_rate=0.01, clipnorm=1.0)
    print(f"SGD clipnorm=1.0: clipnorm={sgd_clipnorm.clipnorm}")

    adam_clipvalue = tf.keras.optimizers.Adam(learning_rate=0.001, clipvalue=0.5)
    print(f"Adam clipvalue=0.5: clipvalue={adam_clipvalue.clipvalue}")

    adam_global = tf.keras.optimizers.Adam(learning_rate=0.001, global_clipnorm=1.0)
    print(f"Adam global_clipnorm=1.0: global_clipnorm={adam_global.global_clipnorm}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=sgd_clipnorm, loss='mse')
    x = tf.random.normal((32, 10))
    y = tf.random.normal((32, 1))
    model.fit(x, y, epochs=2, verbose=0)
    print(f"Training with clipnorm completed.")

    var = tf.Variable([10.0, -10.0])
    with tf.GradientTape() as tape:
        loss = tf.reduce_sum(tf.square(var))
    grads = tape.gradient(loss, var)
    print(f"Gradients before clip: {grads.numpy()}")

    clipped_grads, _ = tf.clip_by_global_norm([grads], 1.0)
    print(f"Gradients after global_norm clip: {clipped_grads[0].numpy()}")

    grads_clipvalue = tf.clip_by_value(grads, -0.5, 0.5)
    print(f"Gradients after clip_value: {grads_clipvalue.numpy()}")

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    model2.compile(optimizer=adam_global, loss='mse')
    model2.fit(tf.random.normal((16, 8)), tf.random.normal((16, 1)), epochs=2, verbose=0)
    print(f"Training with global_clipnorm completed.")

    model3 = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model3.compile(optimizer=adam_clipvalue, loss='mse')
    model3.fit(tf.random.normal((16, 4)), tf.random.normal((16, 1)), epochs=2, verbose=0)
    print(f"Training with clipvalue completed.")
    print("Gradient clipping methods verified.")

if __name__ == "__main__":
    main()
