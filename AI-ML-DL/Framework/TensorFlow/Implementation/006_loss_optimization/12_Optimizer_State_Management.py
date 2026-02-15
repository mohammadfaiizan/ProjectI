"""
get_weights, set_weights, saving/loading optimizer state.
"""
import tensorflow as tf
import tempfile
import os

def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    x = tf.random.normal((32, 8))
    y = tf.random.normal((32, 1))
    model.fit(x, y, epochs=3, verbose=0)

    opt_weights = optimizer.get_weights()
    print(f"Optimizer weights count: {len(opt_weights)}")

    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    opt2 = tf.keras.optimizers.Adam(learning_rate=0.001)
    model2.compile(optimizer=opt2, loss='mse')
    opt2.set_weights(opt_weights)
    print(f"Optimizer state restored via set_weights.")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ckpt")
        ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt.save(ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

        model3 = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
            tf.keras.layers.Dense(1)
        ])
        opt3 = tf.keras.optimizers.Adam(learning_rate=0.001)
        model3.compile(optimizer=opt3, loss='mse')
        ckpt2 = tf.train.Checkpoint(optimizer=opt3, model=model3)
        status = ckpt2.restore(tf.train.latest_checkpoint(tmpdir))
        status.assert_consumed()
        print(f"Checkpoint restored.")

    var = tf.Variable(1.0)
    opt_sgd = tf.keras.optimizers.SGD(0.01, momentum=0.9)
    with tf.GradientTape() as tape:
        loss = (var - 0.5) ** 2
    grad = tape.gradient(loss, var)
    opt_sgd.apply_gradients([(grad, var)])
    sgd_weights = opt_sgd.get_weights()
    print(f"SGD with momentum - state vars: {len(sgd_weights)}")

    opt_new = tf.keras.optimizers.SGD(0.01, momentum=0.9)
    opt_new.build([var])
    opt_new.set_weights(sgd_weights)
    print(f"SGD state transferred to new optimizer.")
    print("Optimizer state management verified.")

if __name__ == "__main__":
    main()
