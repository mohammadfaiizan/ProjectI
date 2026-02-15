"""
tf.keras.optimizers.SGD: learning_rate, momentum, nesterov.
"""
import tensorflow as tf

def main():
    sgd_vanilla = tf.keras.optimizers.SGD(learning_rate=0.01)
    print(f"SGD (vanilla): lr={sgd_vanilla.learning_rate.numpy()}")

    sgd_momentum = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    print(f"SGD (momentum=0.9): momentum={sgd_momentum.momentum}")

    sgd_nesterov = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    print(f"SGD (nesterov=True): nesterov={sgd_nesterov.nesterov}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=sgd_momentum, loss='mse')
    x = tf.random.normal((64, 10))
    y = tf.random.normal((64, 1))
    history = model.fit(x, y, epochs=3, verbose=0)
    print(f"SGD momentum training loss: {history.history['loss'][-1]:.4f}")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.1, 100, 0.96)
    sgd_scheduled = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    model2 = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(1)
    ])
    model2.compile(optimizer=sgd_scheduled, loss='mse')
    model2.fit(tf.random.normal((32, 5)), tf.random.normal((32, 1)), epochs=2, verbose=0)
    print(f"SGD with LR schedule - current lr: {sgd_scheduled.learning_rate(model2.optimizer.iterations).numpy():.6f}")

    sgd_lr_001 = tf.keras.optimizers.SGD(learning_rate=0.001)
    var = tf.Variable(1.0)
    with tf.GradientTape() as tape:
        loss = (var - 0.5) ** 2
    grad = tape.gradient(loss, var)
    sgd_lr_001.apply_gradients([(grad, var)])
    print(f"SGD manual step: var={var.numpy():.4f}")
    print("SGD optimizer verified.")

if __name__ == "__main__":
    main()
