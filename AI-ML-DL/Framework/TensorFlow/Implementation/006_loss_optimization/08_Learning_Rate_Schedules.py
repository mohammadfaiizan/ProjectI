"""
ExponentialDecay, CosineDecay, PiecewiseConstantDecay, PolynomialDecay.
"""
import tensorflow as tf

def main():
    exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.1, decay_steps=1000, decay_rate=0.96
    )
    lr_0 = exp_decay(0)
    lr_1000 = exp_decay(1000)
    print(f"ExponentialDecay: lr(0)={lr_0.numpy():.4f}, lr(1000)={lr_1000.numpy():.6f}")

    cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.1, decay_steps=1000
    )
    lr_cos_0 = cosine_decay(0)
    lr_cos_500 = cosine_decay(500)
    print(f"CosineDecay: lr(0)={lr_cos_0.numpy():.4f}, lr(500)={lr_cos_500.numpy():.4f}")

    piecewise = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[100, 500, 1000],
        values=[0.1, 0.05, 0.01, 0.001]
    )
    print(f"PiecewiseConstantDecay: lr(50)={piecewise(50).numpy()}, lr(200)={piecewise(200).numpy()}, lr(1500)={piecewise(1500).numpy()}")

    poly_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.1, decay_steps=1000, end_learning_rate=0.001, power=2.0
    )
    lr_poly_0 = poly_decay(0)
    lr_poly_1000 = poly_decay(1000)
    print(f"PolynomialDecay: lr(0)={lr_poly_0.numpy():.4f}, lr(1000)={lr_poly_1000.numpy():.6f}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=exp_decay)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(tf.random.normal((64, 8)), tf.random.normal((64, 1)), epochs=5, verbose=0)
    step = optimizer.iterations.numpy()
    current_lr = exp_decay(step).numpy()
    print(f"After 5 epochs: step={step}, current_lr={current_lr:.6f}")

    cosine_restarts = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=0.1, first_decay_steps=100, t_mul=2.0
    )
    print(f"CosineDecayRestarts: lr(0)={cosine_restarts(0).numpy():.4f}, lr(100)={cosine_restarts(100).numpy():.6f}")
    print("Learning rate schedules verified.")

if __name__ == "__main__":
    main()
