"""
tf.keras.mixed_precision: set_global_policy, Policy, LossScaleOptimizer.
"""
import tensorflow as tf

def main():
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    print(f"Policy: {policy.name}, compute_dtype={policy.compute_dtype}, variable_dtype={policy.variable_dtype}")

    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    current = tf.keras.mixed_precision.global_policy()
    print(f"Global policy: {current.name}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, dtype='float32')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
    model.compile(optimizer=opt, loss='mse')
    x = tf.random.normal((32, 8), dtype=tf.float32)
    y = tf.random.normal((32, 1), dtype=tf.float32)
    model.fit(x, y, epochs=2, verbose=0)
    print(f"Mixed precision training completed.")

    tf.keras.mixed_precision.set_global_policy('float32')
    print(f"Reset to float32: {tf.keras.mixed_precision.global_policy().name}")

    tf.keras.mixed_precision.set_global_policy('float16')
    layer = tf.keras.layers.Dense(8, input_shape=(4,))
    layer.build((None, 4))
    print(f"Layer under float16 policy: kernel dtype={layer.kernel.dtype}")
    tf.keras.mixed_precision.set_global_policy('float32')

    opt_base = tf.keras.optimizers.Adam(0.001)
    opt_scaled = tf.keras.mixed_precision.LossScaleOptimizer(opt_base, initial_scale=2**15)
    print(f"LossScaleOptimizer: initial_scale={opt_scaled.initial_scale}")
    print("Mixed precision training verified.")

if __name__ == "__main__":
    main()
