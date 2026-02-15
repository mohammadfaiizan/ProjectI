"""
Performance optimization: XLA, tf.function, memory, I/O tips.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Performance Optimization")
    print("=" * 50)

    print("\nXLA (Accelerated Linear Algebra):")
    tf.config.optimizer.set_jit(True)
    print("  tf.config.optimizer.set_jit(True) enables XLA")

    @tf.function(jit_compile=True)
    def xla_fn(x):
        return tf.reduce_sum(tf.matmul(x, tf.transpose(x)))

    x = tf.random.normal((64, 64))
    result = xla_fn(x)
    print(f"  jit_compile=True on tf.function: {result.shape}")

    print("\ntf.function optimization:")
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 32], dtype=tf.float32)])
    def optimized_fn(x):
        return tf.keras.layers.Dense(10)(x)

    out = optimized_fn(tf.random.normal((8, 32)))
    print("  input_signature reduces retracing")
    print("  experimental_relaxed_shapes for flexible shapes")

    print("\nMemory tips:")
    print("  - Mixed precision: tf.keras.mixed_precision.set_global_policy('mixed_float16')")
    print("  - Gradient checkpointing for large models")
    print("  - Reduce batch size if OOM")

    print("\nI/O tips:")
    ds = tf.data.Dataset.range(1000)
    ds = ds.map(lambda x: x * 2, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.shuffle(100)
    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    print("  num_parallel_calls=AUTOTUNE, cache(), prefetch(AUTOTUNE)")

    print("\nData pipeline optimization:")
    print("  - Interleave for file I/O")
    print("  - options.experimental_optimization.map_vectorization")
    print("  - options.experimental_deterministic=False for speed")

    for batch in ds.take(1):
        print(f"  Sample batch shape: {batch.shape}")

    print("\nPerformance optimization demo complete.")

if __name__ == "__main__":
    main()
