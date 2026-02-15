"""
Memory-efficient preprocessing and tf.data integration.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Efficient Preprocessing")
    print("=" * 50)

    print("\n--- tf.data.Dataset pipeline ---")
    def gen():
        for i in range(20):
            yield (tf.random.normal((32,)), tf.constant(i % 3))
    ds = tf.data.Dataset.from_generator(gen, output_signature=(
        tf.TensorSpec(shape=(32,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ))
    ds = ds.batch(4).prefetch(tf.data.AUTOTUNE)
    for x, y in ds.take(2):
        print(f"Batch x: {x.shape}, y: {y.shape}")

    print("\n--- map with preprocessing ---")
    def preprocess(x, y):
        x_norm = (x - tf.reduce_mean(x)) / (tf.math.reduce_std(x) + 1e-8)
        return x_norm, y
    ds2 = tf.data.Dataset.from_tensor_slices((
        tf.random.normal((100, 10)),
        tf.random.uniform((100,), 0, 3, dtype=tf.int32)
    ))
    ds2 = ds2.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds2 = ds2.batch(16).prefetch(2)
    xb, yb = next(iter(ds2))
    print(f"Mapped batch mean: {tf.reduce_mean(xb).numpy():.4f}")

    print("\n--- cache ---")
    ds3 = tf.data.Dataset.range(10).map(lambda x: x * 2).cache()
    list(ds3)
    list(ds3)
    print("Cache allows repeated iteration without recompute")

    print("\n--- Memory-efficient from_generator ---")
    def large_gen():
        for i in range(5):
            yield tf.random.normal((1000, 100))
    large_ds = tf.data.Dataset.from_generator(
        large_gen,
        output_signature=tf.TensorSpec(shape=(1000, 100), dtype=tf.float32)
    )
    count = sum(1 for _ in large_ds)
    print(f"Generated {count} tensors without loading all in memory")

    print("\n--- Interleave for parallel reads ---")
    files = ["a", "b", "c"]
    def read_fn(f):
        return tf.data.Dataset.from_tensors(tf.constant(1.0))
    interleaved = tf.data.Dataset.from_tensor_slices(files)
    interleaved = interleaved.interleave(
        lambda x: read_fn(x),
        cycle_length=2,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    print(f"Interleaved dataset: {list(interleaved)}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
