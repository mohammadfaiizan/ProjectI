"""
tf.distribute with tf.data, sharding.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Distributed Data Loading")
    print("=" * 50)

    print("\n--- Sharding options ---")
    ds = tf.data.Dataset.range(12)
    ds_shard = ds.shard(num_shards=3, index=0)
    print(f"Shard 0 of 3: {list(ds_shard.as_numpy_iterator())}")

    ds_shard1 = ds.shard(num_shards=3, index=1)
    print(f"Shard 1 of 3: {list(ds_shard1.as_numpy_iterator())}")

    print("\n--- options() for performance ---")
    ds_opt = tf.data.Dataset.range(10)
    ds_opt = ds_opt.map(lambda x: x * 2, num_parallel_calls=tf.data.AUTOTUNE)
    ds_opt = ds_opt.prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_optimization.map_parallelization = True
    ds_opt = ds_opt.with_options(options)
    print(f"Optimized: {list(ds_opt.take(3).as_numpy_iterator())}")

    print("\n--- Determinism ---")
    ds_det = tf.data.Dataset.range(10).shuffle(10, seed=42)
    options_det = tf.data.Options()
    options_det.deterministic = True
    ds_det = ds_det.with_options(options_det)
    run1 = list(ds_det.as_numpy_iterator())
    ds_det2 = tf.data.Dataset.range(10).shuffle(10, seed=42).with_options(options_det)
    run2 = list(ds_det2.as_numpy_iterator())
    print(f"Deterministic runs match: {run1 == run2}")

    print("\n--- Global shuffle (concept) ---")
    ds_global = tf.data.Dataset.range(20)
    ds_global = ds_global.shuffle(20, seed=42, reshuffle_each_iteration=True)
    print(f"Shuffled sample: {list(ds_global.take(5).as_numpy_iterator())}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
