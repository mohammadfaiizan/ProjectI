"""
tf.random: uniform, normal, truncated_normal, shuffle, set_seed, Generator
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Random Generation")
    print("=" * 50)
    
    print("\n--- tf.random.set_seed ---")
    tf.random.set_seed(42)
    
    print("\n--- tf.random.uniform ---")
    u = tf.random.uniform(shape=[2, 3], minval=0, maxval=1)
    print(f"uniform([2,3], 0, 1):\n{u}")
    
    u_int = tf.random.uniform(shape=[5], minval=1, maxval=10, dtype=tf.int32)
    print(f"uniform int: {u_int.numpy()}")
    
    print("\n--- tf.random.normal ---")
    n = tf.random.normal(shape=[2, 3], mean=0.0, stddev=1.0)
    print(f"normal([2,3], 0, 1):\n{n}")
    
    print("\n--- tf.random.truncated_normal ---")
    tn = tf.random.truncated_normal(shape=[5], mean=0.0, stddev=1.0)
    print(f"truncated_normal: {tn.numpy()}")
    
    print("\n--- tf.random.shuffle ---")
    arr = tf.constant([1, 2, 3, 4, 5])
    shuffled = tf.random.shuffle(arr)
    print(f"shuffle: {shuffled.numpy()}")
    
    mat = tf.constant([[1, 2], [3, 4], [5, 6]])
    shuffled_rows = tf.random.shuffle(mat)
    print(f"shuffle rows:\n{shuffled_rows}")
    
    print("\n--- tf.random.Generator ---")
    gen = tf.random.Generator.from_seed(123)
    r1 = gen.uniform(shape=[3])
    r2 = gen.normal(shape=[3])
    print(f"Generator uniform: {r1.numpy()}")
    print(f"Generator normal: {r2.numpy()}")
    
    gen2 = tf.random.Generator.from_non_deterministic_state()
    nd = gen2.uniform(shape=[2])
    print(f"Non-deterministic: {nd.numpy()}")
    
    print("\n--- Reproducibility with seed ---")
    tf.random.set_seed(99)
    a1 = tf.random.normal([2])
    tf.random.set_seed(99)
    a2 = tf.random.normal([2])
    print(f"Same seed -> same values: {tf.reduce_all(tf.equal(a1, a2)).numpy()}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
