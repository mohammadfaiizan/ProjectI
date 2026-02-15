"""
Tensor Dtypes, Casting, and Type Promotion
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Types and Casting")
    print("=" * 50)
    
    print("\n--- Common dtypes ---")
    a = tf.constant([1, 2, 3], dtype=tf.int32)
    b = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    c = tf.constant([True, False], dtype=tf.bool)
    print(f"int32: {a.dtype}, values: {a.numpy()}")
    print(f"float32: {b.dtype}, values: {b.numpy()}")
    print(f"bool: {c.dtype}, values: {c.numpy()}")
    
    print("\n--- tf.cast ---")
    x = tf.constant([1, 2, 3], dtype=tf.int32)
    y = tf.cast(x, tf.float32)
    print(f"Original (int32): {x.numpy()}")
    print(f"After cast to float32: {y.numpy()}")
    
    z = tf.constant([0.0, 1.5, 2.9], dtype=tf.float32)
    w = tf.cast(z, tf.int32)
    print(f"Float to int (truncate): {z.numpy()} -> {w.numpy()}")
    
    print("\n--- Type promotion ---")
    i = tf.constant(1, dtype=tf.int32)
    f = tf.constant(2.0, dtype=tf.float32)
    result = i + f
    print(f"int32 + float32 = {result.dtype}: {result.numpy()}")
    
    print("\n--- Bool casting ---")
    nums = tf.constant([0, 1, -1, 2])
    as_bool = tf.cast(nums, tf.bool)
    print(f"Numbers as bool: {nums.numpy()} -> {as_bool.numpy()}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
