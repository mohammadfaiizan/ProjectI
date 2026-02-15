"""
tf.reduce_sum, reduce_mean, reduce_max, reduce_min, reduce_prod, tf.math.reduce_std
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Reduction Operations")
    print("=" * 50)
    
    t = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"Tensor:\n{t}")
    
    print("\n--- tf.reduce_sum ---")
    print(f"reduce_sum (all): {tf.reduce_sum(t).numpy()}")
    print(f"reduce_sum(axis=0): {tf.reduce_sum(t, axis=0).numpy()}")
    print(f"reduce_sum(axis=1): {tf.reduce_sum(t, axis=1).numpy()}")
    
    print("\n--- tf.reduce_mean ---")
    print(f"reduce_mean (all): {tf.reduce_mean(t).numpy()}")
    print(f"reduce_mean(axis=1): {tf.reduce_mean(t, axis=1).numpy()}")
    
    print("\n--- tf.reduce_max ---")
    print(f"reduce_max: {tf.reduce_max(t).numpy()}")
    print(f"reduce_max(axis=0): {tf.reduce_max(t, axis=0).numpy()}")
    
    print("\n--- tf.reduce_min ---")
    print(f"reduce_min: {tf.reduce_min(t).numpy()}")
    
    print("\n--- tf.reduce_prod ---")
    small = tf.constant([1.0, 2.0, 3.0])
    print(f"reduce_prod: {tf.reduce_prod(small).numpy()}")
    
    print("\n--- tf.math.reduce_std ---")
    data = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])
    std = tf.math.reduce_std(data)
    print(f"reduce_std: {std.numpy()}")
    
    print("\n--- keepdims ---")
    s = tf.reduce_sum(t, axis=1, keepdims=True)
    print(f"reduce_sum(axis=1, keepdims=True) shape: {s.shape}")
    print(f"Result:\n{s}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
