"""
tf.Variable vs tf.constant, assign, assign_add, read_value
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Variables and Constants")
    print("=" * 50)
    
    print("\n--- tf.constant (immutable) ---")
    c = tf.constant([1, 2, 3])
    print(f"constant: {c.numpy()}")
    
    print("\n--- tf.Variable (mutable) ---")
    v = tf.Variable([1.0, 2.0, 3.0])
    print(f"Variable initial: {v.numpy()}")
    
    print("\n--- assign ---")
    v.assign([10.0, 20.0, 30.0])
    print(f"After assign: {v.numpy()}")
    
    print("\n--- assign_add ---")
    v.assign_add([1.0, 1.0, 1.0])
    print(f"After assign_add([1,1,1]): {v.numpy()}")
    
    print("\n--- read_value ---")
    snapshot = v.read_value()
    print(f"read_value(): {snapshot.numpy()}")
    
    print("\n--- Variable in computation ---")
    w = tf.Variable([[1.0], [2.0]])
    x = tf.constant([[3.0, 4.0]])
    y = tf.matmul(x, w)
    print(f"matmul(x, w): {y.numpy()}")
    
    print("\n--- Variable shape and dtype ---")
    print(f"Shape: {v.shape}, dtype: {v.dtype}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
