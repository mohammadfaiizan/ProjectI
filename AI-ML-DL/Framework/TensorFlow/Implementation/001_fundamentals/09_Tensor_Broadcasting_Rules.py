"""
Broadcasting mechanics, compatible shapes, edge cases
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Broadcasting Rules")
    print("=" * 50)
    
    print("\n--- Compatible shapes ---")
    a = tf.constant([[1], [2], [3]])
    b = tf.constant([10, 20, 30])
    print(f"a shape: {a.shape}, b shape: {b.shape}")
    c = a + b
    print(f"a + b (broadcast):\n{c}")
    
    print("\n--- Scalar broadcast ---")
    x = tf.constant([[1, 2], [3, 4]])
    y = x + 10
    print(f"matrix + 10:\n{y}")
    
    print("\n--- Row/column broadcast ---")
    row = tf.constant([[1, 2, 3]])
    col = tf.constant([[1], [2], [3]])
    mat = row + col
    print(f"row + col:\n{mat}")
    
    print("\n--- Edge case: leading ones ---")
    u = tf.constant([1, 2, 3])
    v = tf.reshape(u, [1, 3])
    w = tf.constant([[10], [20]])
    r = v + w
    print(f"[1,3] + [2,1]:\n{r}")
    
    print("\n--- Incompatible shapes (error demo) ---")
    try:
        p = tf.constant([1, 2])
        q = tf.constant([1, 2, 3])
        bad = p + q
    except Exception as ex:
        print(f"Expected error: {type(ex).__name__}")
    
    print("\n--- tf.broadcast_to ---")
    small = tf.constant([1, 2, 3])
    big = tf.broadcast_to(small, [2, 3])
    print(f"broadcast_to([2,3]):\n{big}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
