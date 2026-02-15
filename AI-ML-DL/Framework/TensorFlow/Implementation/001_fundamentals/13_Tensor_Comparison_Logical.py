"""
tf.equal, not_equal, greater, less, logical_and, logical_or, tf.where, tf.sort, tf.argsort
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Comparison and Logical Ops")
    print("=" * 50)
    
    a = tf.constant([1, 2, 3, 4, 5])
    b = tf.constant([1, 3, 2, 4, 5])
    
    print("\n--- Comparison ops ---")
    print(f"equal: {tf.equal(a, b).numpy()}")
    print(f"not_equal: {tf.not_equal(a, b).numpy()}")
    print(f"greater: {tf.greater(a, b).numpy()}")
    print(f"less: {tf.less(a, b).numpy()}")
    print(f"greater_equal: {tf.greater_equal(a, b).numpy()}")
    
    print("\n--- Logical ops ---")
    x = tf.constant([True, True, False, False])
    y = tf.constant([True, False, True, False])
    print(f"logical_and: {tf.logical_and(x, y).numpy()}")
    print(f"logical_or: {tf.logical_or(x, y).numpy()}")
    print(f"logical_not: {tf.logical_not(x).numpy()}")
    
    print("\n--- tf.where ---")
    cond = tf.constant([True, False, True, False])
    t_val = tf.constant([10, 10, 10, 10])
    f_val = tf.constant([0, 0, 0, 0])
    w = tf.where(cond, t_val, f_val)
    print(f"where(cond, 10, 0): {w.numpy()}")
    
    indices = tf.where(tf.greater(a, 3))
    print(f"where(a > 3) indices:\n{indices}")
    
    print("\n--- tf.sort ---")
    vals = tf.constant([3, 1, 4, 1, 5])
    sorted_asc = tf.sort(vals)
    sorted_desc = tf.sort(vals, direction="DESCENDING")
    print(f"sort ascending: {sorted_asc.numpy()}")
    print(f"sort descending: {sorted_desc.numpy()}")
    
    print("\n--- tf.argsort ---")
    idx = tf.argsort(vals)
    print(f"argsort: {idx.numpy()}")
    print(f"gather by argsort: {tf.gather(vals, idx).numpy()}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
