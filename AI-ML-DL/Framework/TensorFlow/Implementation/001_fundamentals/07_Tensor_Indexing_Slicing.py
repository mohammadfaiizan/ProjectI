"""
Basic indexing, slicing, tf.gather, tf.gather_nd, tf.boolean_mask
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Indexing and Slicing")
    print("=" * 50)
    
    t = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Tensor:\n{t}")
    
    print("\n--- Basic indexing ---")
    print(f"t[0]: {t[0].numpy()}")
    print(f"t[1, 2]: {t[1, 2].numpy()}")
    
    print("\n--- Slicing ---")
    print(f"t[1:3]:\n{t[1:3]}")
    print(f"t[:, 1]: {t[:, 1].numpy()}")
    print(f"t[0:2, 1:3]:\n{t[0:2, 1:3]}")
    
    print("\n--- tf.gather ---")
    indices = tf.constant([0, 2])
    g = tf.gather(t, indices, axis=0)
    print(f"gather(axis=0, indices=[0,2]):\n{g}")
    
    g1 = tf.gather(t, [1, 2], axis=1)
    print(f"gather(axis=1, indices=[1,2]):\n{g1}")
    
    print("\n--- tf.gather_nd ---")
    idx = tf.constant([[0, 0], [1, 1], [2, 2]])
    gnd = tf.gather_nd(t, idx)
    print(f"gather_nd([[0,0],[1,1],[2,2]]): {gnd.numpy()}")
    
    print("\n--- tf.boolean_mask ---")
    mask = tf.constant([True, False, True])
    bm = tf.boolean_mask(t, mask, axis=0)
    print(f"boolean_mask(axis=0):\n{bm}")
    
    mask2 = t > 5
    bm2 = tf.boolean_mask(t, mask2)
    print(f"boolean_mask(t > 5): {bm2.numpy()}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
