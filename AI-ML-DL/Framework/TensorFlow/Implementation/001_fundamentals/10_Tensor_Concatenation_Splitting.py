"""
tf.concat, tf.stack, tf.unstack, tf.split, tf.tile
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Concatenation and Splitting")
    print("=" * 50)
    
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5, 6], [7, 8]])
    
    print("\n--- tf.concat ---")
    cat0 = tf.concat([a, b], axis=0)
    print(f"concat(axis=0):\n{cat0}")
    cat1 = tf.concat([a, b], axis=1)
    print(f"concat(axis=1):\n{cat1}")
    
    print("\n--- tf.stack ---")
    st = tf.stack([a, b], axis=0)
    print(f"stack(axis=0) shape: {st.shape}")
    print(f"Result:\n{st}")
    
    st1 = tf.stack([a, b], axis=1)
    print(f"stack(axis=1) shape: {st1.shape}")
    
    print("\n--- tf.unstack ---")
    stacked = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    unstacked = tf.unstack(stacked, axis=0)
    print(f"unstack: {len(unstacked)} tensors")
    for i, u in enumerate(unstacked):
        print(f"  [{i}]:\n{u}")
    
    print("\n--- tf.split ---")
    t = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])
    parts = tf.split(t, 2, axis=1)
    print(f"split(2, axis=1): {len(parts)} parts")
    for i, p in enumerate(parts):
        print(f"  Part {i}:\n{p}")
    
    parts2 = tf.split(t, [1, 3], axis=1)
    print(f"split([1,3], axis=1):")
    print(f"  Part 0: {parts2[0].numpy()}")
    print(f"  Part 1:\n{parts2[1]}")
    
    print("\n--- tf.tile ---")
    small = tf.constant([[1, 2]])
    tiled = tf.tile(small, [2, 3])
    print(f"tile([2,3]):\n{tiled}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
