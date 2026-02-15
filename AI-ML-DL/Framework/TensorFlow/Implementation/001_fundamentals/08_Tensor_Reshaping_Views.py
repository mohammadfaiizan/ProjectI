"""
tf.reshape, tf.transpose, tf.expand_dims, tf.squeeze, tf.tile
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Reshaping and Views")
    print("=" * 50)
    
    t = tf.constant([[1, 2], [3, 4]])
    print(f"Original:\n{t}")
    
    print("\n--- tf.reshape ---")
    r = tf.reshape(t, [1, 4])
    print(f"reshape([1,4]):\n{r}")
    
    print("\n--- tf.transpose ---")
    tp = tf.transpose(t)
    print(f"transpose():\n{tp}")
    
    t3 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    tp3 = tf.transpose(t3, perm=[1, 0, 2])
    print(f"3D transpose(perm=[1,0,2]) shape: {tp3.shape}")
    
    print("\n--- tf.expand_dims ---")
    e = tf.expand_dims(t, axis=0)
    print(f"expand_dims(axis=0) shape: {e.shape}")
    print(f"Result:\n{e}")
    
    e1 = tf.expand_dims(t, axis=-1)
    print(f"expand_dims(axis=-1) shape: {e1.shape}")
    
    print("\n--- tf.squeeze ---")
    s = tf.squeeze(e)
    print(f"squeeze() shape: {s.shape}")
    
    t_extra = tf.constant([[[1], [2], [3]]])
    s2 = tf.squeeze(t_extra)
    print(f"squeeze single-dim: {s2.numpy()}")
    
    print("\n--- tf.tile ---")
    tile = tf.tile(t, [2, 2])
    print(f"tile([2,2]):\n{tile}")
    
    tile2 = tf.tile(tf.reshape(t, [2, 2, 1]), [1, 1, 3])
    print(f"tile for broadcasting: shape {tile2.shape}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
