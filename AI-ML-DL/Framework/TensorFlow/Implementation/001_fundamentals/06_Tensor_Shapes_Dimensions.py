"""
Tensor shape, ndims, size, reshape basics, rank
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Shapes and Dimensions")
    print("=" * 50)
    
    print("\n--- shape ---")
    t = tf.constant([[1, 2, 3], [4, 5, 6]])
    print(f"Tensor:\n{t}")
    print(f"shape: {t.shape}")
    print(f"shape as list: {t.shape.as_list()}")
    
    print("\n--- ndim (rank) ---")
    print(f"ndim: {t.ndim}")
    t3d = tf.ones([2, 3, 4])
    print(f"3D tensor ndim: {t3d.ndim}")
    
    print("\n--- size ---")
    print(f"Total elements: {tf.size(t).numpy()}")
    
    print("\n--- reshape ---")
    flat = tf.reshape(t, [6])
    print(f"Reshape to [6]: {flat.numpy()}")
    
    row = tf.reshape(t, [1, 6])
    print(f"Reshape to [1,6]:\n{row}")
    
    col = tf.reshape(t, [6, 1])
    print(f"Reshape to [6,1]:\n{col}")
    
    print("\n--- -1 for inferred dimension ---")
    back = tf.reshape(t, [-1])
    print(f"Reshape to [-1]: {back.numpy()}")
    mat = tf.reshape(t, [3, -1])
    print(f"Reshape to [3,-1]:\n{mat}")
    
    print("\nVerification complete.")

if __name__ == "__main__":
    main()
