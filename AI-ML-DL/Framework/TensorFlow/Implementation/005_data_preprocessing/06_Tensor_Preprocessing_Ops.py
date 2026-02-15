"""
Manual tensor preprocessing: normalize, standardize, clip.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tensor Preprocessing Operations")
    print("=" * 50)

    print("\n--- Normalize (L2) ---")
    x = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    normalized = tf.math.l2_normalize(x, axis=-1)
    norms = tf.norm(normalized, axis=-1)
    print(f"L2 norms after normalize: {norms.numpy()}")
    print(f"Normalized sample: {normalized[0].numpy()}")

    print("\n--- Standardize (z-score) ---")
    data = tf.constant([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    mean = tf.reduce_mean(data, axis=0)
    std = tf.math.reduce_std(data, axis=0)
    standardized = (data - mean) / (std + 1e-8)
    print(f"Standardized mean: {tf.reduce_mean(standardized).numpy():.4f}")
    print(f"Standardized std: {tf.math.reduce_std(standardized).numpy():.4f}")

    print("\n--- Clip ---")
    vals = tf.constant([-5.0, 2.0, 10.0, 0.5, -1.0])
    clipped = tf.clip_by_value(vals, 0.0, 5.0)
    print(f"Clipped [0,5]: {clipped.numpy()}")

    print("\n--- Clip by global norm ---")
    grads = [tf.constant([3.0, 4.0]), tf.constant([1.0, 0.0])]
    clipped_grads, _ = tf.clip_by_global_norm(grads, 2.0)
    print(f"Clipped grad[0]: {clipped_grads[0].numpy()}")

    print("\n--- Min-max scaling ---")
    raw = tf.constant([[1.0, 5.0], [2.0, 10.0], [3.0, 15.0]])
    min_val = tf.reduce_min(raw, axis=0)
    max_val = tf.reduce_max(raw, axis=0)
    scaled = (raw - min_val) / (max_val - min_val + 1e-8)
    print(f"Min-max scaled: {scaled.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
