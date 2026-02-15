"""
Data transformations: Log, power transform, binning, encoding.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Data Transformation Operations")
    print("=" * 50)

    print("\n--- Log transform ---")
    x = tf.constant([1.0, 10.0, 100.0, 1000.0])
    log_x = tf.math.log(x + 1.0)
    log10_x = tf.math.log(x + 1.0) / tf.math.log(10.0)
    print(f"log(x+1): {log_x.numpy()}")
    print(f"log10(x+1): {log10_x.numpy()}")

    print("\n--- Power transform ---")
    data = tf.constant([1.0, 2.0, 3.0, 4.0])
    sqrt_x = tf.pow(data, 0.5)
    sq_x = tf.pow(data, 2.0)
    print(f"sqrt: {sqrt_x.numpy()}")
    print(f"squared: {sq_x.numpy()}")

    print("\n--- Binning (discretization) ---")
    vals = tf.constant([0.5, 1.5, 2.5, 3.5, 4.5])
    disc_layer = tf.keras.layers.Discretization(bin_boundaries=[1.0, 2.0, 3.0, 4.0])
    bin_indices = disc_layer(vals)
    print(f"Binned indices: {bin_indices.numpy()}")

    print("\n--- One-hot encoding ---")
    labels = tf.constant([0, 1, 2, 0])
    onehot = tf.one_hot(labels, depth=3)
    print(f"One-hot:\n{onehot.numpy()}")

    print("\n--- Binarization (threshold) ---")
    t = tf.constant([0.3, 0.6, 0.2, 0.8])
    binary = tf.cast(t > 0.5, tf.float32)
    print(f"Binary (>0.5): {binary.numpy()}")

    print("\n--- Reciprocal and square root ---")
    r = tf.constant([1.0, 2.0, 4.0])
    recip = tf.math.reciprocal(r)
    print(f"Reciprocal: {recip.numpy()}")

    print("\n--- Box-Cox style (log1p) ---")
    pos = tf.constant([0.1, 1.0, 10.0])
    log1p = tf.math.log1p(pos)
    print(f"log1p: {log1p.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
