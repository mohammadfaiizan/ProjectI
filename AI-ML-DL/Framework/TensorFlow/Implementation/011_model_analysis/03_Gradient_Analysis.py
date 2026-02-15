"""
Gradient flow visualization, vanishing/exploding detection.
"""
import tensorflow as tf
import numpy as np

def build_deep_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def main():
    print("=" * 50)
    print("Gradient Analysis - Vanishing/Exploding Detection")
    print("=" * 50)

    model = build_deep_model()
    x = tf.random.normal((16, 32))
    y = tf.random.uniform((16,), maxval=10, dtype=tf.int32)

    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, logits)
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    print(f"Number of layers with gradients: {len([g for g in grads if g is not None])}")

    grad_norms = []
    for i, (g, v) in enumerate(zip(grads, model.trainable_variables)):
        if g is not None:
            norm = tf.norm(g).numpy()
            grad_norms.append(norm)
            status = "OK"
            if norm < 1e-7:
                status = "VANISHING"
            elif norm > 1e3:
                status = "EXPLODING"
            print(f"Layer {i} ({v.name}): norm={norm:.6e} [{status}]")

    total_norm = np.sqrt(sum(n**2 for n in grad_norms))
    print(f"\nTotal gradient norm: {total_norm:.6e}")

    vanishing_count = sum(1 for n in grad_norms if n < 1e-7)
    exploding_count = sum(1 for n in grad_norms if n > 1e3)
    print(f"Vanishing layers: {vanishing_count}, Exploding layers: {exploding_count}")

    print("\nGradient analysis demo complete.")

if __name__ == "__main__":
    main()
