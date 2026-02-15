"""
Saliency maps and attribution methods for model debugging.
"""
import tensorflow as tf
import numpy as np

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

def saliency_map(model, x, target_class=0):
    with tf.GradientTape() as tape:
        tape.watch(x)
        pred = model(x)
        target = pred[:, target_class]
    grads = tape.gradient(target, x)
    return tf.reduce_mean(tf.abs(grads), axis=0)

def integrated_gradients(model, x, baseline=None, steps=10):
    if baseline is None:
        baseline = tf.zeros_like(x)
    alphas = tf.linspace(0.0, 1.0, steps)
    grads_sum = tf.zeros_like(x)
    for a in alphas:
        interp = baseline + a * (x - baseline)
        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = model(interp)
            target = tf.reduce_sum(pred)
        grads = tape.gradient(target, interp)
        grads_sum += grads
    avg_grads = grads_sum / tf.cast(steps, tf.float32)
    return (x - baseline) * avg_grads

def main():
    print("=" * 50)
    print("Saliency and Attribution")
    print("=" * 50)

    model = build_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    x = tf.random.normal((1, 10))
    sal = saliency_map(model, x)
    print(f"Saliency map shape: {sal.shape}")
    print(f"Saliency values (top 3): {tf.sort(sal, direction='DESCENDING').numpy()[:3]}")

    ig = integrated_gradients(model, x, steps=5)
    print(f"\nIntegrated gradients shape: {ig.shape}")
    print(f"IG sum: {tf.reduce_sum(ig).numpy():.4f}")

    x_batch = tf.random.normal((4, 10))
    sal_batch = saliency_map(model, x_batch)
    print(f"\nBatch saliency mean: {tf.reduce_mean(sal_batch).numpy():.4f}")

    print("\nSaliency and attribution demo complete.")

if __name__ == "__main__":
    main()
