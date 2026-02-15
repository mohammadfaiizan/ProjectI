"""
@tf.function, XLA compilation, batch inference.
"""
import tensorflow as tf
import time

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

@tf.function
def inference_fn(model, x):
    return model(x, training=False)

@tf.function(jit_compile=True)
def inference_xla(model, x):
    return model(x, training=False)

def main():
    print("=" * 50)
    print("Inference Optimization")
    print("=" * 50)

    model = build_model()
    x_single = tf.random.normal((1, 64))
    x_batch = tf.random.normal((32, 64))

    start = time.perf_counter()
    for _ in range(100):
        _ = model(x_single, training=False)
    elapsed = time.perf_counter() - start
    print(f"Eager single-sample (100 runs): {elapsed*1000:.2f} ms")

    start = time.perf_counter()
    for _ in range(100):
        _ = inference_fn(model, x_single)
    elapsed = time.perf_counter() - start
    print(f"tf.function single-sample (100 runs): {elapsed*1000:.2f} ms")

    start = time.perf_counter()
    for _ in range(100):
        _ = inference_fn(model, x_batch)
    elapsed = time.perf_counter() - start
    print(f"tf.function batch-32 (100 runs): {elapsed*1000:.2f} ms")

    try:
        _ = inference_xla(model, x_batch)
        print("XLA compilation successful")
    except Exception as e:
        print(f"XLA fallback: {e}")

    print("\nInference optimization demo complete.")

if __name__ == "__main__":
    main()
