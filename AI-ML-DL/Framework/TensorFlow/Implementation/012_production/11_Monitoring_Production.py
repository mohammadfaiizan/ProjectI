"""
Production monitoring, data drift, logging.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("Production Monitoring")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    x = tf.random.normal((100, 16))
    y = tf.random.uniform((100,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)

    test_input = tf.random.normal((10, 16))
    preds = model.predict(test_input, verbose=0)
    print(f"Predictions shape: {preds.shape}")

    input_mean = tf.reduce_mean(test_input).numpy()
    input_std = tf.math.reduce_std(test_input).numpy()
    print(f"Input stats - mean: {input_mean:.4f}, std: {input_std:.4f}")

    pred_entropy = -tf.reduce_sum(preds * tf.math.log(preds + 1e-10), axis=1).numpy()
    print(f"Prediction entropy range: [{pred_entropy.min():.4f}, {pred_entropy.max():.4f}]")

    log_dir = os.path.join(os.path.dirname(__file__), "monitoring_logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "inference_log.txt"), 'a') as f:
        f.write(f"input_mean={input_mean}, input_std={input_std}, entropy_mean={pred_entropy.mean()}\n")
    print(f"Logged to {log_dir}")

    print("\nMonitoring concepts:")
    print("  - Data drift: track input distribution over time")
    print("  - Prediction drift: log confidence/entropy")
    print("  - Latency: log inference time per request")
    print("  - Error rate: track failed predictions")

    print("Monitoring demo complete.")

if __name__ == "__main__":
    main()
