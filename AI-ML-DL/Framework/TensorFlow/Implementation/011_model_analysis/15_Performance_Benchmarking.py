"""
tf.test.Benchmark, latency/throughput measurement.
"""
import tensorflow as tf
import time

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(128,)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

class ModelBenchmark(tf.test.Benchmark):
    def benchmark_inference(self):
        model = build_model()
        x = tf.random.normal((32, 128))
        result = model(x, training=False)
        self.run_op_benchmark(iters=100, op=result)

def run_manual_benchmark():
    model = build_model()
    x_single = tf.random.normal((1, 128))
    x_batch = tf.random.normal((32, 128))

    warmup = 50
    runs = 200
    for _ in range(warmup):
        _ = model(x_single, training=False)

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x_single, training=False)
    latency_ms = (time.perf_counter() - start) / runs * 1000
    print(f"Single-sample latency: {latency_ms:.3f} ms")
    print(f"Throughput: {1000/latency_ms:.1f} samples/sec")

    start = time.perf_counter()
    for _ in range(runs):
        _ = model(x_batch, training=False)
    batch_time_ms = (time.perf_counter() - start) / runs * 1000
    print(f"Batch-32 latency: {batch_time_ms:.3f} ms")
    print(f"Batch throughput: {32 * 1000 / batch_time_ms:.1f} samples/sec")

def main():
    print("=" * 50)
    print("Performance Benchmarking")
    print("=" * 50)

    run_manual_benchmark()

    try:
        bench = ModelBenchmark()
        bench.benchmark_inference()
        print("\ntf.test.Benchmark run completed")
    except Exception as e:
        print(f"\ntf.test.Benchmark: {e}")

    print("\nPerformance benchmarking demo complete.")

if __name__ == "__main__":
    main()
