"""
Memory tracking, tf.config.experimental.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Memory Profiling - tf.config.experimental")
    print("=" * 50)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"Memory growth config: {e}")

    tf.config.experimental.reset_memory_stats('GPU:0')
    print("Memory stats reset (if GPU available)")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(64,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    x = tf.random.normal((128, 64))
    y = tf.random.uniform((128,), maxval=10, dtype=tf.int32)

    model.fit(x, y, epochs=2, verbose=0)
    print("Model trained")

    try:
        mem_stats = tf.config.experimental.get_memory_info('GPU:0')
        print(f"GPU memory - current: {mem_stats['current'] / 1e6:.2f} MB")
        print(f"GPU memory - peak: {mem_stats['peak'] / 1e6:.2f} MB")
    except (ValueError, RuntimeError) as e:
        print(f"GPU memory stats (CPU fallback): {e}")

    large_tensor = tf.random.normal((1000, 1000))
    del large_tensor
    print("Large tensor created and released")

    print("\nMemory profiling demo complete.")

if __name__ == "__main__":
    main()
