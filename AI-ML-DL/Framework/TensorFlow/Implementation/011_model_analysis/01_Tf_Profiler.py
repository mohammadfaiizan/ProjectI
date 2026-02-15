"""
tf.profiler usage, profiling training steps.
"""
import tensorflow as tf
import os

def build_simple_model():
    return tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

def main():
    print("=" * 50)
    print("TensorFlow Profiler - Training Step Profiling")
    print("=" * 50)

    model = build_simple_model()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    x = tf.random.normal((1024, 32))
    y = tf.random.uniform((1024,), maxval=10, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(32).prefetch(tf.data.AUTOTUNE)

    logdir = "/tmp/tf_profiler_log"
    if os.path.exists(logdir):
        import shutil
        shutil.rmtree(logdir)
    os.makedirs(logdir, exist_ok=True)

    options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3,
        python_tracer_level=1,
        device_tracer_level=1
    )

    tf.profiler.experimental.start(logdir, options=options)
    print("Profiler started")

    model.fit(ds, epochs=2, steps_per_epoch=10, verbose=0)
    print("Training completed")

    tf.profiler.experimental.stop()
    print("Profiler stopped")

    print(f"Profile logs saved to {logdir}")
    print("To view: tensorboard --logdir=" + logdir)
    print("TF Profiler demo complete.")

if __name__ == "__main__":
    main()
