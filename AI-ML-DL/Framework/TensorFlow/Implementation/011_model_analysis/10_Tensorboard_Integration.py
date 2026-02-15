"""
tf.summary (scalar, histogram, image, graph), SummaryWriter.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("TensorBoard Integration - tf.summary")
    print("=" * 50)

    logdir = "/tmp/tensorboard_demo"
    if os.path.exists(logdir):
        import shutil
        shutil.rmtree(logdir)
    os.makedirs(logdir, exist_ok=True)

    writer = tf.summary.create_file_writer(logdir)

    with writer.as_default():
        tf.summary.scalar("loss", 0.5, step=0)
        tf.summary.scalar("loss", 0.3, step=1)
        tf.summary.scalar("accuracy", 0.85, step=0)
        tf.summary.scalar("accuracy", 0.92, step=1)

        weights = tf.random.normal((64, 32))
        tf.summary.histogram("dense_weights", weights, step=0)

        img = tf.random.uniform((1, 28, 28, 1), 0, 1)
        tf.summary.image("sample_image", img, step=0, max_outputs=1)

    writer.flush()
    print(f"Summaries written to {logdir}")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    x = tf.random.normal((4, 16))
    with writer.as_default():
        tf.summary.trace_on(graph=True, profiler=False)
        model(x)
        with tf.summary.record_if(True):
            tf.summary.trace_export(name="model_graph", step=0)

    print("Graph trace recorded")
    print("To view: tensorboard --logdir=" + logdir)
    print("\nTensorBoard integration demo complete.")

if __name__ == "__main__":
    main()
