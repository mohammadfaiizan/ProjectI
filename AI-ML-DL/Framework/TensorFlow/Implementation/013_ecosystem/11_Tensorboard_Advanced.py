"""
TensorBoard advanced: custom scalars, embedding projector, HParams, mesh plugin.
"""
import os
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorBoard Advanced")
    print("=" * 50)

    logdir = os.path.join(os.path.dirname(__file__), "tb_logs")
    os.makedirs(logdir, exist_ok=True)

    writer = tf.summary.create_file_writer(logdir)

    print("\nCustom scalars:")
    with writer.as_default():
        for step in range(5):
            tf.summary.scalar("loss/train", 1.0 / (step + 1), step=step)
            tf.summary.scalar("loss/val", 1.2 / (step + 1), step=step)
    print("  Logged loss/train and loss/val")

    print("\nEmbedding projector:")
    embeddings = tf.random.normal((100, 16))
    metadata = [f"sample_{i}" for i in range(100)]
    metadata_path = os.path.join(logdir, "metadata.tsv")
    with open(metadata_path, "w") as f:
        f.write("index\tlabel\n")
        for i, m in enumerate(metadata):
            f.write(f"{i}\t{m}\n")
    with writer.as_default():
        tf.summary.embedding(embeddings, metadata=metadata, step=0)
    print("  Logged embeddings with metadata for projector")

    print("\nHParams:")
    try:
        from tensorboard.plugins.hparams import api as hp
        HP_LR = hp.HParam("learning_rate", hp.Discrete([0.001, 0.01]))
        HP_UNITS = hp.HParam("units", hp.Discrete([32, 64]))
        with tf.summary.create_file_writer(logdir).as_default():
            hp.hparams_config(
                hparams=[HP_LR, HP_UNITS],
                metrics=[hp.Metric("accuracy", display_name="Accuracy")]
            )
        with writer.as_default():
            hp.hparams({"learning_rate": 0.001, "units": 32})
        print("  HParams plugin: learning_rate, units")
    except ImportError:
        print("  HParams: use hp.hparams_config, hp.hparams")

    print("\nMesh plugin (3D visualization):")
    print("  tf.summary.mesh for 3D point clouds / meshes")
    print("  Requires vertices, faces, colors tensors")

    print("\nView with: tensorboard --logdir=" + logdir)
    print("\nTensorBoard advanced demo complete.")

if __name__ == "__main__":
    main()
