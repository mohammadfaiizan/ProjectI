"""
MultiWorkerMirroredStrategy setup and usage.
"""
import tensorflow as tf
import os

def main():
    print("=" * 50)
    print("MultiWorkerMirroredStrategy")
    print("=" * 50)

    os.environ['TF_CONFIG'] = '{"cluster":{"worker":["localhost:12345","localhost:12346"]},"task":{"type":"worker","index":0}}'
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print(f"Cluster resolver: {strategy.cluster_resolver}")

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy'
        )
    print("Model built under MultiWorker scope")

    num_workers = strategy.num_replicas_in_sync
    batch_size = 32
    global_batch = batch_size * num_workers
    print(f"Workers: {num_workers}, global batch: {global_batch}")

    x = tf.random.normal((320, 16))
    y = tf.random.uniform((320,), maxval=10, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(global_batch)
    ds = strategy.experimental_distribute_dataset(ds)
    print("Dataset sharded across workers")

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)
    print("Auto-shard policy: DATA")

    print("MultiWorkerMirroredStrategy setup verified.")
    print("Run with: python -m tensorflow.python.distribute.multi_worker_runner")

if __name__ == "__main__":
    main()
