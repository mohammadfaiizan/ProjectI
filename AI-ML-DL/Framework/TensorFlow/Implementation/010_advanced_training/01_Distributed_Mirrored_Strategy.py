"""
tf.distribute.MirroredStrategy, scope, dataset distribution.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("MirroredStrategy - Single Machine Multi-GPU")
    print("=" * 50)

    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    print("Model created under strategy scope")

    batch_size = 64
    global_batch_size = batch_size * strategy.num_replicas_in_sync
    print(f"Per-replica batch: {batch_size}, global batch: {global_batch_size}")

    x = tf.random.normal((640, 32))
    y = tf.random.uniform((640,), maxval=10, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(global_batch_size)
    ds = strategy.experimental_distribute_dataset(ds)
    print("Dataset distributed across replicas")

    history = model.fit(ds, epochs=2, verbose=0)
    print(f"Training loss: {history.history['loss'][-1]:.4f}")

    strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])
    print(f"Explicit devices: {strategy.extended.worker_devices}")
    print("MirroredStrategy demo complete.")

if __name__ == "__main__":
    main()
