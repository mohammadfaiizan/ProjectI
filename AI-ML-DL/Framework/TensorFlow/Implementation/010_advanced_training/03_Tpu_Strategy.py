"""
TPUStrategy, TPU resolver, TPU initialization.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TPUStrategy")
    print("=" * 50)

    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print(f"TPU devices: {tf.config.list_logical_devices('TPU')}")

    strategy = tf.distribute.TPUStrategy(resolver)
    print(f"TPU replicas: {strategy.num_replicas_in_sync}")

    with strategy.scope():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    print("Model created under TPUStrategy scope")

    batch_size = 128
    global_batch = batch_size * strategy.num_replicas_in_sync
    print(f"Global batch size: {global_batch}")

    x = tf.random.normal((1280, 64))
    y = tf.random.uniform((1280,), maxval=10, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.batch(global_batch)
    ds = strategy.experimental_distribute_dataset(ds)
    print("Dataset distributed for TPU")

    history = model.fit(ds, epochs=2, verbose=0)
    print(f"TPU training loss: {history.history['loss'][-1]:.4f}")
    print("TPUStrategy demo complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"TPU not available (expected on non-TPU machines): {type(e).__name__}")
        print("Use TPUStrategy on Google Colab or Cloud TPU.")
