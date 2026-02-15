"""
Weight clustering for compression.
"""
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def main():
    print("=" * 50)
    print("Model Optimization - Clustering")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    x = tf.random.normal((200, 32))
    y = tf.random.uniform((200,), maxval=10, dtype=tf.int32)
    model.fit(x, y, epochs=2, verbose=0)
    print("Base model trained")

    cluster_weights = tfmot.clustering.keras.cluster_weights
    CentroidInitialization = tfmot.clustering.keras.CentroidInitialization
    clustering_params = {
        'number_of_clusters': 16,
        'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS
    }
    clustered_model = cluster_weights(model, **clustering_params)
    clustered_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Clustered model created")

    clustered_model.fit(x, y, epochs=2, verbose=0)
    print("Fine-tuning clustering complete")

    final_model = tfmot.clustering.keras.strip_clustering(clustered_model)
    print("Clustering stripped for deployment")

    unique_vals = set()
    for layer in final_model.layers:
        if hasattr(layer, 'kernel'):
            w = layer.kernel.numpy().flatten()
            unique_vals.update(w[:20].tolist())
    print(f"Sample unique weight values (clustered): {len(unique_vals)}")

    print("Clustering demo complete.")

if __name__ == "__main__":
    main()
