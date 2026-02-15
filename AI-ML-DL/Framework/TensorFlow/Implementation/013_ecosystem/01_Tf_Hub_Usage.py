"""
TF Hub: hub.KerasLayer, feature extraction, fine-tuning.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorFlow Hub Usage")
    print("=" * 50)

    try:
        import tensorflow_hub as hub
        print(f"tensorflow_hub version: {hub.__version__}")
    except ImportError:
        print("tensorflow_hub not installed. Install: pip install tensorflow-hub")
        return

    hub_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
    print(f"\nExample Hub URL: {hub_url}")

    print("\nFeature extraction (frozen backbone):")
    feature_extractor = hub.KerasLayer(hub_url, trainable=False, output_shape=[1280])
    model_fe = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_fe.build([None, 224, 224, 3])
    print(f"  Feature vector dim: {feature_extractor.output_shape}")
    print(f"  Trainable params (backbone): 0")

    print("\nFine-tuning (trainable backbone):")
    feature_extractor_ft = hub.KerasLayer(hub_url, trainable=True, output_shape=[1280])
    model_ft = tf.keras.Sequential([
        feature_extractor_ft,
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model_ft.build([None, 224, 224, 3])
    trainable_count = sum(tf.keras.backend.count_params(w) for w in model_ft.trainable_weights)
    print(f"  Trainable params: {trainable_count}")

    print("\nInference example:")
    dummy = tf.random.normal((2, 224, 224, 3))
    out = model_fe(dummy)
    print(f"  Input shape: {dummy.shape} -> Output shape: {out.shape}")

    print("\nTF Hub demo complete.")

if __name__ == "__main__":
    main()
