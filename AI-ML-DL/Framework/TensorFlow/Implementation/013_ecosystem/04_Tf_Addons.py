"""
TF Addons: tfa.layers, tfa.losses, tfa.optimizers (LAMB, SGDW, SWA concepts).
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("TensorFlow Addons")
    print("=" * 50)

    try:
        import tensorflow_addons as tfa
        print(f"tensorflow_addons version: {tfa.__version__}")
    except ImportError:
        print("tensorflow_addons not installed. Install: pip install tensorflow-addons")
        return

    print("\ntfa.layers - GroupNorm:")
    x = tf.random.normal((4, 32, 32, 64))
    gn = tfa.layers.GroupNormalization(groups=8)
    out = gn(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")

    print("\ntfa.losses - TripletSemiHardLoss:")
    anchor = tf.random.normal((8, 64))
    positive = tf.random.normal((8, 64))
    negative = tf.random.normal((8, 64))
    loss_fn = tfa.losses.TripletSemiHardLoss()
    loss = loss_fn(anchor, positive, negative)
    print(f"  Triplet loss value: {loss.numpy():.4f}")

    print("\ntfa.optimizers - LAMB:")
    lamb = tfa.optimizers.LAMB(learning_rate=1e-3)
    print(f"  LAMB optimizer: {lamb.get_config()}")

    print("\ntfa.optimizers - SGDW (weight decay):")
    sgdw = tfa.optimizers.SGDW(weight_decay=0.01, learning_rate=0.01)
    print(f"  SGDW with weight_decay=0.01")

    print("\nSWA (Stochastic Weight Averaging) concept:")
    print("  SWA averages model weights over training for better generalization")
    print("  Use tfa.optimizers.SWA or manual averaging of checkpoints")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10)
    ])
    model.compile(optimizer=lamb, loss='mse')
    x = tf.random.normal((32, 16))
    y = tf.random.normal((32, 10))
    model.fit(x, y, epochs=1, verbose=0)
    print("  Model trained with LAMB for 1 epoch")

    print("\nTF Addons demo complete.")

if __name__ == "__main__":
    main()
