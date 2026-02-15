"""
Categorical preprocessing: CategoryEncoding, StringLookup, IntegerLookup, Hashing.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Categorical Preprocessing Layers")
    print("=" * 50)

    print("\n--- CategoryEncoding (one-hot) ---")
    cat_data = tf.constant([["a"], ["b"], ["a"], ["c"]])
    lookup = tf.keras.layers.StringLookup(vocabulary=["a", "b", "c"])
    encoded = lookup(cat_data)
    enc = tf.keras.layers.CategoryEncoding(num_tokens=4, output_mode="one_hot")
    onehot = enc(encoded)
    print(f"One-hot shape: {onehot.shape}")
    print(f"One-hot sample: {onehot[0].numpy()}")

    print("\n--- StringLookup ---")
    vocab = ["red", "green", "blue"]
    str_lookup = tf.keras.layers.StringLookup(vocabulary=vocab)
    colors = tf.constant([["red"], ["blue"], ["green"], ["red"]])
    indices = str_lookup(colors)
    print(f"StringLookup indices: {indices.numpy().flatten()}")

    print("\n--- IntegerLookup ---")
    int_lookup = tf.keras.layers.IntegerLookup(vocabulary=[10, 20, 30])
    int_data = tf.constant([[10], [30], [20], [10]])
    int_indices = int_lookup(int_data)
    print(f"IntegerLookup indices: {int_indices.numpy().flatten()}")

    print("\n--- Hashing ---")
    hash_layer = tf.keras.layers.Hashing(num_bins=32)
    hashed = hash_layer(tf.constant([["cat"], ["dog"], ["cat"]]))
    print(f"Hashed output: {hashed.numpy().flatten()}")

    print("\n--- CategoryEncoding (multi-hot) ---")
    multi_data = tf.constant([[1, 2], [2, 3], [1, 2]])
    multi_enc = tf.keras.layers.CategoryEncoding(num_tokens=4, output_mode="multi_hot")
    multi_out = multi_enc(multi_data)
    print(f"Multi-hot shape: {multi_out.shape}")
    print(f"Multi-hot [1,2]: {multi_out[0].numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
