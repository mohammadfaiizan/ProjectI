"""
Text preprocessing: TextVectorization with output_mode, max_tokens, output_sequence_length.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Text Preprocessing Layers")
    print("=" * 50)

    print("\n--- TextVectorization (int) ---")
    texts = ["hello world", "world of tensorflow", "hello tensorflow"]
    text_vec = tf.keras.layers.TextVectorization(
        max_tokens=20,
        output_mode="int",
        output_sequence_length=5
    )
    text_vec.adapt(tf.constant(texts))
    encoded = text_vec(tf.constant(["hello world"]))
    print(f"Encoded shape: {encoded.shape}")
    print(f"Encoded: {encoded.numpy()}")

    print("\n--- TextVectorization (multi-hot) ---")
    text_multi = tf.keras.layers.TextVectorization(
        max_tokens=20,
        output_mode="multi_hot",
        output_sequence_length=None
    )
    text_multi.adapt(tf.constant(texts))
    multi_out = text_multi(tf.constant(["hello world"]))
    print(f"Multi-hot shape: {multi_out.shape}")
    print(f"Multi-hot sample: {multi_out[0].numpy()}")

    print("\n--- TextVectorization (count) ---")
    text_count = tf.keras.layers.TextVectorization(
        max_tokens=20,
        output_mode="count"
    )
    text_count.adapt(tf.constant(texts))
    count_out = text_count(tf.constant(["hello hello world"]))
    print(f"Count output shape: {count_out.shape}")
    print(f"Count sample: {count_out[0].numpy()}")

    print("\n--- output_sequence_length padding ---")
    text_pad = tf.keras.layers.TextVectorization(
        max_tokens=15,
        output_mode="int",
        output_sequence_length=8
    )
    text_pad.adapt(tf.constant(["short", "a much longer sentence here"]))
    pad_out = text_pad(tf.constant(["short"]))
    print(f"Padded sequence length: {pad_out.shape[1]}")
    print(f"Padded: {pad_out.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
