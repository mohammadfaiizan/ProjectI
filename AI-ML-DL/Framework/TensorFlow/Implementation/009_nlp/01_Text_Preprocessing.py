"""
Text cleaning, tokenization, tf.strings operations.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Text Preprocessing with tf.strings")
    print("=" * 50)

    texts = tf.constant(["  Hello, World!  ", "TensorFlow NLP", "UPPERCASE text"])
    print("\n--- Lowercasing ---")
    lower = tf.strings.lower(texts)
    print(f"Lower: {[t.numpy().decode() for t in lower.numpy()]}")

    print("\n--- Strip whitespace ---")
    stripped = tf.strings.strip(texts)
    print(f"Stripped: {[t.numpy().decode() for t in stripped.numpy()]}")

    print("\n--- Split by delimiter ---")
    split = tf.strings.split(texts, sep=" ")
    print(f"Split: {split}")

    print("\n--- Join ---")
    joined = tf.strings.join([["a", "b"], ["c", "d"]], separator="-")
    print(f"Joined: {joined.numpy()}")

    print("\n--- Regex replace ---")
    clean = tf.strings.regex_replace(texts, "[^a-zA-Z ]", "")
    print(f"Regex cleaned: {[t.numpy().decode() for t in clean.numpy()]}")

    print("\n--- String length ---")
    lengths = tf.strings.length(texts)
    print(f"Lengths: {lengths.numpy()}")

    print("\n--- Unicode decode ---")
    encoded = tf.constant([b"hello", b"world"])
    decoded = tf.strings.unicode_decode(encoded, "UTF-8")
    print(f"Decoded: {decoded}")

    print("\n--- Tokenization with split ---")
    corpus = tf.constant(["the quick brown fox", "jumps over the lazy dog"])
    tokens = tf.strings.split(corpus)
    print(f"Tokens: {tokens}")

    print("\n--- Vocabulary building (manual) ---")
    flat = tf.strings.split(tf.reshape(corpus, [-1])).flat_values
    unique, _ = tf.unique(flat)
    print(f"Unique tokens: {unique.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
