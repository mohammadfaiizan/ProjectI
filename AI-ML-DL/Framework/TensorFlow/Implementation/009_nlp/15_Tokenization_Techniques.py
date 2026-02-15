"""
TextVectorization, subword tokenization concepts.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Tokenization Techniques")
    print("=" * 50)

    corpus = ["the quick brown fox", "jumps over the lazy dog", "hello world"]

    print("\n--- TextVectorization (word-level) ---")
    tv = tf.keras.layers.TextVectorization(
        max_tokens=100,
        output_mode="int",
        output_sequence_length=10,
        standardize="lower_and_strip_punctuation"
    )
    tv.adapt(corpus)
    encoded = tv(corpus)
    print(f"Encoded shape: {encoded.shape}")
    print(f"Vocabulary size: {tv.vocabulary_size()}")
    print(f"Sample encoded: {encoded[0].numpy()}")

    print("\n--- TextVectorization (word count) ---")
    tv_count = tf.keras.layers.TextVectorization(max_tokens=100, output_mode="count")
    tv_count.adapt(corpus)
    count_out = tv_count(corpus)
    print(f"Count output shape: {count_out.shape}")

    print("\n--- TextVectorization (tf-idf) ---")
    tv_tfidf = tf.keras.layers.TextVectorization(max_tokens=100, output_mode="tf_idf")
    tv_tfidf.adapt(corpus)
    tfidf_out = tv_tfidf(corpus)
    print(f"TF-IDF output shape: {tfidf_out.shape}")

    print("\n--- Subword concept (manual split) ---")
    def char_ngrams(text, n=3):
        text = text.lower().replace(" ", "")
        return [text[i:i+n] for i in range(len(text)-n+1)] if len(text) >= n else [text]
    sample = "hello"
    ngrams = char_ngrams(sample, 2)
    print(f"Bigrams of 'hello': {ngrams}")

    print("\n--- Vocabulary lookup ---")
    vocab = tv.get_vocabulary()
    print(f"First 10 vocab tokens: {vocab[:10]}")

    print("\n--- Decode from indices ---")
    decoded = tv.decode(encoded[0:1])
    print(f"Decoded: {decoded.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
