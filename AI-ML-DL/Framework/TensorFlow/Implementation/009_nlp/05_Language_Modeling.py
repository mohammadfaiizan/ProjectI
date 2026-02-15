"""
Character/word-level language model, next-token prediction.
"""
import tensorflow as tf

def build_char_lm(vocab_size=128, embed_dim=64, rnn_units=128):
    inp = tf.keras.layers.Input(shape=(None,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inp)
    x = tf.keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    x = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inp, x)

def build_word_lm(vocab_size=1000, embed_dim=64, rnn_units=128):
    inp = tf.keras.layers.Input(shape=(None,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inp)
    x = tf.keras.layers.GRU(rnn_units, return_sequences=True)(x)
    x = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inp, x)

def main():
    print("=" * 50)
    print("Language Modeling (Char/Word Level)")
    print("=" * 50)

    batch_size, seq_len = 4, 20
    char_vocab, word_vocab = 128, 1000

    print("\n--- Character-level LM ---")
    char_model = build_char_lm(vocab_size=char_vocab)
    x_char = tf.random.uniform((batch_size, seq_len), 0, char_vocab, dtype=tf.int32)
    logits_char = char_model(x_char)
    print(f"Char logits shape: {logits_char.shape}")

    print("\n--- Word-level LM ---")
    word_model = build_word_lm(vocab_size=word_vocab)
    x_word = tf.random.uniform((batch_size, seq_len), 0, word_vocab, dtype=tf.int32)
    logits_word = word_model(x_word)
    print(f"Word logits shape: {logits_word.shape}")

    print("\n--- Next-token prediction loss ---")
    targets = x_word[:, 1:]
    logits_shifted = logits_word[:, :-1, :]
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits_shifted, from_logits=True)
    print(f"LM loss: {tf.reduce_mean(loss).numpy():.4f}")

    print("\n--- Sampling next token ---")
    last_logits = logits_word[0, -1, :]
    next_token = tf.random.categorical(tf.expand_dims(last_logits, 0), 1)[0, 0]
    print(f"Next token id: {next_token.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
