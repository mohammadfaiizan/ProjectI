"""
LSTM/GRU-based sentiment analysis, Bidirectional.
"""
import tensorflow as tf

def build_sentiment_lstm(vocab_size=1000, embed_dim=64, seq_len=100, lstm_units=64, num_classes=2):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=False))(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)

def build_sentiment_gru(vocab_size=1000, embed_dim=64, seq_len=100, gru_units=64, num_classes=2):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(gru_units, return_sequences=False))(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)

def main():
    print("=" * 50)
    print("Sentiment Analysis with RNN (LSTM/GRU)")
    print("=" * 50)

    batch_size, seq_len, vocab_size = 8, 50, 1000

    print("\n--- Bidirectional LSTM ---")
    lstm_model = build_sentiment_lstm(vocab_size=vocab_size, seq_len=seq_len)
    x = tf.random.uniform((batch_size, seq_len), 0, vocab_size, dtype=tf.int32)
    y_lstm = lstm_model(x)
    print(f"LSTM output shape: {y_lstm.shape}")
    print(f"LSTM params: {lstm_model.count_params():,}")

    print("\n--- Bidirectional GRU ---")
    gru_model = build_sentiment_gru(vocab_size=vocab_size, seq_len=seq_len)
    y_gru = gru_model(x)
    print(f"GRU output shape: {y_gru.shape}")
    print(f"GRU params: {gru_model.count_params():,}")

    print("\n--- Forward inference ---")
    sample = tf.constant([[1, 42, 7, 0, 0]])
    pred = lstm_model(sample)
    print(f"Sample prediction: {pred.numpy()}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
