"""
Extractive QA (start/end logits).
"""
import tensorflow as tf

def build_qa_model(vocab_size=1000, seq_len=128, embed_dim=64, lstm_units=64):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    start_logits = tf.keras.layers.Dense(1)(x)
    end_logits = tf.keras.layers.Dense(1)(x)
    start_logits = tf.keras.layers.Flatten()(start_logits)
    end_logits = tf.keras.layers.Flatten()(end_logits)
    return tf.keras.Model(inp, [start_logits, end_logits])

def main():
    print("=" * 50)
    print("Extractive Question Answering")
    print("=" * 50)

    batch_size, seq_len, vocab_size = 4, 64, 500

    model = build_qa_model(vocab_size=vocab_size, seq_len=seq_len)

    x = tf.random.uniform((batch_size, seq_len), 0, vocab_size, dtype=tf.int32)
    start_logits, end_logits = model(x)

    print(f"Start logits shape: {start_logits.shape}")
    print(f"End logits shape: {end_logits.shape}")

    print("\n--- Span extraction ---")
    start_probs = tf.nn.softmax(start_logits)
    end_probs = tf.nn.softmax(end_logits)
    start_idx = tf.argmax(start_probs, axis=1)
    end_idx = tf.argmax(end_probs, axis=1)
    print(f"Predicted start indices: {start_idx.numpy()}")
    print(f"Predicted end indices: {end_idx.numpy()}")

    print("\n--- QA loss ---")
    start_labels = tf.random.uniform((batch_size,), 0, seq_len, dtype=tf.int32)
    end_labels = tf.random.uniform((batch_size,), 0, seq_len, dtype=tf.int32)
    loss_start = tf.keras.losses.sparse_categorical_crossentropy(start_labels, start_logits, from_logits=True)
    loss_end = tf.keras.losses.sparse_categorical_crossentropy(end_labels, end_logits, from_logits=True)
    loss = tf.reduce_mean(loss_start) + tf.reduce_mean(loss_end)
    print(f"QA loss: {loss.numpy():.4f}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
