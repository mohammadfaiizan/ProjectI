"""
Token classification, BIO tagging.
"""
import tensorflow as tf

def build_ner_model(vocab_size=1000, embed_dim=64, seq_len=50, lstm_units=64, num_tags=9):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units, return_sequences=True))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(num_tags, activation="softmax")(x)
    return tf.keras.Model(inp, out)

def main():
    print("=" * 50)
    print("Named Entity Recognition (BIO Tagging)")
    print("=" * 50)

    batch_size, seq_len, vocab_size = 4, 30, 500
    num_tags = 9

    bio_tags = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
    print(f"\nBIO tags: {bio_tags}")

    model = build_ner_model(vocab_size=vocab_size, seq_len=seq_len, num_tags=num_tags)

    x = tf.random.uniform((batch_size, seq_len), 0, vocab_size, dtype=tf.int32)
    logits = model(x)

    print(f"\nOutput shape: {logits.shape}")

    print("\n--- Token-level prediction ---")
    pred_tags = tf.argmax(logits, axis=-1)
    print(f"Predicted tag ids (first seq): {pred_tags[0].numpy()}")

    print("\n--- Loss (sparse, masked) ---")
    labels = tf.random.uniform((batch_size, seq_len), 0, num_tags, dtype=tf.int32)
    mask = tf.cast(x != 0, tf.float32)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    per_token_loss = loss_fn(labels, logits)
    masked_loss = tf.reduce_sum(per_token_loss * mask) / (tf.reduce_sum(mask) + 1e-8)
    print(f"Masked loss: {masked_loss.numpy():.4f}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
