"""
GPT-style causal LM (decoder-only, causal mask).
"""
import tensorflow as tf
import numpy as np

def causal_mask(seq_len):
    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return mask * -1e9

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis].astype(np.float32)
    dim = np.arange(d_model)[np.newaxis, :].astype(np.float32)
    angle = pos / np.power(10000, 2 * (dim // 2) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(angle)

def gpt_block(x, d_model=128, num_heads=4, ff_dim=256, seq_len=50):
    mask = causal_mask(seq_len)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x, attention_mask=mask)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="gelu"),
        tf.keras.layers.Dense(d_model)
    ])(x)
    return tf.keras.layers.LayerNormalization()(x + ffn)

def build_gpt(vocab_size=1000, seq_len=50, d_model=128, num_heads=4, ff_dim=256, num_layers=2):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inp)
    x = x + positional_encoding(seq_len, d_model)
    for _ in range(num_layers):
        x = gpt_block(x, d_model, num_heads, ff_dim, seq_len)
    logits = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inp, logits)

def main():
    print("=" * 50)
    print("GPT-style Causal Language Model")
    print("=" * 50)

    batch_size, seq_len, vocab = 4, 32, 500

    model = build_gpt(vocab_size=vocab, seq_len=seq_len)

    x = tf.random.uniform((batch_size, seq_len), 0, vocab, dtype=tf.int32)
    logits = model(x)

    print(f"Logits shape: {logits.shape}")
    print(f"Params: {model.count_params():,}")

    print("\n--- Causal LM loss ---")
    targets = x[:, 1:]
    logits_shifted = logits[:, :-1, :]
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, logits_shifted, from_logits=True)
    print(f"Loss: {tf.reduce_mean(loss).numpy():.4f}")

    print("\n--- Autoregressive step ---")
    next_logits = logits[0, -1, :]
    next_token = tf.argmax(next_logits).numpy()
    print(f"Next token (greedy): {next_token}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
