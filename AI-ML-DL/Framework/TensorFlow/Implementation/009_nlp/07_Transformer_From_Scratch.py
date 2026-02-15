"""
Full transformer (encoder + decoder blocks, multi-head attention, positional encoding).
"""
import tensorflow as tf
import numpy as np

def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis].astype(np.float32)
    dim = np.arange(d_model)[np.newaxis, :].astype(np.float32)
    angle = pos / np.power(10000, 2 * (dim // 2) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return tf.constant(angle)

def transformer_block(x, d_model=128, num_heads=4, ff_dim=256):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(d_model)
    ])(x)
    return tf.keras.layers.LayerNormalization()(x + ffn)

def decoder_block(x, enc_out, d_model=128, num_heads=4, ff_dim=256):
    attn1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn1)
    attn2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, enc_out)
    x = tf.keras.layers.LayerNormalization()(x + attn2)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="relu"),
        tf.keras.layers.Dense(d_model)
    ])(x)
    return tf.keras.layers.LayerNormalization()(x + ffn)

def build_transformer(vocab_size=1000, d_model=128, num_heads=4, ff_dim=256, num_enc=2, num_dec=2, seq_len=50):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inp)
    x = x + positional_encoding(seq_len, d_model)
    for _ in range(num_enc):
        x = transformer_block(x, d_model, num_heads, ff_dim)
    enc_out = x
    dec_inp = tf.keras.layers.Input(shape=(seq_len,))
    y = tf.keras.layers.Embedding(vocab_size, d_model)(dec_inp)
    y = y + positional_encoding(seq_len, d_model)
    for _ in range(num_dec):
        y = decoder_block(y, enc_out, d_model, num_heads, ff_dim)
    out = tf.keras.layers.Dense(vocab_size, activation="softmax")(y)
    return tf.keras.Model([inp, dec_inp], out)

def main():
    print("=" * 50)
    print("Transformer from Scratch")
    print("=" * 50)

    batch_size, seq_len, vocab = 4, 20, 500
    model = build_transformer(vocab_size=vocab, seq_len=seq_len)

    enc_inp = tf.random.uniform((batch_size, seq_len), 0, vocab, dtype=tf.int32)
    dec_inp = tf.random.uniform((batch_size, seq_len), 0, vocab, dtype=tf.int32)

    out = model([enc_inp, dec_inp])
    print(f"Output shape: {out.shape}")
    print(f"Params: {model.count_params():,}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
