"""
Neural machine translation with transformer.
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

def build_nmt_transformer(src_vocab=1000, tgt_vocab=1000, seq_len=40, d_model=128, num_heads=4, ff_dim=256, num_layers=2):
    enc_inp = tf.keras.layers.Input(shape=(seq_len,))
    dec_inp = tf.keras.layers.Input(shape=(seq_len,))

    enc_emb = tf.keras.layers.Embedding(src_vocab, d_model)(enc_inp) + positional_encoding(seq_len, d_model)
    for _ in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(enc_emb, enc_emb)
        enc_emb = tf.keras.layers.LayerNormalization()(enc_emb + attn)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])(enc_emb)
        enc_emb = tf.keras.layers.LayerNormalization()(enc_emb + ffn)

    dec_emb = tf.keras.layers.Embedding(tgt_vocab, d_model)(dec_inp) + positional_encoding(seq_len, d_model)
    causal_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    causal_mask = (1 - causal_mask) * -1e9
    for _ in range(num_layers):
        self_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
            dec_emb, dec_emb, attention_mask=causal_mask
        )
        dec_emb = tf.keras.layers.LayerNormalization()(dec_emb + self_attn)
        cross_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(
            dec_emb, enc_emb
        )
        dec_emb = tf.keras.layers.LayerNormalization()(dec_emb + cross_attn)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(d_model)
        ])(dec_emb)
        dec_emb = tf.keras.layers.LayerNormalization()(dec_emb + ffn)

    logits = tf.keras.layers.Dense(tgt_vocab)(dec_emb)
    return tf.keras.Model([enc_inp, dec_inp], logits)

def main():
    print("=" * 50)
    print("Neural Machine Translation (Transformer)")
    print("=" * 50)

    batch_size, seq_len = 4, 25
    src_vocab, tgt_vocab = 800, 800

    model = build_nmt_transformer(src_vocab=src_vocab, tgt_vocab=tgt_vocab, seq_len=seq_len)

    src = tf.random.uniform((batch_size, seq_len), 0, src_vocab, dtype=tf.int32)
    tgt = tf.random.uniform((batch_size, seq_len), 0, tgt_vocab, dtype=tf.int32)

    logits = model([src, tgt])
    print(f"Logits shape: {logits.shape}")
    print(f"Params: {model.count_params():,}")

    print("\n--- Translation loss ---")
    targets = tgt[:, 1:]
    pred_shifted = logits[:, :-1, :]
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, pred_shifted, from_logits=True)
    print(f"Loss: {tf.reduce_mean(loss).numpy():.4f}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
