"""
tf.keras.layers.Embedding, pretrained embeddings, positional encoding.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Embedding Layers for NLP")
    print("=" * 50)

    vocab_size = 1000
    embed_dim = 64
    seq_len = 10
    batch_size = 4

    print("\n--- Basic Embedding layer ---")
    emb_layer = tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=seq_len)
    x = tf.random.uniform((batch_size, seq_len), 0, vocab_size, dtype=tf.int32)
    out = emb_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    print("\n--- Masking ---")
    emb_masked = tf.keras.layers.Embedding(vocab_size, embed_dim, mask_zero=True)
    x_padded = tf.constant([[1, 2, 0, 0], [3, 0, 0, 0]])
    out_masked = emb_masked(x_padded)
    print(f"Mask: {emb_masked.compute_mask(x_padded)}")

    print("\n--- Positional encoding (sinusoidal) ---")
    def positional_encoding(seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        dim = np.arange(d_model)[np.newaxis, :]
        angle = pos / np.power(10000, 2 * (dim // 2) / d_model)
        angle[:, 0::2] = np.sin(angle[:, 0::2])
        angle[:, 1::2] = np.cos(angle[:, 1::2])
        return tf.constant(angle, dtype=tf.float32)

    pe = positional_encoding(seq_len, embed_dim)
    print(f"Positional encoding shape: {pe.shape}")

    print("\n--- Pretrained embedding (random init demo) ---")
    pretrained = np.random.randn(vocab_size, embed_dim).astype(np.float32)
    emb_pretrained = tf.keras.layers.Embedding(vocab_size, embed_dim, embeddings_initializer=tf.keras.initializers.Constant(pretrained))
    out_pretrained = emb_pretrained(x)
    print(f"Pretrained output shape: {out_pretrained.shape}")

    print("\n--- Embedding + positional add ---")
    emb_out = emb_layer(x)
    pe_batch = tf.broadcast_to(pe, (batch_size, seq_len, embed_dim))
    combined = emb_out + pe_batch
    print(f"Combined shape: {combined.shape}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
