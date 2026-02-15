"""
Embedding layer: input_dim, output_dim, pretrained weights, mask_zero.
"""
import tensorflow as tf

def main():
    vocab_size, embed_dim = 10000, 64
    x = tf.random.uniform((4, 20), minval=0, maxval=vocab_size, dtype=tf.int32)

    emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    out = emb(x)
    print(f"Embedding output: {out.shape}")

    emb_built = tf.keras.layers.Embedding(input_dim=5000, output_dim=32)
    emb_built.build((None, 100))
    print(f"Embedding weights shape: {emb_built.embeddings.shape}")

    emb_mask = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
    out_mask = emb_mask(x)
    print(f"Embedding mask_zero: {out_mask.shape}, compute_mask: {emb_mask.compute_mask(x) is not None}")

    pretrained = tf.random.normal((vocab_size, embed_dim))
    emb_pretrained = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim,
        embeddings_initializer=tf.keras.initializers.Constant(pretrained)
    )
    out_pretrained = emb_pretrained(x)
    print(f"Embedding with initializer: {out_pretrained.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64, input_length=20),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    out_model = model(x)
    print(f"Embedding model output: {out_model.shape}")

    emb_input_length = tf.keras.layers.Embedding(vocab_size, 32, input_length=50)
    x_fixed = tf.random.uniform((2, 50), maxval=vocab_size, dtype=tf.int32)
    out_fixed = emb_input_length(x_fixed)
    print(f"Embedding input_length=50: {out_fixed.shape}")
    print("Embedding layers verified.")

if __name__ == "__main__":
    main()
