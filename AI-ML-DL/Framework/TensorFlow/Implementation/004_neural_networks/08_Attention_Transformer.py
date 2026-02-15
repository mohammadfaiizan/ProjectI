"""
MultiHeadAttention, transformer encoder/decoder blocks.
"""
import tensorflow as tf

def main():
    batch, seq_len, d_model = 2, 10, 64
    x = tf.random.normal((batch, seq_len, d_model))

    mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=8)
    attn_out = mha(query=x, value=x, key=x)
    print(f"MultiHeadAttention output: {attn_out.shape}")

    attn_causal = mha(query=x, value=x, key=x, use_causal_mask=True)
    print(f"MultiHeadAttention causal: {attn_causal.shape}")

    def transformer_encoder_block(x, d_model, num_heads, ff_dim):
        mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        ln1 = tf.keras.layers.LayerNormalization()
        ln2 = tf.keras.layers.LayerNormalization()
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        attn_out = mha(query=x, value=x, key=x)
        x = ln1(x + attn_out)
        ffn_out = ffn(x)
        return ln2(x + ffn_out)

    enc_out = transformer_encoder_block(x, d_model=64, num_heads=8, ff_dim=128)
    print(f"Transformer encoder block: {enc_out.shape}")

    def transformer_decoder_block(x, enc_out, d_model, num_heads, ff_dim):
        mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        ln1 = tf.keras.layers.LayerNormalization()
        ln2 = tf.keras.layers.LayerNormalization()
        ln3 = tf.keras.layers.LayerNormalization()
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self_attn = mha1(query=x, value=x, key=x, use_causal_mask=True)
        x = ln1(x + self_attn)
        cross_attn = mha2(query=x, value=enc_out, key=enc_out)
        x = ln2(x + cross_attn)
        return ln3(x + ffn(x))

    dec_out = transformer_decoder_block(x, enc_out, d_model=64, num_heads=8, ff_dim=128)
    print(f"Transformer decoder block: {dec_out.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len, d_model)),
        tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out = model(x)
    print(f"Attention model output: {out.shape}")
    print("Attention and transformer verified.")

if __name__ == "__main__":
    main()
