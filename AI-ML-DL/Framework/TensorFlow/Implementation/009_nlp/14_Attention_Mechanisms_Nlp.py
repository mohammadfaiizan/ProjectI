"""
Bahdanau, Luong, scaled dot-product, multi-head attention.
"""
import tensorflow as tf
import math

def bahdanau_attention(query, values):
    query_exp = tf.expand_dims(query, 1)
    query_tiled = tf.tile(query_exp, [1, tf.shape(values)[1], 1])
    concat = tf.concat([query_tiled, values], axis=-1)
    dense1 = tf.keras.layers.Dense(values.shape[-1], activation="tanh")(concat)
    score = tf.keras.layers.Dense(1)(dense1)
    attn_weights = tf.nn.softmax(score, axis=1)
    context = tf.reduce_sum(attn_weights * values, axis=1)
    return context, attn_weights

def luong_attention(query, values):
    query_exp = tf.expand_dims(query, 1)
    score = tf.matmul(query_exp, values, transpose_b=True)
    attn_weights = tf.nn.softmax(score, axis=-1)
    context = tf.matmul(attn_weights, values)
    context = tf.squeeze(context, 1)
    return context, attn_weights

def scaled_dot_product_attention(q, k, v, mask=None):
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(dk)
    if mask is not None:
        scores += mask * -1e9
    attn_weights = tf.nn.softmax(scores, axis=-1)
    output = tf.matmul(attn_weights, v)
    return output, attn_weights

def main():
    print("=" * 50)
    print("Attention Mechanisms for NLP")
    print("=" * 50)

    batch_size, seq_len, dim = 4, 10, 64

    query = tf.random.normal((batch_size, dim))
    values = tf.random.normal((batch_size, seq_len, dim))

    print("\n--- Bahdanau (additive) ---")
    context_b, attn_b = bahdanau_attention(query, values)
    print(f"Context shape: {context_b.shape}")
    print(f"Attention weights shape: {attn_b.shape}")

    print("\n--- Luong (dot) ---")
    context_l, attn_l = luong_attention(query, values)
    print(f"Context shape: {context_l.shape}")

    print("\n--- Scaled dot-product ---")
    q = tf.random.normal((batch_size, 4, dim // 4))
    k = tf.random.normal((batch_size, seq_len, dim // 4))
    v = tf.random.normal((batch_size, seq_len, dim // 4))
    out, attn_s = scaled_dot_product_attention(q, k, v)
    print(f"Output shape: {out.shape}")

    print("\n--- Multi-head attention (Keras) ---")
    mha = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
    x = tf.random.normal((batch_size, seq_len, dim))
    mha_out = mha(x, x)
    print(f"MHA output shape: {mha_out.shape}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
