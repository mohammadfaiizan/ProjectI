"""
Attention for vision (SE block, CBAM, self-attention).
"""
import tensorflow as tf

def se_block(x, ratio=16):
    channels = x.shape[-1]
    squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
    excite = tf.keras.layers.Dense(channels // ratio, activation='relu')(squeeze)
    excite = tf.keras.layers.Dense(channels, activation='sigmoid')(excite)
    return x * tf.reshape(excite, (-1, 1, 1, channels))

def channel_attention(x, ratio=8):
    channels = x.shape[-1]
    avg = tf.keras.layers.GlobalAveragePooling2D()(x)
    max_p = tf.keras.layers.GlobalMaxPooling2D()(x)
    shared = tf.keras.Sequential([
        tf.keras.layers.Dense(channels // ratio, activation='relu'),
        tf.keras.layers.Dense(channels)
    ])
    ca = tf.keras.activations.sigmoid(shared(avg) + shared(max_p))
    return x * tf.reshape(ca, (-1, 1, 1, channels))

def spatial_attention(x):
    avg = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_p = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = tf.keras.layers.Concatenate()([avg, max_p])
    sa = tf.keras.layers.Conv2D(1, 7, padding='same', activation='sigmoid')(concat)
    return x * sa

def cbam_block(x, ratio=8):
    x = channel_attention(x, ratio)
    x = spatial_attention(x)
    return x

def self_attention_2d(x):
    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x_flat = tf.reshape(x, (-1, H * W, C))
    key_dim = max(1, C // 4)
    attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=key_dim)(x_flat, x_flat)
    return tf.reshape(attn, (-1, H, W, C))

def main():
    x = tf.random.normal((2, 28, 28, 64))
    se_out = se_block(x, ratio=8)
    print(f"SE block output: {se_out.shape}")
    cbam_out = cbam_block(x, ratio=8)
    print(f"CBAM output: {cbam_out.shape}")
    attn_out = self_attention_2d(x)
    print(f"Self-attention output: {attn_out.shape}")
    def add_cbam(x):
        return cbam_block(x, 8)
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 3, padding='same', input_shape=(28, 28, 3)),
        tf.keras.layers.Lambda(add_cbam),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out = model(tf.random.normal((2, 28, 28, 3)))
    print(f"CBAM model output: {out.shape}")
    print("Attention mechanisms verified.")

if __name__ == "__main__":
    main()
