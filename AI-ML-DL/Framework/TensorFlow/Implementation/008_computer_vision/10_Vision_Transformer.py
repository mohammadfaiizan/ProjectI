"""
ViT implementation (patch embedding, position encoding, transformer encoder).
"""
import tensorflow as tf

def patch_embed(x, patch_size=16, embed_dim=768):
    B, H, W, C = tf.shape(x)[0], x.shape[1], x.shape[2], x.shape[3]
    patches = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size)(x)
    patches = tf.reshape(patches, (-1, (H // patch_size) * (W // patch_size), embed_dim))
    return patches

def build_vit(img_size=224, patch_size=16, num_classes=1000, embed_dim=768, num_heads=12, ff_dim=3072, num_layers=12):
    num_patches = (img_size // patch_size) ** 2
    inp = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size)(inp)
    x = tf.keras.layers.Reshape((num_patches, embed_dim))(x)
    cls_token = tf.keras.layers.Dense(embed_dim)(tf.keras.layers.Input(shape=(1,)))
    pos_embed = tf.keras.layers.Embedding(num_patches + 1, embed_dim)(tf.range(num_patches + 1))
    x = x + pos_embed
    for _ in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim)
        ])(x)
        x = tf.keras.layers.LayerNormalization()(x + ffn)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)

def build_vit_simple(img_size=64, patch_size=8, num_classes=10, embed_dim=128, num_heads=4, num_layers=2):
    num_patches = (img_size // patch_size) ** 2
    inp = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size)(inp)
    x = tf.keras.layers.Reshape((num_patches, embed_dim))(x)
    pos_embed = tf.keras.layers.Embedding(num_patches, embed_dim)(tf.range(num_patches))
    x = x + pos_embed
    for _ in range(num_layers):
        attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn)
        ffn = tf.keras.Sequential([tf.keras.layers.Dense(embed_dim * 2, activation='gelu'), tf.keras.layers.Dense(embed_dim)])(x)
        x = tf.keras.layers.LayerNormalization()(x + ffn)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inp, out)

def main():
    model = build_vit_simple()
    x = tf.random.normal((2, 64, 64, 3))
    y = model(x)
    print(f"ViT output shape: {y.shape}")
    print(f"Params: {model.count_params():,}")
    print("Vision Transformer built.")

if __name__ == "__main__":
    main()
