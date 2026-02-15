"""
BERT-style model (masked LM, [CLS] token, fine-tuning pattern).
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

def bert_block(x, d_model=128, num_heads=4, ff_dim=256):
    attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn)
    ffn = tf.keras.Sequential([
        tf.keras.layers.Dense(ff_dim, activation="gelu"),
        tf.keras.layers.Dense(d_model)
    ])(x)
    return tf.keras.layers.LayerNormalization()(x + ffn)

def build_bert_encoder(vocab_size=1000, seq_len=64, d_model=128, num_heads=4, ff_dim=256, num_layers=2):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, d_model)(inp)
    x = x + positional_encoding(seq_len, d_model)
    for _ in range(num_layers):
        x = bert_block(x, d_model, num_heads, ff_dim)
    return tf.keras.Model(inp, x)

def build_masked_lm(vocab_size=1000, seq_len=64, d_model=128):
    encoder = build_bert_encoder(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model)
    inp = tf.keras.layers.Input(shape=(seq_len,))
    enc_out = encoder(inp)
    mlm_logits = tf.keras.layers.Dense(vocab_size)(enc_out)
    return tf.keras.Model(inp, mlm_logits)

def build_cls_classifier(vocab_size=1000, seq_len=64, d_model=128, num_classes=2):
    encoder = build_bert_encoder(vocab_size=vocab_size, seq_len=seq_len, d_model=d_model)
    inp = tf.keras.layers.Input(shape=(seq_len,))
    enc_out = encoder(inp)
    cls_out = tf.keras.layers.Lambda(lambda t: t[:, 0, :])(enc_out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(cls_out)
    return tf.keras.Model(inp, out)

def main():
    print("=" * 50)
    print("BERT-style Implementation")
    print("=" * 50)

    batch_size, seq_len, vocab = 4, 32, 500

    print("\n--- Masked LM ---")
    mlm_model = build_masked_lm(vocab_size=vocab, seq_len=seq_len)
    x = tf.random.uniform((batch_size, seq_len), 0, vocab, dtype=tf.int32)
    mlm_logits = mlm_model(x)
    print(f"MLM logits shape: {mlm_logits.shape}")

    print("\n--- [CLS] classification ---")
    cls_model = build_cls_classifier(vocab_size=vocab, seq_len=seq_len, num_classes=2)
    cls_pred = cls_model(x)
    print(f"CLS output shape: {cls_pred.shape}")

    print("\n--- Fine-tuning pattern ---")
    encoder = build_bert_encoder(vocab_size=vocab, seq_len=seq_len)
    enc_out = encoder(x)
    cls_token = enc_out[:, 0, :]
    print(f"CLS token shape: {cls_token.shape}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
