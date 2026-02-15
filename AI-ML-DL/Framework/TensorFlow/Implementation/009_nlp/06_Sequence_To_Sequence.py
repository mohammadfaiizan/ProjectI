"""
Encoder-decoder seq2seq with attention.
"""
import tensorflow as tf

def build_seq2seq_attention(enc_vocab=1000, dec_vocab=1000, embed_dim=64, units=128, max_enc_len=50, max_dec_len=50):
    enc_inp = tf.keras.layers.Input(shape=(max_enc_len,))
    dec_inp = tf.keras.layers.Input(shape=(max_dec_len,))

    enc_emb = tf.keras.layers.Embedding(enc_vocab, embed_dim, mask_zero=True)(enc_inp)
    enc_out, enc_h, enc_c = tf.keras.layers.LSTM(units, return_sequences=True, return_state=True)(enc_emb)

    dec_emb = tf.keras.layers.Embedding(dec_vocab, embed_dim, mask_zero=True)(dec_inp)
    attention = tf.keras.layers.Attention()([dec_emb, enc_out])
    dec_concat = tf.keras.layers.Concatenate()([dec_emb, attention])
    dec_lstm = tf.keras.layers.LSTM(units, return_sequences=True)(dec_concat, initial_state=[enc_h, enc_c])
    dec_dense = tf.keras.layers.Dense(dec_vocab, activation="softmax")(dec_lstm)

    return tf.keras.Model([enc_inp, dec_inp], dec_dense)

def main():
    print("=" * 50)
    print("Sequence-to-Sequence with Attention")
    print("=" * 50)

    batch_size, enc_len, dec_len, vocab = 4, 20, 15, 500

    model = build_seq2seq_attention(enc_vocab=vocab, dec_vocab=vocab, max_enc_len=enc_len, max_dec_len=dec_len)

    enc_inp = tf.random.uniform((batch_size, enc_len), 0, vocab, dtype=tf.int32)
    dec_inp = tf.random.uniform((batch_size, dec_len), 0, vocab, dtype=tf.int32)

    out = model([enc_inp, dec_inp])
    print(f"Output shape: {out.shape}")
    print(f"Params: {model.count_params():,}")

    print("\n--- Teacher forcing step ---")
    targets = dec_inp[:, 1:]
    pred_shifted = out[:, :-1, :]
    loss = tf.keras.losses.sparse_categorical_crossentropy(targets, pred_shifted)
    print(f"Seq2seq loss: {tf.reduce_mean(loss).numpy():.4f}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
