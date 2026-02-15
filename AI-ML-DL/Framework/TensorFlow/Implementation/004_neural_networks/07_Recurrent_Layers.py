"""
SimpleRNN, LSTM, GRU, Bidirectional, return_sequences, return_state.
"""
import tensorflow as tf

def main():
    x = tf.random.normal((2, 10, 32))

    rnn = tf.keras.layers.SimpleRNN(64, return_sequences=False)
    out = rnn(x)
    print(f"SimpleRNN output: {out.shape}")

    rnn_seq = tf.keras.layers.SimpleRNN(64, return_sequences=True)
    out_seq = rnn_seq(x)
    print(f"SimpleRNN return_sequences: {out_seq.shape}")

    lstm = tf.keras.layers.LSTM(64, return_sequences=True, return_state=True)
    out_lstm, h, c = lstm(x)
    print(f"LSTM output: {out_lstm.shape}, h: {h.shape}, c: {c.shape}")

    gru = tf.keras.layers.GRU(64, return_sequences=False)
    out_gru = gru(x)
    print(f"GRU output: {out_gru.shape}")

    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
    out_bi = bi_lstm(x)
    print(f"Bidirectional LSTM: {out_bi.shape}")

    bi_gru_seq = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, return_sequences=True))
    out_bi_seq = bi_gru_seq(x)
    print(f"Bidirectional GRU return_sequences: {out_bi_seq.shape}")

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, 32)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    out_model = model(x)
    print(f"Stacked LSTM model output: {out_model.shape}")

    lstm_state = tf.keras.layers.LSTM(32, return_state=True)
    _, h, c = lstm_state(x)
    print(f"LSTM state h shape: {h.shape}")
    print("Recurrent layers verified.")

if __name__ == "__main__":
    main()
