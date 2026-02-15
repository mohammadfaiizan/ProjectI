"""
CNN and dense-based text classification.
"""
import tensorflow as tf

def build_cnn_classifier(vocab_size=1000, embed_dim=64, seq_len=100, num_filters=64, kernel_sizes=[3, 4, 5], num_classes=2):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inp)
    conv_outputs = []
    for k in kernel_sizes:
        c = tf.keras.layers.Conv1D(num_filters, k, activation="relu")(x)
        c = tf.keras.layers.GlobalMaxPooling1D()(c)
        conv_outputs.append(c)
    x = tf.keras.layers.Concatenate()(conv_outputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)

def build_dense_classifier(vocab_size=1000, embed_dim=64, seq_len=100, num_classes=2):
    inp = tf.keras.layers.Input(shape=(seq_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_dim)(inp)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inp, out)

def main():
    print("=" * 50)
    print("Text Classification (CNN and Dense)")
    print("=" * 50)

    batch_size, seq_len, vocab_size, num_classes = 8, 50, 1000, 2

    print("\n--- CNN classifier ---")
    cnn_model = build_cnn_classifier(vocab_size=vocab_size, seq_len=seq_len, num_classes=num_classes)
    x = tf.random.uniform((batch_size, seq_len), 0, vocab_size, dtype=tf.int32)
    y_cnn = cnn_model(x)
    print(f"CNN output shape: {y_cnn.shape}")
    print(f"CNN params: {cnn_model.count_params():,}")

    print("\n--- Dense classifier ---")
    dense_model = build_dense_classifier(vocab_size=vocab_size, seq_len=seq_len, num_classes=num_classes)
    y_dense = dense_model(x)
    print(f"Dense output shape: {y_dense.shape}")
    print(f"Dense params: {dense_model.count_params():,}")

    print("\n--- Training step ---")
    labels = tf.random.uniform((batch_size,), 0, num_classes, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels, num_classes)
    with tf.GradientTape() as tape:
        pred = cnn_model(x)
        loss = tf.keras.losses.categorical_crossentropy(labels_onehot, pred)
    grads = tape.gradient(loss, cnn_model.trainable_variables)
    print(f"Loss: {tf.reduce_mean(loss).numpy():.4f}")

    print("\nVerification complete.")

if __name__ == "__main__":
    main()
