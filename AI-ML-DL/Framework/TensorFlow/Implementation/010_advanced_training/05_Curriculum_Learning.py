"""
Difficulty scoring, data scheduling, progressive training.
"""
import tensorflow as tf
import numpy as np

def main():
    print("=" * 50)
    print("Curriculum Learning")
    print("=" * 50)

    def difficulty_score(x, y):
        return tf.reduce_mean(tf.abs(x - tf.cast(y, tf.float32)[:, tf.newaxis]))

    x = tf.random.normal((200, 16))
    y = tf.random.uniform((200,), maxval=10, dtype=tf.int32)
    scores = []
    for i in range(200):
        s = difficulty_score(x[i:i+1], y[i:i+1]).numpy()
        scores.append(s)
    scores = np.array(scores)
    order = np.argsort(scores)
    print(f"Difficulty range: {scores.min():.4f} to {scores.max():.4f}")

    x_sorted = tf.gather(x, order)
    y_sorted = tf.gather(y, order)
    ds = tf.data.Dataset.from_tensor_slices((x_sorted, y_sorted)).batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    curriculum_epochs = 3
    for phase in range(curriculum_epochs):
        frac = (phase + 1) / curriculum_epochs
        n_samples = int(200 * frac)
        ds_phase = tf.data.Dataset.from_tensor_slices(
            (x_sorted[:n_samples], y_sorted[:n_samples])
        ).batch(32)
        model.fit(ds_phase, epochs=1, verbose=0)
        print(f"Phase {phase + 1}: trained on {n_samples} easy-to-hard samples")

    pred = model.predict(x[:10], verbose=0)
    acc = np.mean(np.argmax(pred, axis=1) == y[:10].numpy())
    print(f"Sample accuracy: {acc:.2%}")
    print("Curriculum learning complete.")

if __name__ == "__main__":
    main()
