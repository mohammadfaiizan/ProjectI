"""
Gradient accumulation for large effective batch sizes.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Gradient Accumulation")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    micro_batch_size = 8
    accumulation_steps = 4
    effective_batch_size = micro_batch_size * accumulation_steps
    print(f"Micro batch: {micro_batch_size}, accumulation steps: {accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    x = tf.random.normal((128, 16))
    y = tf.random.uniform((128,), maxval=10, dtype=tf.int32)
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(micro_batch_size)

    for epoch in range(2):
        total_loss = 0.0
        accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
        step_count = 0
        for batch_x, batch_y in ds:
            with tf.GradientTape() as tape:
                pred = model(batch_x, training=True)
                loss = loss_fn(batch_y, pred) / accumulation_steps
            grads = tape.gradient(loss, model.trainable_variables)
            accumulated_grads = [g + ag for g, ag in zip(grads, accumulated_grads)]
            step_count += 1
            if step_count % accumulation_steps == 0:
                optimizer.apply_gradients(zip(accumulated_grads, model.trainable_variables))
                accumulated_grads = [tf.zeros_like(v) for v in model.trainable_variables]
            total_loss += tf.reduce_mean(loss).numpy()
        print(f"Epoch {epoch + 1} avg loss: {total_loss / (128 // micro_batch_size):.4f}")

    print("Gradient accumulation complete.")

if __name__ == "__main__":
    main()
