"""
FGSM, PGD attacks, adversarial training loop.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Adversarial Training")
    print("=" * 50)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(28*28,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    def fgsm_attack(model, x, y, eps=0.1):
        with tf.GradientTape() as tape:
            tape.watch(x)
            pred = model(x, training=False)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, pred))
        grad = tape.gradient(loss, x)
        grad = tf.sign(grad)
        return x + eps * grad

    def pgd_attack(model, x, y, eps=0.1, alpha=0.01, steps=5):
        x_adv = tf.identity(x)
        for _ in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                pred = model(x_adv, training=False)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y, pred))
            grad = tape.gradient(loss, x_adv)
            x_adv = x_adv + alpha * tf.sign(grad)
            x_adv = x + tf.clip_by_value(x_adv - x, -eps, eps)
        return x_adv

    x = tf.random.normal((64, 28*28))
    y = tf.random.uniform((64,), maxval=10, dtype=tf.int32)

    x_adv_fgsm = fgsm_attack(model, x, y, eps=0.1)
    x_adv_pgd = pgd_attack(model, x, y, eps=0.1, alpha=0.02, steps=3)
    print("FGSM and PGD attacks generated")

    optimizer = tf.keras.optimizers.Adam(0.001)
    for _ in range(2):
        with tf.GradientTape() as tape:
            pred_clean = model(x, training=True)
            pred_adv = model(x_adv_fgsm, training=True)
            loss_clean = tf.keras.losses.sparse_categorical_crossentropy(y, pred_clean)
            loss_adv = tf.keras.losses.sparse_categorical_crossentropy(y, pred_adv)
            loss = tf.reduce_mean(loss_clean) + 0.5 * tf.reduce_mean(loss_adv)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print(f"Adversarial training loss: {loss.numpy():.4f}")
    print("Adversarial training complete.")

if __name__ == "__main__":
    main()
