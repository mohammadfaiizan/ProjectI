"""
Teacher-student, soft targets, temperature scaling.
"""
import tensorflow as tf

def main():
    print("=" * 50)
    print("Knowledge Distillation")
    print("=" * 50)

    teacher = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    teacher.build((None, 32))
    for w in teacher.weights:
        w.assign(tf.random.normal(w.shape) * 0.1)

    student = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    temperature = 4.0
    alpha = 0.7

    def distillation_loss(y_true, y_student, y_teacher, T):
        soft_teacher = tf.nn.softmax(y_teacher / T)
        soft_student = tf.nn.softmax(y_student / T)
        kl = tf.reduce_mean(
            tf.reduce_sum(soft_teacher * (tf.math.log(soft_teacher + 1e-8) - tf.math.log(soft_student + 1e-8)), axis=1)
        )
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_student)
        return alpha * (T ** 2) * kl + (1 - alpha) * tf.reduce_mean(hard_loss)

    x = tf.random.normal((128, 32))
    y = tf.random.uniform((128,), maxval=10, dtype=tf.int32)
    optimizer = tf.keras.optimizers.Adam(0.001)

    for _ in range(5):
        with tf.GradientTape() as tape:
            logits_teacher = teacher(x, training=False)
            logits_student = student(x, training=True)
            loss = distillation_loss(y, logits_student, logits_teacher, temperature)
        grads = tape.gradient(loss, student.trainable_variables)
        optimizer.apply_gradients(zip(grads, student.trainable_variables))
    print(f"Distillation loss: {loss.numpy():.4f}")

    pred_teacher = tf.argmax(teacher(x[:10]), axis=1)
    pred_student = tf.argmax(student(x[:10]), axis=1)
    agreement = tf.reduce_mean(tf.cast(pred_teacher == pred_student, tf.float32))
    print(f"Teacher-student agreement: {agreement.numpy():.2%}")
    print("Knowledge distillation complete.")

if __name__ == "__main__":
    main()
