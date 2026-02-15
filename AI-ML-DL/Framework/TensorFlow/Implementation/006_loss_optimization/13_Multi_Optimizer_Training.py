"""
Different optimizers for different model parts (GAN-style).
"""
import tensorflow as tf

def main():
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='sigmoid')
    ])
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    opt_g = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt_d = tf.keras.optimizers.Adam(learning_rate=0.0002)

    bce = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(real_samples, noise):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake = generator(noise, training=True)
            real_out = discriminator(real_samples, training=True)
            fake_out = discriminator(fake, training=True)
            real_labels = tf.ones_like(real_out)
            fake_labels = tf.zeros_like(fake_out)
            loss_d_real = bce(real_labels, real_out)
            loss_d_fake = bce(fake_labels, fake_out)
            loss_d = loss_d_real + loss_d_fake
            gen_labels = tf.ones_like(fake_out)
            loss_g = bce(gen_labels, fake_out)

        grad_g = gen_tape.gradient(loss_g, generator.trainable_variables)
        grad_d = disc_tape.gradient(loss_d, discriminator.trainable_variables)
        opt_g.apply_gradients(zip(grad_g, generator.trainable_variables))
        opt_d.apply_gradients(zip(grad_d, discriminator.trainable_variables))
        return loss_g, loss_d

    real = tf.random.normal((16, 8))
    noise = tf.random.normal((16, 8))
    for _ in range(3):
        lg, ld = train_step(real, noise)
    print(f"GAN step - gen loss: {lg.numpy():.4f}, disc loss: {ld.numpy():.4f}")

    class MultiOptModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
                tf.keras.layers.Dense(8)
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(16, activation='relu', input_shape=(8,)),
                tf.keras.layers.Dense(4)
            ])
            self.opt_enc = tf.keras.optimizers.Adam(0.001)
            self.opt_dec = tf.keras.optimizers.SGD(0.01, momentum=0.9)

        def call(self, x):
            z = self.encoder(x)
            return self.decoder(z)

        def train_step_custom(self, data):
            x, y = data
            with tf.GradientTape() as tape:
                pred = self(x, training=True)
                loss = tf.reduce_mean(tf.square(y - pred))
            grads = tape.gradient(loss, self.trainable_variables)
            enc_vars = self.encoder.trainable_variables
            dec_vars = self.decoder.trainable_variables
            enc_grads = [g for g, v in zip(grads, self.trainable_variables) if v in enc_vars]
            dec_grads = [g for g, v in zip(grads, self.trainable_variables) if v in dec_vars]
            self.opt_enc.apply_gradients(zip(enc_grads, enc_vars))
            self.opt_dec.apply_gradients(zip(dec_grads, dec_vars))
            return {"loss": loss}

    multi_model = MultiOptModel()
    multi_model.compile()
    x_m = tf.random.normal((8, 4))
    y_m = tf.random.normal((8, 4))
    multi_model.train_step_custom((x_m, y_m))
    print(f"Multi-optimizer model step completed.")
    print("Multi-optimizer training verified.")

if __name__ == "__main__":
    main()
